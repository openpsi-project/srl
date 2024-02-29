from torch.distributions import Categorical
from typing import Dict, Union
import copy
import itertools
import math
import numpy as np
import torch
import torch.nn as nn

from api.trainer import SampleBatch
from base.namedarray import recursive_apply, NamedArray, recursive_aggregate
from legacy.algorithm.ppo.actor_critic_policies.actor_critic_policy import PPORolloutAnalyzedResult
from legacy.algorithm.ppo.mappo import SampleAnalyzedResult
from legacy.environment.smac.smac_env import SMACAction, get_smac_shapes
import api.policy
import legacy.algorithm.modules as modules


class SMACPolicyState(NamedArray):

    def __init__(
        self,
        actor_hx: np.ndarray,
        critic_hx: np.ndarray,
    ):
        super(SMACPolicyState, self).__init__(actor_hx=actor_hx, critic_hx=critic_hx)


class SMACAgentwiseEncoder(nn.Module):

    def __init__(self, self_dim, others_shapes, hidden_dim):
        super().__init__()
        self.others_shapes = others_shapes
        self.embedding = modules.CatSelfEmbedding(self_dim, self.others_shapes, hidden_dim // 2)
        self.attn = modules.MultiHeadSelfAttention(hidden_dim // 2, hidden_dim // 2, 4)
        self.dense = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(inplace=True), nn.LayerNorm(hidden_dim))

    def forward(self, x_self, x, mask):
        x_self, x_others = self.embedding(x_self, **{k: v for k, v in x.items() if k in self.others_shapes})
        attn_other = self.attn(x_others, mask)
        pooled_attn_other = modules.masked_avg_pooling(attn_other, mask)
        x = torch.cat([x_self, pooled_attn_other], dim=-1)
        return self.dense(x)


class SMACAgentwiseObsEncoder(nn.Module):

    def __init__(self, obs_shapes, hidden_dim):
        super().__init__()
        self.normalize_keys = ['obs_self', 'obs_move', 'obs_allies', 'obs_enemies']
        for k in self.normalize_keys:
            if k in obs_shapes:
                setattr(self, k + '_norm', nn.LayerNorm([obs_shapes[k][-1]]))
        others_shapes = copy.deepcopy(obs_shapes)
        others_shapes.pop('obs_self')
        others_shapes.pop('obs_mask')
        self.encoder = SMACAgentwiseEncoder(obs_shapes['obs_self'][0], others_shapes, hidden_dim)

    def forward(self, local_obs):
        for k in self.normalize_keys:
            if k in local_obs.keys():
                local_obs[k] = getattr(self, k + '_norm')(local_obs[k])
        return self.encoder(local_obs.obs_self, local_obs, local_obs.obs_mask)


class SMACAgentwiseStateEncoder(nn.Module):

    def __init__(self, state_shapes, hidden_dim):
        super().__init__()
        self.normalize_keys = ['state_self', 'state_move', 'state_allies', 'state_enemies']
        for k in self.normalize_keys:
            if k in state_shapes:
                setattr(self, k + '_norm', nn.LayerNorm([state_shapes[k][-1]]))
        others_shapes = copy.deepcopy(state_shapes)
        others_shapes.pop('state_self')
        others_shapes.pop('state_mask')
        self.encoder = SMACAgentwiseEncoder(state_shapes['state_self'][0], others_shapes, hidden_dim)

    def forward(self, state):
        for k in self.normalize_keys:
            if k in state.keys():
                state[k] = getattr(self, k + '_norm')(state[k])
        return self.encoder(state.state_self, state, state.state_mask)


class SMACNet(nn.Module):

    def __init__(
        self,
        obs_shape: Union[tuple, Dict],
        state_shape: Union[tuple, Dict],
        hidden_dim: int,
        act_dim: int,
        agent_specific_obs: bool = False,
        agent_specific_state: bool = False,
        act_init_gain: float = 0.01,
        num_rnn_layers: int = 1,
        unbiased_popart: bool = False,
        popart_beta: float = 1 - 1e-5,
    ):
        super().__init__()

        if agent_specific_obs:
            assert isinstance(obs_shape, Dict), obs_shape
            self.actor_base = SMACAgentwiseObsEncoder(obs_shape, hidden_dim)
        else:
            self.actor_base = nn.Sequential(
                nn.LayerNorm([obs_shape[0]]),
                modules.mlp([obs_shape[0], hidden_dim, hidden_dim], nn.ReLU, layernorm=True))

        if agent_specific_state:
            assert isinstance(state_shape, Dict), state_shape
            self.critic_base = SMACAgentwiseStateEncoder(state_shape, hidden_dim)
        else:
            self.critic_base = nn.Sequential(
                nn.LayerNorm([state_shape[0]]),
                modules.mlp([state_shape[0], hidden_dim, hidden_dim], nn.ReLU, layernorm=True))

        self.num_rnn_layers = num_rnn_layers
        if num_rnn_layers > 0:
            self.actor_rnn = modules.AutoResetRNN(hidden_dim, hidden_dim, num_layers=num_rnn_layers)
            self.critic_rnn = modules.AutoResetRNN(hidden_dim, hidden_dim, num_layers=num_rnn_layers)
            self.actor_rnn_norm = nn.LayerNorm([hidden_dim])
            self.critic_rnn_norm = nn.LayerNorm([hidden_dim])

        self.policy_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = modules.PopArtValueHead(
            input_dim=hidden_dim,
            critic_dim=1,
            beta=popart_beta,
            burn_in_updates=1000 if unbiased_popart else torch.inf,
        )

        # initialization
        for k, p in itertools.chain(self.actor_base.named_parameters(), self.critic_base.named_parameters()):
            if 'weight' in k and len(p.data.shape) >= 2:
                # filter out layer norm weights
                nn.init.orthogonal_(p.data, gain=math.sqrt(2))
            if 'bias' in k:
                nn.init.zeros_(p.data)

        if num_rnn_layers > 0:
            for k, p in itertools.chain(self.actor_rnn.named_parameters(),
                                        self.critic_rnn.named_parameters()):
                if 'weight' in k:
                    nn.init.orthogonal_(p.data, gain=1)
                if 'bias' in k:
                    nn.init.zeros_(p.data)

        # policy head should have a smaller scale
        nn.init.orthogonal_(self.policy_head.weight.data, gain=act_init_gain)
        nn.init.zeros_(self.policy_head.bias.data)

    def forward(self, local_obs, state, available_action, actor_hx, critic_hx, on_reset=None):
        actor_features = self.actor_base(local_obs)
        critic_features = self.critic_base(state)

        if self.num_rnn_layers > 0:
            actor_features, actor_hx = self.actor_rnn(actor_features, actor_hx, on_reset)
            critic_features, critic_hx = self.critic_rnn(critic_features, critic_hx, on_reset)
            actor_features = self.actor_rnn_norm(actor_features)
            critic_features = self.critic_rnn_norm(critic_features)

        logits = self.policy_head(actor_features)
        logits[available_action == 0] = -1e10

        return Categorical(logits=logits), self.value_head(critic_features), actor_hx, critic_hx


class SMACPolicy(api.policy.SingleModelPytorchPolicy):

    @property
    def default_policy_state(self):
        return SMACPolicyState(self.__rnn_default_hidden, self.__rnn_default_hidden)

    def __init__(
        self,
        map_name: str,
        hidden_dim: int,
        chunk_len: int,
        seed: int = 0,
        shared: bool = False,
        agent_specific_obs: bool = False,
        agent_specific_state: bool = False,
        act_init_gain: float = 0.01,
        num_rnn_layers: int = 1,
        denormalize_value_during_rollout: bool = False,
        popart: bool = True,
        unbiased_popart: bool = False,
        popart_beta: float = 1 - 1e-5,
    ):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        obs_shape, state_shape, act_dim, n_agents = get_smac_shapes(map_name,
                                                                    agent_specific_obs=agent_specific_obs,
                                                                    agent_specific_state=agent_specific_state)
        self.__act_dim = act_dim
        self.__rnn_hidden_dim = hidden_dim
        self.__chunk_len = chunk_len
        self.__denormalize_value_during_rollout = denormalize_value_during_rollout
        self.__popart = popart
        self.__shared = shared
        if shared:
            self.__rnn_default_hidden = np.zeros((n_agents, 1, self.__rnn_hidden_dim), dtype=np.float32)
        else:
            self.__rnn_default_hidden = np.zeros((1, self.__rnn_hidden_dim), dtype=np.float32)
        neural_network = SMACNet(
            obs_shape=obs_shape,
            state_shape=state_shape,
            hidden_dim=hidden_dim,
            act_dim=act_dim,
            agent_specific_obs=agent_specific_obs,
            agent_specific_state=agent_specific_state,
            act_init_gain=act_init_gain,
            num_rnn_layers=num_rnn_layers,
            unbiased_popart=unbiased_popart,
            popart_beta=popart_beta,
        )
        super(SMACPolicy, self).__init__(neural_network)

    @property
    def module(self):
        return self.net.module if hasattr(self.net, "module") else self.net

    @property
    def denormalize_value_during_rollout(self):
        return self.__denormalize_value_during_rollout

    def normalize_value(self, x):
        return self.module.value_head.normalize(x)

    def denormalize_value(self, x):
        return self.module.value_head.denormalize(x)

    def update_popart(self, x, mask):
        return self.module.value_head.update(x, mask)

    def analyze(self, sample: SampleBatch, burn_in_steps=0, **kwargs) -> SampleAnalyzedResult:
        """ Generate outputs required for loss computation during training,
            e.g. value target and action distribution entropies. Typically,
            data has a shape of [T, B, *D] and RNN states have a shape of
            [num_layers, B, hidden_dim].
        Args:
            sample (SampleBatch): Arrays of (obs, action ...) containing
                all data required for loss computation.
        Returns:
            analyzed_result[SampleAnalyzedResult]: Data generated for loss computation.
        """
        sample_steps = sample.on_reset.shape[0] - burn_in_steps
        num_chunks = sample_steps // self.__chunk_len
        bs = sample.on_reset.shape[1]
        if self.__shared:

            def _merge_batch_agent(x):
                return x.view(x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:])

            observation = recursive_apply(sample.obs[burn_in_steps:],
                                          lambda x: modules.to_chunk(_merge_batch_agent(x), num_chunks))
            # when one step is done, rnn states of the NEXT step should be reset
            on_reset = recursive_apply(sample.on_reset[burn_in_steps:],
                                       lambda x: modules.to_chunk(_merge_batch_agent(x), num_chunks))
            action = recursive_apply(sample.action[burn_in_steps:],
                                     lambda x: modules.to_chunk(_merge_batch_agent(x), num_chunks))
            policy_state = recursive_apply(sample.policy_state,
                                           lambda x: modules.to_chunk(_merge_batch_agent(x), num_chunks))
        else:
            observation = recursive_apply(sample.obs[burn_in_steps:],
                                          lambda x: modules.to_chunk(x, num_chunks))
            # when one step is done, rnn states of the NEXT step should be reset
            on_reset = recursive_apply(sample.on_reset[burn_in_steps:],
                                       lambda x: modules.to_chunk(x, num_chunks))
            action = recursive_apply(sample.action[burn_in_steps:], lambda x: modules.to_chunk(x, num_chunks))
            policy_state = recursive_apply(sample.policy_state, lambda x: modules.to_chunk(x, num_chunks))

        actor_hx = critic_hx = None
        if burn_in_steps == 0:
            actor_hx = policy_state.actor_hx[0].transpose(0, 1)
            critic_hx = policy_state.critic_hx[0].transpose(0, 1)
        else:

            def _make_burn_in_data(x):
                xs = []
                for i in range(num_chunks):
                    xs.append(x[i * self.__chunk_len:i * self.__chunk_len + burn_in_steps])
                return recursive_aggregate(xs, lambda x: torch.cat(x, dim=1))

            if self.__shared:
                burn_in_obs = recursive_apply(sample.obs, lambda x: _make_burn_in_data(_merge_batch_agent(x)))
                burn_in_policy_state = recursive_apply(sample.policy_state,
                                                       lambda x: _make_burn_in_data(_merge_batch_agent(x)))
                burn_in_on_reset = recursive_apply(sample.on_reset,
                                                   lambda x: _make_burn_in_data(_merge_batch_agent(x)))
            else:
                burn_in_obs = recursive_apply(sample.obs, _make_burn_in_data)
                burn_in_policy_state = recursive_apply(sample.policy_state, _make_burn_in_data)
                burn_in_on_reset = recursive_apply(sample.on_reset, _make_burn_in_data)
            with torch.no_grad():
                _, _, actor_hx, critic_hx = self.net(burn_in_obs.local_obs, burn_in_obs.state,
                                                     burn_in_obs.available_action,
                                                     burn_in_policy_state.actor_hx[0].transpose(0, 1),
                                                     burn_in_policy_state.critic_hx[0].transpose(0, 1),
                                                     burn_in_on_reset)

        action_distribution, state_values, _, _ = self.net(observation.local_obs, observation.state,
                                                           observation.available_action, actor_hx, critic_hx,
                                                           on_reset)

        old_log_probs = sample.analyzed_result.log_probs[burn_in_steps:]
        if self.__shared:
            new_log_probs = action_distribution.log_prob(action.x.squeeze(-1)).unsqueeze(-1)
            # Don't backpropagate policy loss for death steps.
            new_log_probs[observation.is_alive == 0] = -torch.inf

            def _split_batch_agent(x):
                return x.view(x.shape[0], bs, x.shape[1] // bs, *x.shape[2:])

            new_log_probs = _split_batch_agent(modules.back_to_trajectory(new_log_probs, num_chunks))
            entropy = _split_batch_agent(modules.back_to_trajectory(action_distribution.entropy(),
                                                                    num_chunks))
            state_values = _split_batch_agent(modules.back_to_trajectory(state_values, num_chunks))
        else:
            new_log_probs = modules.back_to_trajectory(
                action_distribution.log_prob(action.x.squeeze(-1)).unsqueeze(-1), num_chunks)
            entropy = modules.back_to_trajectory(action_distribution.entropy(), num_chunks)
            state_values = modules.back_to_trajectory(state_values, num_chunks)

        analyzed_result = SampleAnalyzedResult(old_action_log_probs=old_log_probs,
                                               new_action_log_probs=new_log_probs,
                                               state_values=state_values,
                                               entropy=entropy.unsqueeze(-1))

        return analyzed_result

    def rollout(self, requests: api.policy.RolloutRequest, **kwargs):
        """ Provide inference results for actor workers. Typically,
            data and masks have a shape of [batch_size, *D], and RNN states
            have a shape of [batch_size, num_layers, hidden_dim].
        Args:
            requests (RolloutRequest): Observations, policy states,
                evaluation masks and reset masks. 
        Returns:
            RolloutResult: Actions and new policy states, optionally
                with other entries depending on the algorithm.
        """
        requests = recursive_apply(requests,
                                   lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=self.device))
        bs = requests.length(dim=0)
        if self.__shared:
            requests = recursive_apply(requests, lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]))
        actor_hx = critic_hx = torch.from_numpy(np.stack([self.__rnn_default_hidden for _ in range(bs)],
                                                         0)).to(dtype=torch.float32, device=self.device)
        if self.__shared:
            actor_hx = actor_hx.view(actor_hx.shape[0] * actor_hx.shape[1], *actor_hx.shape[2:])
            critic_hx = critic_hx.view(critic_hx.shape[0] * critic_hx.shape[1], *critic_hx.shape[2:])
        if requests.policy_state is not None:
            reset_mask = requests.on_reset.unsqueeze(1)
            actor_hx = reset_mask * actor_hx + (1 - reset_mask) * requests.policy_state.actor_hx
            critic_hx = reset_mask * critic_hx + (1 - reset_mask) * requests.policy_state.critic_hx
        actor_hx = actor_hx.transpose(0, 1)
        critic_hx = critic_hx.transpose(0, 1)

        with torch.no_grad():
            local_obs = recursive_apply(requests.obs.local_obs, lambda x: x.unsqueeze(0))
            state = recursive_apply(requests.obs.state, lambda x: x.unsqueeze(0))
            action_distribution, value, actor_hx, critic_hx = self.net(
                local_obs, state, requests.obs.available_action.unsqueeze(0), actor_hx, critic_hx)
            # .squeeze(0) removes the time dimension
            value = value.squeeze(0)
            deterministic_actions = action_distribution.probs.argmax(dim=-1).squeeze(0)
            stochastic_actions = action_distribution.sample().squeeze(0)
            # now deterministic/stochastic actions have shape [batch_size]
            eval_mask = requests.is_evaluation.squeeze(-1)
            actions = eval_mask * deterministic_actions + (1 - eval_mask) * stochastic_actions
            log_probs = action_distribution.log_prob(actions).squeeze(0)
            if self.__popart and self.__denormalize_value_during_rollout:
                value = self.denormalize_value(value)

        # .unsqueeze(-1) adds a trailing dimension 1
        rollout_result = api.policy.RolloutResult(
            action=SMACAction(actions.unsqueeze(-1).cpu()),
            analyzed_result=PPORolloutAnalyzedResult(log_probs=log_probs.unsqueeze(-1).cpu(),
                                                     value=value.cpu()),
            policy_state=SMACPolicyState(actor_hx.transpose(0, 1).cpu(),
                                         critic_hx.transpose(0, 1).cpu()),
        )
        if self.__shared:
            rollout_result = recursive_apply(rollout_result,
                                             lambda x: x.view(bs, x.shape[0] // bs, *x.shape[1:]))
        return recursive_apply(rollout_result, lambda x: x.numpy())

    def load_checkpoint(self, checkpoint):
        self._version = checkpoint.get("steps", 0)
        self.set_state_dict(checkpoint["state_dict"])

    def set_state_dict(self, state_dict):
        # loading partial state dict is allowed
        cur_state_dict = self.net.state_dict()
        for k in state_dict:
            assert k in cur_state_dict, k
        state_dict_ = {}
        for k, v in cur_state_dict.items():
            if k in state_dict:
                state_dict_[k] = state_dict[k]
            else:
                state_dict_[k] = cur_state_dict[k]
        self.net.load_state_dict(state_dict_)


api.policy.register("smac_rnn", SMACPolicy)
