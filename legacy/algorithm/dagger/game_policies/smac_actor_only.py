from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from legacy.algorithm.ppo.game_policies.smac_rnn import SMACPolicyState, SMACAgentwiseObsEncoder
from api.policy import Policy, register
from api.trainer import SampleBatch
from base.namedarray import recursive_apply
from legacy.environment.smac.smac_env import get_smac_shapes
import legacy.algorithm.modules as modules


class SMACActorOnlyNet(nn.Module):

    def __init__(self,
                 obs_shape: Union[tuple, Dict],
                 hidden_dim: int,
                 act_dim: int,
                 shared: bool = False,
                 agent_specific_obs: bool = False,
                 agent_specific_state: bool = False):
        super().__init__()
        self.shared = shared

        if agent_specific_obs:
            assert isinstance(obs_shape, Dict), obs_shape
            self.actor_base = SMACAgentwiseObsEncoder(obs_shape, hidden_dim)
        else:
            self.actor_base = nn.Sequential(
                nn.LayerNorm([obs_shape[0]]),
                modules.mlp([obs_shape[0], hidden_dim, hidden_dim], nn.ReLU, layernorm=True))

        self.actor_rnn = modules.AutoResetRNN(hidden_dim, hidden_dim)
        self.actor_rnn_norm = nn.LayerNorm([hidden_dim])

        self.policy_head = nn.Linear(hidden_dim, act_dim)

        for k, p in self.actor_base.named_parameters():
            if 'weight' in k and len(p.data.shape) >= 2:
                # filter out layer norm weights
                nn.init.orthogonal_(p.data, gain=math.sqrt(2))
            if 'bias' in k:
                nn.init.zeros_(p.data)

        for k, p in self.actor_rnn.named_parameters():
            if 'weight' in k and len(p.data.shape) >= 2:
                # filter out layer norm weights
                nn.init.orthogonal_(p.data)
            if 'bias' in k:
                nn.init.zeros_(p.data)

        # policy head should have a smaller scale
        nn.init.orthogonal_(self.policy_head.weight.data, gain=0.01)

    def forward(self, local_obs, available_action, actor_hx, on_reset=None):
        if self.shared:
            bs = available_action.shape[1]
            # merge agent axis into batch axis
            local_obs = recursive_apply(local_obs,
                                        lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:]))
            actor_hx = actor_hx.view(actor_hx.shape[0], actor_hx.shape[1] * actor_hx.shape[2],
                                     *actor_hx.shape[3:])
            if on_reset is not None:
                on_reset = on_reset.view(on_reset.shape[0], on_reset.shape[1] * on_reset.shape[2],
                                         *on_reset.shape[3:])

        actor_features = self.actor_base(local_obs)

        actor_features, actor_hx = self.actor_rnn(actor_features, actor_hx, on_reset)
        actor_features = self.actor_rnn_norm(actor_features)

        if self.shared:
            # recover agent axis
            actor_features = actor_features.view(actor_features.shape[0], bs, actor_features.shape[1] // bs,
                                                 *actor_features.shape[2:])
            actor_hx = actor_hx.view(actor_hx.shape[0], bs, actor_hx.shape[1] // bs, *actor_hx.shape[2:])

        logits = self.policy_head(actor_features)
        logits[available_action == 0] = -1e10

        return Categorical(logits=logits), actor_hx


class SMACActorOnlyPolicy(Policy):

    @property
    def default_policy_state(self):
        return SMACPolicyState(self.__rnn_default_hidden, self.__rnn_default_hidden)

    @property
    def net(self):
        return self.__net

    def __init__(self,
                 map_name: str,
                 hidden_dim: int,
                 chunk_len: int,
                 seed: int = 0,
                 shared: bool = False,
                 agent_specific_obs: bool = False,
                 agent_specific_state: bool = False):
        super().__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        obs_shape, _, act_dim, n_agents = get_smac_shapes(map_name,
                                                          agent_specific_obs=agent_specific_obs,
                                                          agent_specific_state=agent_specific_state)
        self.__act_dim = act_dim
        self.__rnn_hidden_dim = hidden_dim
        self.__chunk_len = chunk_len
        if shared:
            self.__rnn_default_hidden = np.zeros((1, n_agents, self.__rnn_hidden_dim), dtype=np.float32)
        else:
            self.__rnn_default_hidden = np.zeros((1, self.__rnn_hidden_dim), dtype=np.float32)
        self.__net = SMACActorOnlyNet(obs_shape, hidden_dim, act_dim, shared, agent_specific_obs,
                                      agent_specific_state).to(self.device)

    def distributed(self):
        if dist.is_initialized():
            if self.device == 'cpu':
                self.__net = DDP(self.net)
            else:
                self.__net = DDP(self.net, device_ids=[self.device], output_device=self.device)

    def analyze(self, sample: SampleBatch, **kwargs):
        sample = recursive_apply(sample,
                                 lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=self.device))

        num_chunks = (sample.on_reset.shape[0] - 1) // self.__chunk_len
        observation = recursive_apply(sample.obs[:-1], lambda x: modules.to_chunk(x, num_chunks))
        policy_state = recursive_apply(sample.policy_state[:-1], lambda x: modules.to_chunk(x, num_chunks))
        # when one step is done, rnn states of the NEXT step should be reset
        on_reset = recursive_apply(sample.on_reset[:-1], lambda x: modules.to_chunk(x, num_chunks))
        action = recursive_apply(sample.action[:-1], lambda x: modules.to_chunk(x, num_chunks))

        bs = on_reset.shape[1]
        if policy_state is not None:
            actor_hx = policy_state.actor_hx[0].transpose(0, 1)
        else:
            actor_hx = torch.from_numpy(np.stack([self.__rnn_default_hidden for _ in range(bs)],
                                                 1)).to(dtype=torch.float32, device=self.device)

        action_distribution, _ = self.__net(observation.local_obs, observation.available_action, actor_hx,
                                            on_reset)

        new_log_probs = modules.back_to_trajectory(
            action_distribution.log_prob(action.x.squeeze(-1)).unsqueeze(-1), num_chunks)

        return new_log_probs

    def rollout(self, *args, **kwargs):
        raise NotImplementedError("SMACActorOnlyPolicy should only be used in imitation learning.")

    def parameters(self):
        return self.__net.parameters(recurse=True)

    def state_dict(self):
        return self.__net.state_dict()

    def set_state_dict(self, state_dict):
        self.__net.load_state_dict(state_dict)

    def train_mode(self):
        self.__net.train()

    def eval_mode(self):
        self.__net.eval()


register("smac_actor_only", SMACActorOnlyPolicy)
