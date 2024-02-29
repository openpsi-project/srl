from typing import Optional, Union, List, Dict, Tuple
from torch.distributions import Categorical
import itertools
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from legacy.algorithm import modules
from api.env_utils import DiscreteAction
from api.policy import SingleModelPytorchPolicy, RolloutRequest, RolloutResult, register
from api.trainer import SampleBatch
from legacy.algorithm.ppo.mappo import SampleAnalyzedResult as PPOAnalyzeResult
from base.namedarray import recursive_apply, NamedArray, recursive_aggregate
from legacy.algorithm.ppo.actor_critic_policies.utils import get_action_indices
from legacy.algorithm.ppo.actor_critic_policies.actor_critic_policy import PPORolloutAnalyzedResult

import time


class VizDoomActorCritic(nn.Module):

    def __init__(self, obs_shapes: Dict[str, Tuple[int]], action_dims: List[int], hidden_dim, dense_layers,
                 rnn_type, num_rnn_layers, popart, activation, layernorm, popart_beta, **kwargs):
        super(VizDoomActorCritic, self).__init__()

        self.action_dims = action_dims
        self.total_action_dim = sum(action_dims)

        if activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'tanh':
            self.activation = nn.Tanh
        elif activation == 'elu':
            self.activation = nn.ELU
        else:
            raise NotImplementedError(f"Activation function {activation} not implemented.")

        input_channels = obs_shapes['obs'][-3]
        output_pixel_shape = obs_shapes['obs'][-2:]
        conv_specs = [[input_channels, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]
        conv = [nn.LayerNorm(obs_shapes['obs'])]
        for spec in conv_specs:
            conv.append(nn.Conv2d(spec[0], spec[1], kernel_size=spec[2], stride=spec[3]))
            conv.append(self.activation())
            output_pixel_shape = modules.cnn_output_dim(
                output_pixel_shape,
                padding=(0, 0),
                dilation=(1, 1),
                kernel_size=(spec[2], spec[2]),
                stride=(spec[3], spec[3]),
            )
        conv += [nn.Flatten(), nn.Linear(int(np.prod(output_pixel_shape)) * 128, hidden_dim)]
        self.pixel_encoder = nn.Sequential(*conv)

        for k, p in self.pixel_encoder.named_parameters():
            if 'weight' in k and len(p.data.shape) >= 4:
                # filter out layer norm weights
                nn.init.orthogonal_(p.data)
            if 'bias' in k:
                nn.init.zeros_(p.data)

        self.vector_encoder = None
        feature_dim = hidden_dim
        if 'measurements' in obs_shapes:
            if layernorm:
                self.vector_encoder = nn.Sequential(nn.LayerNorm(obs_shapes['measurements'][0]),
                                                    nn.Linear(obs_shapes['measurements'][0], 128),
                                                    self.activation(), nn.LayerNorm(128), nn.Linear(128, 128),
                                                    self.activation(), nn.LayerNorm(128))
            else:
                self.vector_encoder = nn.Sequential(nn.LayerNorm(obs_shapes['measurements'][0]),
                                                    nn.Linear(obs_shapes['measurements'][0], 128),
                                                    self.activation(), nn.Linear(128, 128), self.activation())
            feature_dim += 128

            for k, p in self.vector_encoder.named_parameters():
                if 'weight' in k and len(p.data.shape) >= 2:
                    # filter out layer norm weights
                    nn.init.orthogonal_(p.data)
                if 'bias' in k:
                    nn.init.zeros_(p.data)

        self.num_rnn_layers = num_rnn_layers
        if num_rnn_layers > 0:
            self.rnn = modules.AutoResetRNN(feature_dim,
                                            hidden_dim,
                                            num_layers=num_rnn_layers,
                                            rnn_type=rnn_type)

        feature_dim = hidden_dim if num_rnn_layers > 0 else feature_dim

        self.actor_mlp_before_output = modules.mlp([feature_dim] + [hidden_dim for _ in range(dense_layers)],
                                                   activation=self.activation,
                                                   layernorm=layernorm)
        self.critic_mlp_before_output = modules.mlp([feature_dim] + [hidden_dim for _ in range(dense_layers)],
                                                    activation=self.activation,
                                                    layernorm=layernorm)

        for k, p in itertools.chain(self.actor_mlp_before_output.named_parameters(),
                                    self.critic_mlp_before_output.named_parameters()):
            if 'weight' in k and len(p.data.shape) >= 2:
                # filter out layer norm weights
                nn.init.orthogonal_(p.data)
            if 'bias' in k:
                nn.init.zeros_(p.data)

        feature_dim = hidden_dim if dense_layers > 0 else feature_dim

        self.actor_head = nn.Linear(feature_dim, self.total_action_dim)
        offs = 0
        for act_dim in action_dims:
            # initialization trick taken from hide-and-seek paper
            weight = torch.randn_like(self.actor_head.weight.data[offs:offs + act_dim])
            weight *= 0.01 / torch.sqrt(torch.square(weight).sum(0, keepdim=True))
            self.actor_head.weight.data[offs:offs + act_dim] = weight
            offs += act_dim
        assert offs == self.total_action_dim

        self.__popart = popart
        if self.__popart:
            self.critic_head = modules.PopArtValueHead(feature_dim, 1, beta=popart_beta)
        else:
            self.critic_head = nn.Linear(feature_dim, 1)

    def _init_layer(self, model: nn.Module):
        nn.init.orthogonal_(model.weight.data, gain=0.01)
        nn.init.zeros_(model.bias.data)
        return model

    def forward(self,
                obs_: NamedArray,
                hx: torch.Tensor,
                on_reset: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pixel_shape = obs_['obs'].shape
        feature = self.pixel_encoder(obs_['obs'].view(-1, *pixel_shape[-3:]) / 255.0)
        feature = feature.view(*pixel_shape[:-3], -1)

        if getattr(obs_, 'measurements', None) is not None:
            vec_feature = self.vector_encoder(obs_.measurements)
            feature = torch.cat([vec_feature, feature], -1)

        if self.num_rnn_layers > 0:
            feature, hx = self.rnn(feature, hx, on_reset)

        actor_features = self.actor_mlp_before_output(feature)
        critic_features = self.critic_mlp_before_output(feature)

        return self.actor_head(actor_features), self.critic_head(critic_features), hx


class VizDoomActorCriticPolicy(SingleModelPytorchPolicy):

    def __init__(self,
                 obs_shapes: Dict[str, Tuple[int]],
                 action_dims: List[int],
                 hidden_dim: int = 512,
                 chunk_len: int = 10,
                 num_dense_layers: int = 0,
                 rnn_type: str = "gru",
                 num_rnn_layers: int = 1,
                 popart: bool = True,
                 activation: str = "elu",
                 layernorm: bool = False,
                 popart_beta: float = 0.99999,
                 use_symmetric_kl: bool = False,
                 seed=0,
                 **kwargs):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.__action_indices = get_action_indices(action_dims)
        self.__chunk_len = chunk_len
        self.__popart = popart
        self.__num_rnn_layers = num_rnn_layers
        if rnn_type == "gru":
            self.__rnn_default_hidden = np.zeros((num_rnn_layers, 1, hidden_dim), dtype=np.float32)
        elif rnn_type == "lstm":
            self.__rnn_default_hidden = np.zeros((num_rnn_layers, 1, hidden_dim * 2), dtype=np.float32)
        else:
            raise ValueError(f"Unknown rnn_type {rnn_type} for ActorCriticPolicy.")

        self._avg_from_numpy_time = 0
        self._rollout_count = 0
        self._use_symmetric_kl = use_symmetric_kl

        neural_network = VizDoomActorCritic(obs_shapes=obs_shapes,
                                            action_dims=action_dims,
                                            hidden_dim=hidden_dim,
                                            dense_layers=num_dense_layers,
                                            rnn_type=rnn_type,
                                            num_rnn_layers=num_rnn_layers,
                                            popart=popart,
                                            activation=activation,
                                            layernorm=layernorm,
                                            popart_beta=popart_beta,
                                            **kwargs)
        super(VizDoomActorCriticPolicy, self).__init__(neural_network)

    @property
    def popart_head(self):
        if not self.__popart:
            raise ValueError("Set popart=True in policy config to activate popart value head.")
        return self.net.module.critic_head if dist.is_initialized() else self.net.critic_head

    def normalize_value(self, x):
        return self.popart_head.normalize(x)

    def denormalize_value(self, x):
        return self.popart_head.denormalize(x)

    def update_popart(self, x, mask):
        return self.popart_head.update(x, mask)

    @property
    def default_policy_state(self):
        return NamedArray(hx=self.__rnn_default_hidden[:, 0, :]) if self.__num_rnn_layers > 0 else None

    def analyze(self, sample: SampleBatch, target="ppo", **kwargs):
        """ Generate outputs required for loss computation during training,
            e.g. value target and action distribution entropies. Typically,
            data has a shape of [T, B, *D] and RNN states have a shape of
            [num_layers, B, hidden_size].
        Args:
            sample (SampleBatch): Arrays of (obs, action ...) containing
                all data required for loss computation.
            target (str): style by which the algorithm should be analyzed.
            kwargs: addition keyword arguments.
        Returns:
            analyzed_result[SampleAnalyzedResult]: Data generated for loss computation.
        """
        if target == "ppo":
            return self._ppo_analyze(sample, **kwargs)
        else:
            raise ValueError(
                f"Analyze method for algorithm {target} not implemented for {self.__class__.__name__}")

    def _ppo_analyze(self, sample_, burn_in_steps=0, **kwargs):
        """Perform PPO style analysis on sample.
        Args:
            bootstrap_steps: the last {bootstrap_steps} steps will not be analyzed. For example, if the
                training algorithm uses GAE only, then `bootstrap_steps` should be align with the config of
                actor workers. Or, if the training algorithm uses Vtrace, then `bootstrap_steps` should be 1
                because all log-probs will be used except for the last step.
        """
        sample_steps = sample_.on_reset.shape[0] - burn_in_steps
        num_chunks = sample_steps // self.__chunk_len
        sample = recursive_apply(sample_[burn_in_steps:], lambda x: modules.to_chunk(x, num_chunks))

        if self.__num_rnn_layers == 0:
            hx = None
        elif burn_in_steps == 0:
            hx = sample.policy_state.hx[0].transpose(0, 1)
        else:

            def _make_burn_in_data(x):
                xs = []
                for i in range(num_chunks):
                    xs.append(x[i * self.__chunk_len:i * self.__chunk_len + burn_in_steps])
                return recursive_aggregate(xs, lambda x: torch.cat(x, dim=1))

            bi_obs = recursive_apply(sample_.obs, _make_burn_in_data)
            bi_hx = recursive_apply(sample_.policy_state.hx,
                                    lambda x: _make_burn_in_data(x)[0].transpose(0, 1))
            bi_onreset = _make_burn_in_data(sample_.on_reset)
            with torch.no_grad():
                _, _, hx = self.net(bi_obs, bi_hx, bi_onreset)

        action_logits, value, _ = self.net(sample.obs, hx, sample.on_reset)
        action_dists = [
            Categorical(logits=action_logits[..., idx.start:idx.end]) for idx in self.__action_indices
        ]
        new_log_probs = torch.sum(torch.stack(
            [dist.log_prob(sample.action.x[..., i]) for i, dist in enumerate(action_dists)], dim=-1),
                                  dim=-1,
                                  keepdim=True)
        if not self._use_symmetric_kl:
            entropy = torch.sum(torch.stack([dist.entropy() for dist in action_dists], dim=-1),
                                dim=-1,
                                keepdim=True)
        else:
            kls = []
            for dist in action_dists:
                uniform_probs = torch.ones_like(dist.probs) / dist.probs.shape[-1]
                log_uniform_probs = torch.log(uniform_probs)
                log_dist_probs = (dist.probs + 1e-5).log()
                kl = (dist.probs * (log_dist_probs - log_uniform_probs)).sum(-1, keepdim=True)
                inverse_kl = (uniform_probs * (log_uniform_probs - log_dist_probs)).sum(-1, keepdim=True)
                kls.append(torch.clamp((kl + inverse_kl) / 2, max=30))
            entropy = -torch.sum(torch.cat(kls, dim=-1), dim=-1, keepdim=True)

        analyzed_result = PPOAnalyzeResult(
            old_action_log_probs=modules.back_to_trajectory(sample.log_probs, num_chunks),
            new_action_log_probs=modules.back_to_trajectory(new_log_probs, num_chunks),
            state_values=modules.back_to_trajectory(value, num_chunks),
            entropy=modules.back_to_trajectory(entropy, num_chunks))

        return analyzed_result

    def avg_from_numpy_time(self):
        return self._avg_from_numpy_time

    def clear_timer(self):
        self._rollout_count = 0
        self._avg_from_numpy_time = 0

    def rollout(self, requests: RolloutRequest, **kwargs):
        """ Provide inference results for actor workers. Typically,
            data and masks have a shape of [batch_size, *D], and RNN states
            have a shape of [batch_size, num_layers, hidden_size].
        Returns:
            RolloutResult: Actions and new policy states, optionally
                with other entries depending on the algorithm.
        """
        st = time.monotonic()
        eval_mask = torch.from_numpy(requests.is_evaluation).to(dtype=torch.int32, device=self.device)
        requests = recursive_apply(requests,
                                   lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=self.device))
        t1 = time.monotonic() - st

        obs = recursive_apply(requests.obs, lambda x: x.unsqueeze(0))
        hx = requests.policy_state.hx.transpose(0, 1) if self.__num_rnn_layers > 0 else None
        on_reset = requests.on_reset.unsqueeze(0)

        with torch.no_grad():
            action_logits, value, hx = self.net(obs, hx, on_reset)
            # .squeeze(0) removes the time dimension
            value = value.squeeze(0)
            action_logits = action_logits.squeeze(0)
            action_dists = [
                Categorical(logits=action_logits[..., idx.start:idx.end]) for idx in self.__action_indices
            ]
            deterministic_actions = torch.stack([dist.probs.argmax(dim=-1) for dist in action_dists], dim=-1)
            # dist.sample adds an additional dimension
            stochastic_actions = torch.stack([dist.sample() for dist in action_dists], dim=-1)
            # now deterministic/stochastic actions have shape [batch_size]
            actions = eval_mask * deterministic_actions + (1 - eval_mask) * stochastic_actions
            log_probs = torch.sum(torch.stack(
                [dist.log_prob(actions[..., i]) for i, dist in enumerate(action_dists)], dim=-1),
                                  dim=-1,
                                  keepdim=True)
        st = time.monotonic()
        res = RolloutResult(
            action=DiscreteAction(actions.cpu().numpy()),
            analyzed_result=PPORolloutAnalyzedResult(
                log_probs=log_probs.cpu().numpy(),
                value=value.cpu().numpy(),
            ),
            policy_state=NamedArray(
                hx=hx.transpose(0, 1).cpu().numpy()) if self.__num_rnn_layers > 0 else None,
        )
        t2 = time.monotonic() - st
        t = t1 + t2

        self._avg_from_numpy_time = (self._avg_from_numpy_time * self._rollout_count +
                                     t) / (self._rollout_count + 1)
        self._rollout_count += 1

        return res


register("vizdoom", VizDoomActorCriticPolicy)
