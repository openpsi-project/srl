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

DMLAB_INSTRUCTIONS = "INSTR"
DMLAB_VOCABULARY_SIZE = 1000


class DMLabActorCritic(nn.Module):

    def __init__(self,
                 obs_shapes: Dict[str, Tuple[int]],
                 action_dim: int,
                 hidden_dim,
                 dense_layers,
                 rnn_type,
                 num_rnn_layers,
                 popart,
                 activation,
                 layernorm,
                 embedding_size=20,
                 instrunctions_lstm_units=64,
                 instruction_lstm_layers=1,
                 popart_beta=0.99999,
                 **kwargs):
        super(DMLabActorCritic, self).__init__()

        self.action_dim = action_dim

        # language encoder, same as IMPALA paper
        self.embedding_size = embedding_size
        self.instructions_lstm_units = instrunctions_lstm_units
        self.instructions_lstm_layers = instruction_lstm_layers
        self.word_embedding = nn.Embedding(num_embeddings=DMLAB_VOCABULARY_SIZE,
                                           embedding_dim=self.embedding_size,
                                           padding_idx=0)
        self.instructions_lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.instructions_lstm_units,
            num_layers=self.instructions_lstm_layers,
            batch_first=True,
        )

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
        conv_specs = [[input_channels, 16, 8, 4], [16, 32, 4, 2]]
        conv = []
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
        conv += [nn.Flatten(), nn.Linear(int(np.prod(output_pixel_shape)) * conv_specs[-1][1], hidden_dim)]
        self.pixel_encoder = nn.Sequential(*conv)

        for k, p in self.pixel_encoder.named_parameters():
            if 'weight' in k and len(p.data.shape) >= 4:
                # filter out layer norm weights
                nn.init.orthogonal_(p.data)
            if 'bias' in k:
                nn.init.zeros_(p.data)

        feature_dim = self.instructions_lstm_units + hidden_dim

        self.num_rnn_layers = num_rnn_layers
        if num_rnn_layers > 0:
            self.rnn = modules.AutoResetRNN(feature_dim,
                                            hidden_dim // 2,
                                            num_layers=num_rnn_layers,
                                            rnn_type=rnn_type)

        feature_dim = hidden_dim // 2 if num_rnn_layers > 0 else feature_dim

        self.actor_mlp_before_output = modules.mlp([feature_dim] +
                                                   [hidden_dim // 2 for _ in range(dense_layers)],
                                                   activation=self.activation,
                                                   layernorm=layernorm)
        self.critic_mlp_before_output = modules.mlp([feature_dim] +
                                                    [hidden_dim // 2 for _ in range(dense_layers)],
                                                    activation=self.activation,
                                                    layernorm=layernorm)

        for k, p in itertools.chain(self.actor_mlp_before_output.named_parameters(),
                                    self.critic_mlp_before_output.named_parameters()):
            if 'weight' in k and len(p.data.shape) >= 2:
                # filter out layer norm weights
                nn.init.orthogonal_(p.data)
            if 'bias' in k:
                nn.init.zeros_(p.data)

        feature_dim = hidden_dim // 2 if dense_layers > 0 else feature_dim

        self.actor_head = self._init_layer(nn.Linear(feature_dim, self.action_dim))

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
        shape_prefix = obs_['obs'].shape[:-3]
        feature = self.pixel_encoder(obs_['obs'].view(-1, *pixel_shape[-3:]) / 255.0)
        feature = feature.view(*pixel_shape[:-3], -1)

        instr = obs_.INSTR.view(-1, obs_.INSTR.shape[-1]).long()
        instr_lengths = torch.clamp((instr != 0).sum(-1), min=1).long()
        max_instr_len = instr_lengths.max().item()
        instr = instr[..., :max_instr_len]
        instr_embed = self.word_embedding(instr)
        instr_packed = torch.nn.utils.rnn.pack_padded_sequence(
            instr_embed,
            instr_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        rnn_output, _ = self.instructions_lstm(instr_packed)
        rnn_outputs, sequence_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        rnn_batch_indices = torch.arange(rnn_outputs.shape[0], device=rnn_outputs.device)
        instr_feature = rnn_outputs[rnn_batch_indices, sequence_lengths - 1]

        feature = torch.cat([feature, instr_feature.view(*shape_prefix, -1)], -1)

        if self.num_rnn_layers > 0:
            feature, hx = self.rnn(feature, hx, on_reset)

        actor_features = self.actor_mlp_before_output(feature)
        critic_features = self.critic_mlp_before_output(feature)

        return self.actor_head(actor_features), self.critic_head(critic_features), hx


class DMLabActorCriticPolicy(SingleModelPytorchPolicy):

    def __init__(self,
                 obs_shapes: Dict[str, Tuple[int]],
                 action_dim: List[int],
                 hidden_dim: int = 512,
                 chunk_len: int = 10,
                 num_dense_layers: int = 0,
                 rnn_type: str = "lstm",
                 num_rnn_layers: int = 1,
                 popart: bool = True,
                 activation: str = "relu",
                 layernorm: bool = False,
                 seed=0,
                 popart_beta: float = 0.99999,
                 **kwargs):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.__action_indices = get_action_indices([action_dim])
        self.__chunk_len = chunk_len
        self.__popart = popart
        self.__num_rnn_layers = num_rnn_layers
        if rnn_type == "gru":
            self.__rnn_default_hidden = np.zeros((num_rnn_layers, 1, hidden_dim // 2), dtype=np.float32)
        elif rnn_type == "lstm":
            self.__rnn_default_hidden = np.zeros((num_rnn_layers, 1, hidden_dim), dtype=np.float32)
        else:
            raise ValueError(f"Unknown rnn_type {rnn_type} for ActorCriticPolicy.")

        self._avg_from_numpy_time = 0
        self._rollout_count = 0

        neural_network = DMLabActorCritic(obs_shapes=obs_shapes,
                                          action_dim=action_dim,
                                          hidden_dim=hidden_dim,
                                          dense_layers=num_dense_layers,
                                          rnn_type=rnn_type,
                                          num_rnn_layers=num_rnn_layers,
                                          popart=popart,
                                          activation=activation,
                                          layernorm=layernorm,
                                          popart_beta=popart_beta,
                                          **kwargs)
        super(DMLabActorCriticPolicy, self).__init__(neural_network)

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
        entropy = torch.sum(torch.stack([dist.entropy() for dist in action_dists], dim=-1),
                            dim=-1,
                            keepdim=True)

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


register("dmlab", DMLabActorCriticPolicy)
