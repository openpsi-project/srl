from typing import Literal, Optional, Union
import copy
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from legacy.algorithm.q_learning.deep_q_learning import SampleAnalyzedResult as QlearningAnalyzedResult
import api.policy
import api.env_utils
import base.namedarray
import base.timeutil
import legacy.algorithm.modules as modules


class DQNPolicyState(api.policy.PolicyState, base.namedarray.NamedArray):

    def __init__(self, hx: np.ndarray, epsilon: np.ndarray):
        super().__init__(hx=hx, epsilon=epsilon)


class DQNRolloutAnalyzedResult(api.policy.AnalyzedResult, base.namedarray.NamedArray):

    def __init__(self, value: np.ndarray, target_value: np.ndarray, ret: Optional[np.ndarray] = None):
        super().__init__(value=value, target_value=target_value, ret=ret)


class _AtariDQN(nn.Module):

    def __init__(
        self,
        num_actions,
        num_rnn_layers: int,
        dueling: bool,
        hidden_dim: int = 512,
        rnn_type: Literal['gru', 'lstm'] = 'gru',
        rnn_include_last_action: bool = False,
        rnn_include_last_reward: bool = False,
    ):
        super().__init__()
        # The input shape is assumed to be [4, 84, 84].
        # The convolution architechture is borrowed from DQN pushlished in Nature.
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.rnn_inlude_last_action = rnn_include_last_action
        self.rnn_inlude_last_reward = rnn_include_last_reward

        self.num_rnn_layers = num_rnn_layers
        if self.num_rnn_layers > 0:
            rnn_input_dim = 7 * 7 * 64
            if self.rnn_inlude_last_action:
                rnn_input_dim += num_actions
            if self.rnn_inlude_last_reward:
                rnn_input_dim += 1
            self.rnn = modules.AutoResetRNN(input_dim=rnn_input_dim,
                                            output_dim=hidden_dim,
                                            num_layers=self.num_rnn_layers,
                                            rnn_type=rnn_type)
            # for k, v in self.rnn.named_parameters():
            #     if 'weight' in k:
            #         nn.init.orthogonal_(v.data)
            #     if 'bias' in k:
            #         nn.init.zeros_(v.data)
            # self.rnn_norm = nn.LayerNorm(hidden_dim)

        self.dueling = dueling
        fc_input_dim = 512 if self.num_rnn_layers > 0 else 7 * 7 * 64
        self.fc = nn.Sequential(nn.Linear(fc_input_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, num_actions))
        if dueling:
            self.fc_v = nn.Sequential(nn.Linear(fc_input_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, x, hx, on_reset=None, last_action=None, last_reward=None):
        # assert (x >= 0).all(), x.min()
        # assert (x <= 1).all(), x.max()
        shape = x.shape
        x = x.view(-1, *x.shape[-3:])
        act_fn = F.relu
        x = act_fn(self.conv1(x))
        x = act_fn(self.conv2(x))
        x = act_fn(self.conv3(x))
        x = x.reshape(*shape[:-3], -1)

        if self.num_rnn_layers > 0:
            if self.rnn_inlude_last_action:
                x = torch.cat([x, last_action], -1)
            if self.rnn_inlude_last_reward:
                x = torch.cat([x, last_reward], -1)
            x, hx = self.rnn(x, hx, on_reset)
            # x = self.rnn_norm(x)

        if not self.dueling:
            return self.fc(x), hx
        else:
            adv = self.fc(x)
            v = self.fc_v(x)
            return v + adv - adv.mean(-1, keepdim=True), hx


class AtariDQN(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.net = _AtariDQN(**kwargs)
        self.target_net = copy.deepcopy(self.net)
        for p in self.target_net.parameters():
            p.requires_grad = False

    def forward(self, *x, target: bool = False, **kwargs):
        return self.net(*x, **kwargs) if not target else self.target_net(*x, **kwargs)


class AtariDQNPolicy(api.policy.SingleModelPytorchPolicy):

    def __init__(
        self,
        act_dim: int,
        dueling: bool,
        chunk_len: int = 40,
        use_double_q: bool = True,
        seed: int = 42,
        num_rnn_layers: int = 0,
        hidden_dim: int = 512,
        rnn_type: Literal['gru', 'lstm'] = 'gru',
        rnn_include_last_action: bool = False,
        rnn_include_last_reward: bool = False,
        use_env_epsilon: bool = True,  # environment provides epislon as an observation
        epsilon_scheduler: Optional[Union[base.timeutil.Scheduler, base.timeutil.ChainedScheduler]] = None,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.__use_double_q = use_double_q
        self.__act_dim = act_dim
        self.__chunk_len = chunk_len
        hx = None if num_rnn_layers == 0 else np.zeros(
            (num_rnn_layers, hidden_dim * 2 if rnn_type == 'lstm' else hidden_dim), dtype=np.float32)
        self.__default_policy_state = DQNPolicyState(hx=hx, epsilon=np.ones(1, dtype=np.float32))

        self.__use_env_epsilon = use_env_epsilon
        self.__exploration = epsilon_scheduler

        net = AtariDQN(
            num_actions=act_dim,
            dueling=dueling,
            num_rnn_layers=num_rnn_layers,
            hidden_dim=hidden_dim,
            rnn_type=rnn_type,
            rnn_include_last_action=rnn_include_last_action,
            rnn_include_last_reward=rnn_include_last_reward,
        )
        super().__init__(net)

    @property
    def default_policy_state(self):
        return self.__default_policy_state

    def analyze(self, sample, target="dqn", **kwargs):
        if target == 'dqn':
            return self._dqn_analyze(sample, **kwargs)
        else:
            raise NotImplementedError(f"Unkown analyze target {target}.")

    def _dqn_analyze(self, sample_, burn_in_steps=0):
        sample_steps = sample_.on_reset.shape[0] - burn_in_steps
        num_chunks = sample_steps // self.__chunk_len

        sample = base.namedarray.recursive_apply(sample_,
                                                 lambda x: modules.to_chunk(x[burn_in_steps:], num_chunks))
        if sample.policy_state.hx is None:
            hx = None
        elif burn_in_steps == 0:
            hx = sample.policy_state.hx[0].transpose(0, 1)
        else:

            def _make_burn_in_data(x):
                xs = []
                for i in range(num_chunks):
                    xs.append(x[i * self.__chunk_len:i * self.__chunk_len + burn_in_steps])
                return base.namedarray.recursive_aggregate(xs, lambda x: torch.cat(x, dim=1))

            burn_in_obs_ = base.namedarray.recursive_apply(sample_.obs, lambda x: _make_burn_in_data(x))
            burn_in_last_reward = getattr(burn_in_obs_, "reward", None)
            burn_in_last_action = getattr(burn_in_obs_, "action", None)
            burn_in_policy_state = base.namedarray.recursive_apply(sample_.policy_state,
                                                                   lambda x: _make_burn_in_data(x))
            burn_in_hx = burn_in_policy_state.hx[0].transpose(0, 1)
            burn_in_on_reset = base.namedarray.recursive_apply(sample_.on_reset,
                                                               lambda x: _make_burn_in_data(x))

            with torch.no_grad():
                _, hx = self.net(
                    burn_in_obs_.obs / 255.,
                    burn_in_hx,
                    burn_in_on_reset,
                    last_action=burn_in_last_action,
                    last_reward=burn_in_last_reward,
                )

        obs = sample.obs.obs
        on_reset = sample.on_reset
        action = sample.action.x

        last_action = getattr(sample.obs, "action", None)
        last_reward = getattr(sample.obs, "reward", None)

        qf, _ = self.net(
            obs / 255.,
            hx,
            on_reset,
            last_action=last_action,
            last_reward=last_reward,
        )
        qa = torch.gather(qf, dim=-1, index=action.long())

        with torch.no_grad():
            target_qf, _ = self.net(obs / 255.,
                                    hx,
                                    on_reset,
                                    last_action=last_action,
                                    last_reward=last_reward,
                                    target=True)
            if self.__use_double_q:
                greedy_action = qf.argmax(-1, keepdim=True)
                target_qa = target_qf.gather(dim=-1, index=greedy_action.long())
            else:
                target_qa = target_qf.max(-1, keepdim=True).values

        return QlearningAnalyzedResult(
            q_tot=modules.back_to_trajectory(qa, num_chunks),
            target_q_tot=modules.back_to_trajectory(target_qa, num_chunks),
        )

    @torch.no_grad()
    def rollout(self, requests: api.policy.RolloutRequest) -> api.policy.RolloutResult:
        assert requests.obs.obs.dtype == np.uint8, requests.obs.obs.dtype
        requests = base.namedarray.recursive_apply(
            requests, lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=self.device))

        bs = requests.length(0)

        obs = requests.obs.obs.unsqueeze(0) / 255.
        last_action = getattr(requests.obs, "action").unsqueeze(0)
        last_reward = getattr(requests.obs, "reward").unsqueeze(0)
        hx = requests.policy_state.hx.transpose(0, 1) if requests.policy_state.hx is not None else None
        on_reset = requests.on_reset.unsqueeze(0)

        qf, hx_ = self.net(obs, hx, on_reset, last_action=last_action, last_reward=last_reward)
        qf = qf.squeeze(0)

        greedy_action = qf.argmax(-1, keepdim=True)
        rnd_action = torch.randint_like(greedy_action, low=0, high=self.__act_dim)
        # Atari observation includes per-environment epsilon.
        epsilon = requests.obs.epsilon if self.__use_env_epsilon else self.__exploration.get(
            max(0, self.version))
        eps_greedy_mask = (torch.rand(*greedy_action.shape).to(self.device) > epsilon).float()
        eps_greedy_action = eps_greedy_mask * greedy_action + (1 - eps_greedy_mask) * rnd_action

        action = requests.is_evaluation * greedy_action + (1 - requests.is_evaluation) * eps_greedy_action
        qa = qf.gather(-1, action.long())

        target_qf, _ = self.net(obs,
                                hx,
                                on_reset,
                                last_action=last_action,
                                last_reward=last_reward,
                                target=True)
        target_qf = target_qf.squeeze(0)
        if self.__use_double_q:
            target_qa = target_qf.gather(-1, greedy_action.long())
        else:
            target_qa = target_qf.max(-1, keepdim=True).values

        return api.policy.RolloutResult(
            action=api.env_utils.DiscreteAction(action.cpu().numpy()),
            policy_state=DQNPolicyState(
                hx=hx_.transpose(0, 1).cpu().numpy() if hx_ is not None else None,
                epsilon=requests.obs.epsilon.cpu().numpy() if self.__use_env_epsilon else epsilon * np.ones(
                    (bs, 1), dtype=np.float32),
            ),
            analyzed_result=DQNRolloutAnalyzedResult(
                value=qa.cpu().numpy(),
                target_value=target_qa.cpu().numpy(),
            ),
        )

    @property
    def net_module(self):
        return self.net.module if dist.is_initialized() else self.net

    def soft_target_update(self, tau):
        modules.soft_update(self.net_module.target_net, self.net_module.net, tau)

    def hard_target_update(self):
        modules.hard_update(self.net_module.target_net, self.net_module.net)


api.policy.register("atari-dqn", AtariDQNPolicy)