from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
import copy
import numpy as np
import torch.distributed as dist

from api.policy import SingleModelPytorchPolicy, RolloutRequest, RolloutResult, register
from api.trainer import SampleBatch
from legacy.algorithm.q_learning.deep_q_learning import SampleAnalyzedResult as QMIXAnalyzeResult
from base.namedarray import recursive_apply, NamedArray, recursive_aggregate
from .mixer import *
import base.timeutil
import legacy.algorithm.modules as modules


def merge_BN(x):
    if x is None:
        return x
    return x.reshape(x.shape[0], -1, *x.shape[3:])


def split_BN(x, batch_size, num_agents):
    if x is None:
        return x
    return x.reshape(x.shape[0], batch_size, num_agents, *x.shape[2:])


class PolicyState(NamedArray):

    def __init__(self, hx: np.ndarray, epsilon: np.ndarray):
        super(PolicyState, self).__init__(hx=hx, epsilon=epsilon)


class AgentQFunction(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_dim, num_dense_layers, rnn_type, num_rnn_layers):
        super(AgentQFunction, self).__init__()
        self.obs_norm = nn.LayerNorm([obs_dim])
        self.backbone = modules.RecurrentBackbone(obs_dim=obs_dim,
                                                  hidden_dim=hidden_dim,
                                                  dense_layers=num_dense_layers,
                                                  num_rnn_layers=num_rnn_layers,
                                                  rnn_type=rnn_type)
        self.head = nn.Linear(self.backbone.feature_dim, action_dim)
        nn.init.orthogonal_(self.head.weight.data, gain=0.01)

    def forward(self, obs, hx, on_reset=None, available_action=None):
        obs = self.obs_norm(obs)
        features, hx = self.backbone(obs, hx, on_reset)
        q = self.head(features)
        if available_action is not None:
            q[available_action == 0.] = -1e10
        return q, hx


class _Qtot(nn.Module):

    def __init__(self, num_agents, obs_dim, action_dim, state_dim, q_i, q_i_config, mixer, mixer_config):
        super(_Qtot, self).__init__()
        self.__num_agents = num_agents
        self.__obs_dim = obs_dim
        self.__action_dim = action_dim
        self.__state_dim = state_dim

        self.q_i = q_i(obs_dim, action_dim, **q_i_config)
        self.mixer = mixer(num_agents, state_dim, **mixer_config)

    def forward(self, obs, hx, on_reset=None, available_action=None, action=None, state=None, mode="rollout"):
        batch_size, num_agents = obs.shape[1], self.__num_agents

        obs = merge_BN(obs)
        hx = merge_BN(hx)
        on_reset = merge_BN(on_reset)
        available_action = merge_BN(available_action)

        q_i_full, hx = self.q_i(obs, hx, available_action=available_action)
        greedy_q_i, greedy_action = q_i_full.max(dim=-1, keepdim=True)

        hx = split_BN(hx, batch_size, num_agents)
        q_i_full = split_BN(q_i_full, batch_size, num_agents)
        greedy_q_i = split_BN(greedy_q_i, batch_size, num_agents)
        greedy_action = split_BN(greedy_action, batch_size, num_agents)

        if mode == "rollout":
            return q_i_full, greedy_q_i, greedy_action, hx
        elif mode == "analyze":
            if action is None:
                action = greedy_action
            q_i = torch.gather(q_i_full, -1, action.to(dtype=torch.int64))
            q_tot = self.mixer(q_i, state)
            return q_i_full, q_i, greedy_q_i, greedy_action, q_tot.reshape(-1, batch_size, 1)
        else:
            raise NotImplementedError


class Qtot(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = _Qtot(*args, **kwargs)
        self.target_net = copy.deepcopy(self.net)
        for p in self.target_net.parameters():
            p.requires_grad = False

    def forward(self, *x, target: bool = False, **kwargs):
        return self.net(*x, **kwargs) if not target else self.target_net(*x, **kwargs)


class QtotPolicy(SingleModelPytorchPolicy):

    @property
    def default_policy_state(self):
        return PolicyState(self.__rnn_default_hidden[:, 0, :, :], np.ones((1,)))

    def __init__(self,
                 num_agents,
                 obs_dim,
                 action_dim,
                 state_dim,
                 chunk_len=100,
                 use_double_q=True,
                 epsilon_start=1.0,
                 epsilon_finish=0.05,
                 epsilon_anneal_time=5000,
                 q_i_config=dict(hidden_dim=128, num_dense_layers=2, rnn_type="gru", num_rnn_layers=1),
                 mixer="qmix",
                 mixer_config=dict(hidden_dim=64, num_hypernet_layers=2, hypernet_hidden_dim=64,
                                   popart=False),
                 state_use_all_local_obs=False,
                 state_concate_all_local_obs=False,
                 seed=0):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.__num_agents = num_agents
        self.__obs_dim = obs_dim
        self.__action_dim = action_dim
        self.__state_dim = state_dim
        self.__chunk_len = chunk_len
        self.__use_double_q = use_double_q
        self.exploration = base.timeutil.ChainedScheduler([
            base.timeutil.LinearScheduler(
                init_value=epsilon_start,
                total_iters=epsilon_anneal_time,
                end_value=epsilon_finish,
            ),
            base.timeutil.ConstantScheduler(
                init_value=epsilon_finish,
                total_iters=int(1e10),
            )
        ])

        self.__state_use_all_local_obs = state_use_all_local_obs
        self.__state_concate_all_local_obs = state_concate_all_local_obs

        if self.__state_use_all_local_obs:
            self.__state_dim = self.__obs_dim * self.__num_agents
        if self.__state_concate_all_local_obs:
            self.__state_dim += self.__obs_dim * self.__num_agents

        self.__rnn_default_hidden = np.zeros(
            (q_i_config["num_rnn_layers"], 1, num_agents, q_i_config["hidden_dim"]))

        super(QtotPolicy, self).__init__(
            Qtot(
                self.__num_agents,
                self.__obs_dim,
                self.__action_dim,
                self.__state_dim,
                AgentQFunction,
                q_i_config,
                Mixer[mixer],
                mixer_config,
            ))

    def analyze(self, sample: SampleBatch, target="qmix", **kwargs):
        """ Generate outputs required for loss computation during training,
            e.g. value target and action distribution entropies. Typically,
            data has a shape of [T, B, *D] and RNN states have a shape of
            [num_layers, B, hidden_size].
        Args:
            sample (SampleBatch): Arrays of (obs, action ...) containing
                all data required for loss computation.
            target (str): style by which the algorithm should be analyzed.
        Returns:
            analyzed_result[SampleAnalyzedResult]: Data generated for loss computation.
        """
        if target == "qmix":
            return self._qmix_analyze(sample, **kwargs)
        else:
            raise ValueError(
                f"Analyze method for algorithm {target} not implemented for {self.__class__.__name__}")

    def _qmix_analyze(self, sample_, burn_in_steps=0):
        n_agents = sample_.on_reset.shape[2]
        sample_steps = sample_.on_reset.shape[0] - burn_in_steps
        num_chunks = sample_steps // self.__chunk_len
        sample = recursive_apply(sample_, lambda x: modules.to_chunk(x[burn_in_steps:], num_chunks))
        if burn_in_steps == 0:
            hx = sample.policy_state.hx[0].transpose(0, 1)
        else:

            def _make_burn_in_data(x):
                xs = []
                for i in range(num_chunks):
                    xs.append(x[i * self.__chunk_len:i * self.__chunk_len + burn_in_steps])
                return recursive_aggregate(xs, lambda x: torch.cat(x, dim=1))

            burn_in_obs = recursive_apply(sample_.obs, lambda x: _make_burn_in_data(x))
            burn_in_policy_state = recursive_apply(sample_.policy_state, lambda x: _make_burn_in_data(x))
            burn_in_on_reset = recursive_apply(sample_.on_reset, lambda x: _make_burn_in_data(x))

            with torch.no_grad():
                _, _, _, hx = self._net(
                    burn_in_obs.local_obs if hasattr(burn_in_obs, "local_obs") else burn_in_obs.obs,
                    burn_in_policy_state.hx[0].transpose(0, 1),
                    on_reset=burn_in_on_reset,
                    available_action=burn_in_obs.available_action
                    if hasattr(burn_in_obs, "available_action") else None,
                    mode="rollout")

        if hasattr(sample.obs, "obs"):
            obs = sample.obs.obs
        elif hasattr(sample.obs, "local_obs"):
            obs = sample.obs.local_obs
        else:
            raise RuntimeError("sample obs doesn't have local_obs or obs.")

        on_reset = sample.on_reset
        action = sample.action.x
        available_action = sample.obs.available_action if hasattr(sample.obs, "available_action") else None

        state = None
        if hasattr(sample.obs, "state"):
            state = sample.obs.state.mean(dim=2)  # average over agent dimension
        if self.__state_use_all_local_obs:
            state = obs.reshape(obs.shape[0], obs.shape[1], -1)
            assert state.shape[2] == self.__state_dim
        if self.__state_concate_all_local_obs:
            state = torch.cat([state, obs.reshape(obs.shape[0], obs.shape[1], -1)], dim=-1)

        q_i_full, q_i, greedy_q_i, greedy_action, q_tot = self._net(obs,
                                                                    hx,
                                                                    on_reset=on_reset,
                                                                    available_action=available_action,
                                                                    action=action,
                                                                    state=state,
                                                                    mode="analyze")

        with torch.no_grad():
            if self.__use_double_q:
                _, _, _, _, target_q_tot = self._net(obs,
                                                     hx,
                                                     on_reset=on_reset,
                                                     action=greedy_action,
                                                     available_action=available_action,
                                                     state=state,
                                                     mode="analyze",
                                                     target=True)
            else:
                _, _, _, _, target_q_tot = self._net(obs,
                                                     hx,
                                                     on_reset=on_reset,
                                                     action=None,
                                                     available_action=available_action,
                                                     state=state,
                                                     mode="analyze",
                                                     target=True)

        # Add an agent dim.
        analyzed_result = QMIXAnalyzeResult(
            q_tot=modules.back_to_trajectory(q_tot, num_chunks).unsqueeze(2),
            target_q_tot=modules.back_to_trajectory(target_q_tot, num_chunks).unsqueeze(2),
        )
        return analyzed_result

    def rollout(self, requests: RolloutRequest, **kwargs):
        """ Provide inference results for actor workers. Typically,
            data and masks have a shape of [batch_size, *D], and RNN states
            have a shape of [batch_size, num_layers, hidden_size].
        Args:
            requests (RolloutRequest): Observations, policy states,
                evaluation masks and reset masks. 
        Returns:
            RolloutResult: Actions and new policy states, optionally
                with other entries depending on the algorithm.
        """
        eval_mask = torch.from_numpy(requests.is_evaluation).to(dtype=torch.int32, device=self.device)
        requests = recursive_apply(requests,
                                   lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=self.device))

        bs = requests.length(0)
        if hasattr(requests.obs, "obs"):
            obs = requests.obs.obs.unsqueeze(0)
        elif hasattr(requests.obs, "local_obs"):
            obs = requests.obs.local_obs.unsqueeze(0)
        else:
            raise RuntimeError("requests obs doesn't have local_obs or obs")

        available_action = None
        if hasattr(requests.obs, "available_action"):
            available_action = requests.obs.available_action.unsqueeze(0)

        hx = requests.policy_state.hx.transpose(0, 1)
        on_reset = requests.on_reset.unsqueeze(0)
        epsilon = self.exploration.get(max(0, self._version))

        with torch.no_grad():
            q_i_full, greedy_q_i, greedy_action, hx = self._net(obs,
                                                                hx,
                                                                on_reset=on_reset,
                                                                available_action=available_action,
                                                                mode="rollout")
            q_i_full = q_i_full.squeeze(0)
            greedy_q_i = greedy_q_i.squeeze(0)
            greedy_action = greedy_action.squeeze(0)

            onehot_greedy_action = torch.scatter(torch.zeros_like(q_i_full),
                                                 dim=-1,
                                                 src=torch.ones_like(greedy_action).to(dtype=torch.float32),
                                                 index=greedy_action)
            if available_action is not None:
                available_action = available_action.squeeze(0)
                random_prob = available_action / available_action.sum(dim=-1, keepdim=True)
            else:
                random_prob = 1 / self.__action_dim

            prob = eval_mask * onehot_greedy_action + (1. -
                                                       eval_mask) * (epsilon * random_prob +
                                                                     (1. - epsilon) * onehot_greedy_action)
            prob = Categorical(prob)
            action = prob.sample().unsqueeze(-1)

            hx = hx.transpose(0, 1).cpu().numpy()

        return action.cpu().numpy(), PolicyState(hx, epsilon * np.ones((bs, 1)))

    @property
    def net_module(self):
        return self.net.module if dist.is_initialized() else self.net

    def normalize_value(self, x):
        return self.net_module.net.mixer.popart_head.normalize(x)

    def normalize_target_value(self, x):
        return self.net_module.target_net.mixer.popart_head.normalize(x)

    def denormalize_value(self, x):
        return self.net_module.net.mixer.popart_head.denormalize(x)

    def denormalize_target_value(self, x):
        return self.net_module.target_net.mixer.popart_head.denormalize(x)

    def update_popart(self, x, mask=None):
        return self.net_module.net.mixer.popart_head.update(x, mask=mask)

    def soft_target_update(self, tau):
        modules.soft_update(self.net_module.target_net, self.net_module.net, tau)

    def hard_target_update(self):
        modules.hard_update(self.net_module.target_net, self.net_module.net)


register("qtot-policy", QtotPolicy)
