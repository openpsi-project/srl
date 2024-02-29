from argparse import Namespace
from torch.distributions import Categorical
import copy
import gym
import numpy as np
import torch
import torch.nn as nn
import time
import logging

from api.policy import SingleModelPytorchPolicy, RolloutRequest, RolloutResult
from api.trainer import SampleBatch
from legacy.algorithm.muzero.trainer import MuzeroSampleAnalyzedResult
from legacy.algorithm.muzero.mcts import MCTS
from legacy.algorithm.muzero.utils.utils import _n2t, pad_sampled_actions
from base.namedarray import recursive_apply, NamedArray
from api import environment

logger = logging.getLogger("MuZero Policy")


class Action(environment.Action, NamedArray):

    def __init__(self, x: np.ndarray):
        super(Action, self).__init__(x=x)


class AnalyzedResult(NamedArray):

    def __init__(self, policy: np.ndarray, actions: np.ndarray, value: np.ndarray, entropy: np.ndarray,
                 actor_entropy: np.ndarray):
        super(AnalyzedResult, self).__init__(policy=policy,
                                             actions=actions,
                                             value=value,
                                             entropy=entropy,
                                             actor_entropy=actor_entropy)


class ReanalyzedResult(NamedArray):

    def __init__(self, mask: np.ndarray, target_policy: np.ndarray, target_actions: np.ndarray,
                 target_value: np.ndarray, target_value_prefix: np.ndarray):
        super(ReanalyzedResult, self).__init__(mask=mask,
                                               target_policy=target_policy,
                                               target_actions=target_actions,
                                               target_value=target_value,
                                               target_value_prefix=target_value_prefix)


class MuzeroBasePolicy(SingleModelPytorchPolicy):

    @property
    def default_policy_state(self):
        return None

    def default_reward_hidden(self, batch_size):
        if self.lstm_hidden_size == 0:
            return None
        return (torch.zeros(1, batch_size, self.lstm_hidden_size).to(self.device),
                torch.zeros(1, batch_size, self.lstm_hidden_size).to(self.device))

    def __init__(
            self,
            act_dim: int,
            action_space,
            discount: float,
            # tree search
            use_mcts: bool,
            value_delta_max: float,
            num_simulations: int,
            root_dirichlet_alpha: float,
            root_exploration_fraction: float,
            pb_c_base: int,
            pb_c_init: float,
            num_threads: int,
            # stack observation
            stacked_observations: int,
            # value_prefix
            use_value_prefix: bool,
            value_prefix_horizon: int,
            lstm_hidden_size: int,
            # data augmentation
            use_augmentation: bool,
            # reanalyze
            reanalyze_ratio_schedule,
            td_steps: int,
            num_unroll_steps: int,
            # actor exploration
            visit_softmax_temperature_fn,
            warm_up_version: int,
            # consistency
            proj_out: int,
            pred_out: int,
            # model
            neural_network,
            # rollout & reanalyze network update interval
            rollout_update_interval: int = 100,
            reanalyze_update_interval: int = 200,
            # sampled muzero
            mcts_use_sampled_actions: bool = False,
            mcts_num_sampled_actions: int = 0,
            # others
            use_available_action: bool = False,
            seed: int = 0):
        """MuZero Base Policy
        Parameters
        ----------
        act_dim: int
            if use sampled muzero, act_dim is the number of actions for MCTS; otherwise, action_space should be gym.spaces.Discrete and act_dim = action_space.n
        discount: float
            discount factor for MDP horizon
        use_mcts: bool
            whether use monte-carlo tree search for roll out
        value_delta_max: float
            value normalization boundary in UCB score computation
        num_simulations: int
            number of searched nodes in MCTS
        root_dirichlet_alpha: float
            scale of dirichlet noise
        root_exploration_fraction: float
            fraction of exploration noise
        pb_c_base: int
            UCB score parameter
        pb_c_init: float
            UCB score parameter
        num_threads: int
            number of parallel threads for mcts
        stacked_observations: int
            number of stacked observations
        use_value_prefix: bool
            True -> predict value prefix instead of reward
        value_prefix_horizon: int
            horizon of value prefix reset
        lstm_hidden_size: int
            hidden size of lstm in dynamic network (only for predicting rewards/value_prefix)
        use_augmentation: bool
            True -> augment observation when training. Method 'transform' should be implemented in sub-class
        reanalyze_ratio_schedule: Any
            given policy version, outputs reanalyze ratio. (only used by reanalyze worker)
        td_steps: int
            td_steps for value bootstrap
        num_unroll_steps: int
            number of steps to unroll during training
        visit_softmax_temperature_fn: Any
            given policy version, outputs softmax temperature for roll out.
        warm_up_version: int
            perform random actions when policy version is less than warm_up_version
        proj_out: int
            projection output dimension
        pred_out: int
            projection output dimension
        neural_network: nn.Module
            neural network, including representation/dynamics/prediction and, optionally, projection network
        rollout_update_interval: int
            interval of rollout policy update
        reanalyze_update_interval: int
            interval of reanalyze policy update. 
        mcts_use_sampled_actions: bool
            use sampled actions to do tree search. Recommended for complex action space.
            Reference: Learning and Planning in Complex Action Spaces. http://arxiv.org/abs/2104.06303.
        mcts_num_sampled_actions: int
            number of sampled actions for MCTS when mcts_use_sampled_actions=True
        use_available_action: bool
            True -> enable available actions
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        super(MuzeroBasePolicy, self).__init__(neural_network)
        if not use_value_prefix:
            value_prefix_horizon = 1
        self.act_dim = act_dim
        self.action_space = gym.spaces.Discrete(act_dim)
        self.mcts_config = Namespace(act_dim=act_dim,
                                     action_space=action_space,
                                     discount=discount,
                                     value_delta_max=value_delta_max,
                                     num_simulations=num_simulations,
                                     root_dirichlet_alpha=root_dirichlet_alpha,
                                     root_exploration_fraction=root_exploration_fraction,
                                     pb_c_base=pb_c_base,
                                     pb_c_init=pb_c_init,
                                     num_threads=num_threads,
                                     value_prefix_horizon=value_prefix_horizon,
                                     mcts_use_sampled_actions=mcts_use_sampled_actions,
                                     device=self.device)
        self.discount = discount
        self.use_value_prefix = use_value_prefix
        self.value_prefix_horizon = value_prefix_horizon
        self.lstm_hidden_size = lstm_hidden_size
        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.proj_out = proj_out
        self.pred_out = pred_out
        self.use_mcts = use_mcts
        self.stacked_observations = stacked_observations
        self.use_augmentation = use_augmentation
        self.reanalyze_ratio_schedule = reanalyze_ratio_schedule
        self.td_steps = td_steps
        self.num_unroll_steps = num_unroll_steps
        self.mcts_use_sampled_actions = mcts_use_sampled_actions
        self.mcts_num_sampled_actions = mcts_num_sampled_actions
        self.use_available_action = use_available_action

        if mcts_use_sampled_actions and use_available_action:
            raise RuntimeError(
                "Currently using available actions and using sampled MuZero is not compatible.")
        if not mcts_use_sampled_actions:
            assert (isinstance(action_space, gym.spaces.Discrete)), (
                f"Expected Discrete action space when using standard MCTS, but get {action_space}.")
            assert (action_space.n == act_dim), (
                f"Expected act_dim=action_space.n when using standard MCTS, but get act_dim={act_dim} action_space={action_space}."
            )

        # use random policy during warm up stage
        self.warm_up_version = warm_up_version

        self._version = -1

        # init rollout and reanalyze network checkpoint
        # Note reanalyze network takes two checkpoints but only the former one is used for reanalyze
        self.rollout_update_interval = rollout_update_interval
        self.reanalyze_update_interval = reanalyze_update_interval
        ckpt = super(MuzeroBasePolicy, self).get_checkpoint()["state_dict"]
        self.rollout_ckpt = ckpt
        self.reanalyze_ckpts = [ckpt, ckpt]
        self.current = "train"

    def inc_version(self):
        self._version += 1
        if self._version % self.rollout_update_interval == 0:
            self.rollout_ckpt = super(MuzeroBasePolicy, self).get_checkpoint()["state_dict"]
        if self._version % self.reanalyze_update_interval == 0:
            self.reanalyze_ckpts = self.reanalyze_ckpts[1:] + [
                super(MuzeroBasePolicy, self).get_checkpoint()["state_dict"]
            ]

    def load_checkpoint(self, checkpoint):
        self._version = checkpoint.get("steps", 0)
        self._net.load_state_dict(checkpoint["state_dict"])
        self.rollout_ckpt = checkpoint["rollout_ckpt"]
        self.reanalyze_ckpts = checkpoint["reanalyze_ckpts"]

    def get_checkpoint(self):
        ckpt = super(MuzeroBasePolicy, self).get_checkpoint()
        return dict(**ckpt, rollout_ckpt=self.rollout_ckpt, reanalyze_ckpts=self.reanalyze_ckpts)

    def _unfold_stacked_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # obs shape: [t, b, s*c, w, h]
        obs = obs.view(obs.shape[0], obs.shape[1], self.stacked_observations, -1, obs.shape[-2],
                       obs.shape[-1])
        t_obs = obs[:, :, -1]
        before_t_obs = obs[0, :, :-1].transpose(0, 1)
        return torch.cat([before_t_obs, t_obs], dim=0)

    def fold_stacked_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # obs shape: [t+s-1, bs, c, w, h]
        bs, c, w, h = obs.shape[1:]
        t = obs.shape[0] - self.stacked_observations + 1
        x = torch.stack([obs[i:i + self.stacked_observations] for i in range(t)], 0)  # [t, s, bs, c, w, h]
        assert x.shape == (t, self.stacked_observations, bs, c, w, h)
        return x.transpose(1, 2).reshape(t * bs, self.stacked_observations * c,
                                         *x.shape[-2:])  # [t*bs, s*c, w, h]

    def analyze(self, sample: SampleBatch, do_consistency: bool):
        # print(recursive_apply(sample, lambda x: x.shape if x is not None else x))
        self.net.train()
        num_unroll_steps, td_steps = self.num_unroll_steps, self.td_steps
        assert sample.on_reset.shape[0] == (1 + num_unroll_steps + td_steps)
        length = 1 + num_unroll_steps + td_steps
        batch_size = sample.on_reset.shape[1]
        tik = time.perf_counter()

        if not do_consistency:
            # only load first step observation into gpu
            obs = self._unfold_stacked_obs(sample.obs.obs[:1]).to(self.device).float()
            obs_batch = obs[:self.stacked_observations]  # [s, b, c, w, h]
            target_obs_batch = None
        else:
            # only load first num_unroll_steps observations into gpu
            obs = self._unfold_stacked_obs(sample.obs.obs[:num_unroll_steps + 1]).to(self.device).float()
            obs_batch = obs[:self.stacked_observations]  # [s, b, c, w, h]
            target_obs_batch = obs  # [s + t - 1, b, c, w, h]

        if self.use_augmentation:
            if target_obs_batch is not None:
                target_obs_batch = self.transform(target_obs_batch)
                obs_batch = target_obs_batch[:self.stacked_observations]
            else:
                obs_batch = self.transform(obs_batch)

        action = sample.action.x.to(self.device)

        # reanalyzed target elements: value, value prefix, policy, mask
        target_value = sample.analyzed_result.target_value.to(self.device).float()
        target_value_prefix = sample.analyzed_result.target_value_prefix.to(self.device).float()
        target_policy = sample.analyzed_result.target_policy.to(self.device).float()
        target_actions = sample.analyzed_result.target_actions.to(self.device).float()
        mask = sample.analyzed_result.mask.to(self.device).float()

        t1 = time.perf_counter()

        value = None
        value_prefix = None
        policy_logits = torch.zeros_like(target_policy)
        policy_entropy = torch.zeros(num_unroll_steps + 1,
                                     batch_size,
                                     device=self.device,
                                     dtype=torch.float32)
        if do_consistency:
            dynamic_proj = torch.zeros(num_unroll_steps + 1,
                                       batch_size,
                                       self.pred_out,
                                       device=self.device,
                                       dtype=torch.float32)
            observation_proj = torch.zeros(num_unroll_steps + 1,
                                           batch_size,
                                           self.proj_out,
                                           device=self.device,
                                           dtype=torch.float32)
        else:
            dynamic_proj = None
            observation_proj = None

        tmp_value, _, policy_logits[0], policy_entropy[
            0], _, hidden_state, reward_hidden = self.net.module.initial_inference(
                self.process_stack_observation(obs_batch), sampled_actions=target_actions[0])
        value = torch.zeros((num_unroll_steps + 1, *tmp_value.shape), device=self.device, dtype=torch.float32)
        value[0] = tmp_value

        t2 = time.perf_counter()

        if do_consistency:
            consistency_target_obs_batch = self.fold_stacked_obs(target_obs_batch[1:]) / 255.0
            consistency_target_actions = target_actions[1:].flatten(end_dim=1)
            _, _, _, _, _, presentation_state, _ = self.net.module.initial_inference(
                consistency_target_obs_batch, sampled_actions=consistency_target_actions)
            cproj = self.net.module.project(presentation_state, with_grad=False)
            observation_proj[1:] = cproj.view(num_unroll_steps, -1, *cproj.shape[1:])

        t3 = time.perf_counter()

        for step in range(1, num_unroll_steps + 1):
            tt1 = time.perf_counter()
            # unroll with the dynamic function
            # TODO: DDP won't sync gradient if we use .module to do inference
            value[step], tmp_value_prefix, policy_logits[step], policy_entropy[
                step], _, hidden_state, reward_hidden = self.net.module.recurrent_inference(
                    hidden_state, reward_hidden, action[step - 1], sampled_actions=target_actions[step])
            if do_consistency:
                dynamic_proj[step] = self.net.module.project(hidden_state, with_grad=True)

            tt2 = time.perf_counter()
            if value_prefix is None:
                value_prefix = torch.zeros(
                    (num_unroll_steps + 1, *tmp_value_prefix.shape)).to(self.device).float()
            value_prefix[step] = tmp_value_prefix

            # Follow Muzero, set half gradient
            hidden_state.register_hook(lambda grad: grad * 0.5)

            # reset hidden states
            if step % self.value_prefix_horizon == 0:
                reward_hidden = self.default_reward_hidden(batch_size)
            tt3 = time.perf_counter()
            # logger.info(f"unroll loop time: recurrent inf {tt2 - tt1}, misc {tt3 - tt2}, total {tt3 - tt1}")
        t4 = time.perf_counter()
        # logger.info(f"MuZero Analyze time: {t4 - tik}. transform + gpu {t1 - tik}, "
        #             f"initial inf {t2 - t1}, consistency {t3 - t2}, unroll {t4 - t3}")

        return MuzeroSampleAnalyzedResult(value=value,
                                          target_value=target_value,
                                          value_prefix=value_prefix,
                                          target_value_prefix=target_value_prefix,
                                          policy_logits=policy_logits,
                                          policy_entropy=policy_entropy,
                                          target_policy=target_policy,
                                          mask=mask,
                                          dynamic_proj=dynamic_proj,
                                          observation_proj=observation_proj)

    @torch.no_grad()
    def reanalyze(self, sample, target, **kwargs):
        if self.current != "reanalyze":
            self.current = "reanalyze"
            self._net.load_state_dict(self.reanalyze_ckpts[0])
        self.net.eval()

        num_unroll_steps, td_steps = self.num_unroll_steps, self.td_steps
        assert sample.on_reset.shape[0] == (1 + num_unroll_steps + td_steps + 1)

        length = 1 + num_unroll_steps + td_steps
        batch_size = sample.on_reset.shape[1]
        num_re = int(np.ceil(batch_size * self.reanalyze_ratio_schedule.eval(self.version)))

        # mask
        done = sample.on_reset[1:].astype(np.int32)
        sample = sample[:-1]
        mask = 1 - (done.cumsum(axis=0) > 0).astype(np.int32)
        policy_version_steps = copy.deepcopy(sample.policy_version_steps)
        sample = recursive_apply(
            sample, lambda x: x if x is None or x.dtype == np.dtype('<U7') else
            (x.reshape(length, batch_size, -1) * mask).astype(x.dtype).reshape(*x.shape))
        sample.policy_version_steps = policy_version_steps

        # reanalyze value and policy
        with torch.no_grad():
            sample.analyzed_result.value[td_steps:, :] = self.reanalyze_value(sample[td_steps:, :])
        if num_re > 0:
            with torch.no_grad():
                # sample.analyzed_result.value[td_steps:, :num_re] = self.reanalyze_value(
                #     sample[td_steps:, :num_re])
                sample.analyzed_result.policy[:num_unroll_steps + 1, :
                                              num_re], sample.analyzed_result.actions[:num_unroll_steps + 1, :
                                                                                      num_re], _ = self.mcts_reanalyze(
                                                                                          sample[:
                                                                                                 num_unroll_steps
                                                                                                 +
                                                                                                 1, :num_re])
        sample = recursive_apply(
            sample, lambda x: x if x is None or x.dtype == np.dtype('<U7') else
            (x.reshape(length, batch_size, -1) * mask).astype(x.dtype).reshape(*x.shape))
        sample.policy_version_steps = policy_version_steps

        # randomize masked actions
        if not self.mcts_use_sampled_actions:
            # use uniform sampling
            random_action = self.generate_random_action((length, batch_size))
            sample.action.x = (sample.action.x.reshape(length, batch_size, -1) * mask +
                               random_action.reshape(length, batch_size, -1) * (1 - mask)).astype(
                                   sample.action.x.dtype).reshape(sample.action.x.shape)
        else:
            # sample masked actions from policy network
            rows_to_mask = [b for b in range(batch_size) if any(mask[:, b] < 1)]
            if len(rows_to_mask) > 0:
                random_action = self.generate_sampled_action(sample[:, rows_to_mask], length)
                action_shape = sample.action.x.shape[2:]
                sample.action.x[:, rows_to_mask] = (
                    sample.action.x[:, rows_to_mask].reshape(length, len(rows_to_mask),
                                                             np.prod(action_shape)) * mask[:, rows_to_mask] +
                    random_action.reshape(length, len(rows_to_mask), np.prod(action_shape)) *
                    (1 - mask[:, rows_to_mask])).astype(sample.action.x.dtype).reshape(
                        length, len(rows_to_mask), *action_shape)

        # reward
        reward = sample.reward

        # target value
        target_value = np.zeros((num_unroll_steps + 1, batch_size, 1))
        target_value += sample.analyzed_result.value[td_steps:] * (self.discount**td_steps)
        for step in range(td_steps):
            target_value += reward[step:step + num_unroll_steps + 1] * (self.discount**step)
        assert (target_value * (1 - mask[:num_unroll_steps + 1])).sum() == 0

        # target value prefix
        target_value_prefix = np.zeros((num_unroll_steps + 1, batch_size, 1))
        for step in range(1, num_unroll_steps + 1):
            if (step - 1) % self.value_prefix_horizon == 0:
                target_value_prefix[step] = reward[step - 1]
            else:
                target_value_prefix[step] = target_value_prefix[step - 1] + reward[step - 1]

        # target policy
        target_policy = (sample.analyzed_result.policy * mask)[:num_unroll_steps + 1]

        # target sampled actions
        target_actions = sample.analyzed_result.actions[:num_unroll_steps + 1]

        mask = mask[:num_unroll_steps + 1]

        sample.analyzed_result = ReanalyzedResult(mask=mask,
                                                  target_policy=target_policy,
                                                  target_actions=target_actions,
                                                  target_value=target_value,
                                                  target_value_prefix=target_value_prefix)

        alive_indicies = [b for b in range(batch_size) if mask[0, b].mean() == 1]
        sample = recursive_apply(sample, lambda x: None if x is None else x[:, alive_indicies])

        sample.policy_version_steps[sample.policy_version_steps < 0] = 0
        sample.policy_version_steps[:] = self.version

        return sample

    def reanalyze_value(self, sample):
        assert not self.net.training
        length, batch_size = sample.on_reset.shape[:2]
        # [t, b, w, h, s, c] -> [t*b, w, h, s, c]
        # obs = recursive_apply(sample.obs, lambda x: x.reshape(-1, *x.shape[2:]))
        # [t*b, s*c, w, h]
        obs = torch.from_numpy(sample.obs.obs).to(self.device).float().flatten(end_dim=1) / 255.0
        # obs = torch.from_numpy(self.rollout_process_observation(obs)).to(self.device).float()
        assert len(obs.shape) == 4
        network_output = self.net.initial_inference(obs)
        value = network_output.value.reshape(length, batch_size, 1)
        return value

    def mcts_reanalyze(self, sample):
        assert not self.net.training
        length, batch_size = sample.on_reset.shape[:2]

        requests = RolloutRequest(
            obs=recursive_apply(sample.obs, lambda x: x.reshape(-1, *x.shape[2:])),
            policy_state=None,
            is_evaluation=np.zeros((length * batch_size, 1), dtype=np.uint8),
            on_reset=sample.on_reset.reshape(length * batch_size, 1),
            # Meta-data below
            client_id=None,
            request_id=None,
            received_time=None,
        )

        result = self.rollout(requests)
        policy = result.analyzed_result.policy.reshape(length, batch_size, self.act_dim)
        actions = result.analyzed_result.actions.reshape(length, batch_size,
                                                         *result.analyzed_result.actions.shape[1:])
        value = result.analyzed_result.value.reshape(length, batch_size, 1)

        return policy, actions, value

    @torch.no_grad()
    def rollout(self, requests: RolloutRequest):
        assert not self.net.training
        if self.current != "rollout":
            self.current = "rollout"
            self._net.load_state_dict(self.rollout_ckpt)
        self.net.eval()

        batch_size = requests.on_reset.shape[0]

        is_evaluation = requests.is_evaluation.reshape(batch_size, 1)
        available_action = None
        if self.use_available_action:
            available_action = requests.obs.available_action.reshape(batch_size, self.action_space.n)
            available_action[available_action.sum(axis=-1) == 0] = 1.

        # uint8 numpy array [b, w, h, s, c] -> 0-1 normalized torch float tensor [b, s*c, w, h]
        obs = torch.from_numpy(requests.obs.obs).to(self.device).float() / 255.0
        assert len(obs.shape) == 4
        # obs = torch.from_numpy(self.rollout_process_observation(requests.obs)).to(self.device).float()
        network_output = self.net.initial_inference(obs)
        hidden_state_roots = network_output.hidden_state
        reward_hidden_roots = network_output.reward_hidden
        value_prefix_pool = network_output.value_prefix
        policy_logits_pool = network_output.policy_logits
        actions_pool = network_output.actions
        if self.mcts_use_sampled_actions:
            # pad zero actions, policy logits and use available mask
            available_action, policy_logits_pool, actions_pool = pad_sampled_actions(
                self.act_dim, actions_pool, policy_logits_pool)

        policy_pool = np.exp(policy_logits_pool)
        if available_action is not None:
            policy_pool[available_action == 0] = 0
        policy_pool = policy_pool / policy_pool.sum(axis=-1).reshape(batch_size, 1)

        temperature = 1 if self.visit_softmax_temperature_fn is None else self.visit_softmax_temperature_fn(
            self.version)
        if self.use_mcts:
            # add exploration noises to policy
            noises = np.random.dirichlet(self.mcts_config.root_dirichlet_alpha * np.ones((self.act_dim,)),
                                         (batch_size,))
            if available_action is not None:
                noises[available_action == 0] = 0
            noises = noises / noises.sum(axis=-1).reshape(batch_size, 1)
            root_exploration_fraction = (1. - is_evaluation) * self.mcts_config.root_exploration_fraction
            policy_pool = policy_pool * (1. - root_exploration_fraction) + noises * root_exploration_fraction

            # tree search
            roots_distributions, roots_value = MCTS(**dict(self.mcts_config._get_kwargs())).search(
                self.net,
                hidden_state_roots,
                reward_hidden_roots,
                value_prefix_pool,
                policy_pool,
                actions_pool,
                available_action=available_action)

            act_prob = np.array(roots_distributions)**(1 / temperature)
            act_prob = act_prob / act_prob.sum(axis=-1).reshape(batch_size, 1)
            value = np.array(roots_value)

            target_policy = np.array(roots_distributions)
            target_policy = target_policy / target_policy.sum(axis=-1).reshape(batch_size, 1)
        else:
            target_policy = act_prob = policy_pool
            value = network_output.value.reshape(-1)

        # perform random actions to warm up training
        if self.version <= self.warm_up_version:
            act_prob = np.ones_like(act_prob) * (1. - is_evaluation) + act_prob * is_evaluation

        if available_action is not None:
            act_prob[available_action == 0] = 0
            target_policy[available_action == 0] = 0
        act_prob = act_prob / act_prob.sum(axis=-1).reshape(batch_size, 1)
        target_policy = target_policy / target_policy.sum(axis=-1).reshape(batch_size, 1)

        # best action for evaluation
        best_action = act_prob.argmax(axis=-1).reshape(batch_size, 1)

        # sample action from exploration policy
        act_dist = Categorical(torch.from_numpy(act_prob))
        action = act_dist.sample()
        action = action.cpu().numpy().reshape(batch_size, 1)

        action = (is_evaluation * best_action + (1 - is_evaluation) * action).astype(np.int32)

        entropy = -(np.log(target_policy + 1e-8) * target_policy).sum(axis=-1).reshape(batch_size, 1)
        actor_entropy = -(np.log(act_prob + 1e-8) * act_prob).sum(axis=-1).reshape(batch_size, 1)

        action = action.reshape(-1).tolist()
        action = np.array(
            [sampled_actions[action_idx] for action_idx, sampled_actions in zip(action, actions_pool)])
        act_prob = act_prob.reshape(batch_size, self.act_dim)
        value = value.reshape(batch_size, 1)

        return RolloutResult(action=Action(action),
                             analyzed_result=AnalyzedResult(policy=target_policy,
                                                            actions=actions_pool,
                                                            value=value,
                                                            entropy=entropy,
                                                            actor_entropy=actor_entropy),
                             policy_state=None)

    @torch.no_grad()
    def generate_sampled_action(self, sample, length):
        """Sample action from current policy 
        """
        obs = self._unfold_stacked_obs(sample.obs.obs[:1]).to(self.device).float()
        obs_batch = obs[:self.stacked_observations]
        _, _, _, _, actions, hidden_state, reward_hidden = self.net.initial_inference(
            self.process_stack_observation(obs_batch), num_sampled_actions=1)
        actions = np.array(actions)[:, 0]
        gen = [actions]
        for step in range(1, length):
            hidden_state = torch.from_numpy(hidden_state).to(self.device).float()
            reward_hidden = (torch.from_numpy(reward_hidden[0]).to(self.device).float(),
                             torch.from_numpy(reward_hidden[1]).to(self.device).float())
            _, _, _, _, actions, hidden_state, reward_hidden = self.net.recurrent_inference(
                hidden_state, reward_hidden, torch.from_numpy(actions).to(self.device), num_sampled_actions=1)
            actions = np.array(actions)[:, 0]
            gen.append(actions)
        gen = np.stack(gen, axis=0)
        return gen

    def generate_random_action(self, shape):
        """Generate random action of shape (*shape, *action_shape).
        """
        if self.action_space.__class__.__name__ == "Discrete":
            random_action = np.random.randint(0, self.action_space.n, (*shape, 1))
        elif self.action_space.__class__.__name__ == "Box":
            high = self.action_space.high.reshape(*tuple([1] * len(shape) + [-1]))
            low = self.action_space.low.reshape(*tuple([1] * len(shape) + [-1]))
            random_action = np.random.uniform(0, 1, (*shape, *self.action_space.shape)) * (high - low) + low
        elif self.action_space.__class__.__name__ == "MultiDiscrete":
            random_action = []
            for d in self.action_space.nvec.reshape(-1):
                random_action.append(np.random.randint(0, d, shape))
            random_action = np.stack(random_action, axis=-1).reshape(*shape, *self.action_space.shape)
        else:
            raise NotImplementedError
        return random_action

    def rollout_process_observation(self, obs):
        raise NotImplementedError

    def analyze_process_observation(self, obs):
        raise NotImplementedError

    def process_stack_observation(self, obs):
        raise NotImplementedError

    def inverse_reward_transform(self, reward_logits):
        '''Convert vectorized reward to scalar reward
        '''
        raise NotImplementedError

    def inverse_value_transform(self, value_logits):
        '''Convert vectorized value to scalar value
        '''
        raise NotImplementedError

    def scalar_reward_loss(self, value_prefix, target_value_prefix):
        '''Compute value prefix loss given vectorized value_prefix and scalar target_value_prefix
        '''
        raise NotImplementedError

    def scalar_value_loss(self, value, target_value):
        '''Compute value prefix loss given vectorized value and scalar target_value
        '''
        raise NotImplementedError
