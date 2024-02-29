from typing import List
import torch

from legacy.algorithm.muzero.model.MuzeroMLPNet import MuzeroMLPNet
from legacy.algorithm.muzero.policy.base_policy import MuzeroBasePolicy
from legacy.algorithm.muzero.utils.scalar_transform import DiscreteSupport, inverse_scalar_transform, phi
from legacy.algorithm.muzero.utils.utils import LinearSchedule
from api.policy import register


class MuzeroMLPPolicy(MuzeroBasePolicy):

    def __init__(
            self,
            action_space,
            act_dim: int,
            obs_dim: int,
            discount: float = 0.999,
            # tree search
            use_mcts: bool = True,
            value_delta_max: float = 1.01,
            num_simulations: int = 100,
            root_dirichlet_alpha: float = 0.3,
            root_exploration_fraction: float = 0.25,
            pb_c_base: int = 19652,
            pb_c_init: float = 1.25,
            num_threads: int = 16,
            # network initialization
            state_dim: int = 128,
            init_zero: bool = True,
            value_fc_layers: List = [64, 32],
            policy_fc_layers: List = [64, 32],
            reward_fc_layers: List = [64, 32],
            num_blocks: int = 10,
            # reanalyze
            reanalyze_ratio_schedule=LinearSchedule(v_init=1., v_end=1., t_end=1),
            td_steps: int = 10,
            num_unroll_steps: int = 5,
            # value & reward transform
            value_support: DiscreteSupport = DiscreteSupport(-100, 100, delta=1),
            reward_support: DiscreteSupport = DiscreteSupport(-100, 100, delta=1),
            # actor exploration
            visit_softmax_temperature_fn=None,
            warm_up_version: int = 2000,
            # rollout & reanalyze network update interval
            rollout_update_interval: int = 100,
            reanalyze_update_interval: int = 200,
            # sampled muzero
            mcts_use_sampled_actions: bool = False,
            mcts_num_sampled_actions: int = 0,
            # others
            use_available_action: bool = False,
            seed: int = 1,
            # use global state or local observation
            obs_type: str = "local",
            **kwargs):

        self.obs_type = obs_type
        assert self.obs_type in ['local', 'global']

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_space = action_space

        # reward & value transform
        self.value_support = value_support
        self.reward_support = reward_support

        neural_network = MuzeroMLPNet(action_space=action_space,
                                      obs_dim=self.obs_dim,
                                      act_dim=self.act_dim,
                                      state_dim=state_dim,
                                      value_fc_layers=value_fc_layers,
                                      policy_fc_layers=policy_fc_layers,
                                      reward_fc_layers=reward_fc_layers,
                                      num_blocks=num_blocks,
                                      reward_support_size=self.reward_support.size,
                                      value_support_size=self.value_support.size,
                                      inverse_value_transform=self.inverse_value_transform,
                                      inverse_reward_transform=self.inverse_reward_transform,
                                      init_zero=init_zero,
                                      mcts_use_sampled_actions=mcts_use_sampled_actions,
                                      mcts_num_sampled_actions=mcts_num_sampled_actions)

        super(MuzeroMLPPolicy, self).__init__(
            act_dim=act_dim,
            action_space=action_space,
            discount=discount,
            # tree search
            use_mcts=use_mcts,
            value_delta_max=value_delta_max,
            num_simulations=num_simulations,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_exploration_fraction=root_exploration_fraction,
            pb_c_base=pb_c_base,
            pb_c_init=pb_c_init,
            num_threads=num_threads,
            # stack observations
            stacked_observations=1,
            # value_prefix
            use_value_prefix=False,
            value_prefix_horizon=1,
            lstm_hidden_size=1,
            # image augmentation
            use_augmentation=False,
            # reanalyze
            reanalyze_ratio_schedule=reanalyze_ratio_schedule,
            td_steps=td_steps,
            num_unroll_steps=num_unroll_steps,
            # actor exploration
            visit_softmax_temperature_fn=visit_softmax_temperature_fn,
            warm_up_version=warm_up_version,
            # consistency
            proj_out=1,
            pred_out=1,
            # model
            neural_network=neural_network,
            rollout_update_interval=rollout_update_interval,
            reanalyze_update_interval=reanalyze_update_interval,
            # sampled muzero
            mcts_use_sampled_actions=mcts_use_sampled_actions,
            mcts_num_sampled_actions=mcts_num_sampled_actions,
            # available actions
            use_available_action=use_available_action,
            # others
            seed=seed,
            **kwargs)

    def inverse_reward_transform(self, reward_logits):
        return inverse_scalar_transform(reward_logits, self.reward_support)

    def inverse_value_transform(self, value_logits):
        return inverse_scalar_transform(value_logits, self.value_support)

    def rollout_process_observation(self, obs):
        if self.obs_type == 'local':
            obs = obs.obs
        else:
            obs = obs.state
        assert len(obs.shape) == 2  # obs is of shape (B, D)
        return obs

    def analyze_process_observation(self, obs):
        if self.obs_type == 'local':
            obs = obs.obs
        else:
            obs = obs.state
        assert len(obs.shape) == 3  # obs is of shape (T, B, D)
        return obs

    def process_stack_observation(self, obs: torch.Tensor):
        """transform shape to do initial inference
        """
        assert len(obs.shape) == 3 and obs.shape[0] == 1  # obs is of shape (1, B, D)
        return obs[0]

    def scalar_reward_loss(self, value_prefix, target_value_prefix):
        '''Compute value prefix loss given vectorized value_prefix and scalar target_value_prefix
        '''
        target_value_prefix_phi = phi(target_value_prefix.squeeze(-1), self.reward_support)
        value_prefix_loss = -(torch.log_softmax(value_prefix, dim=-1) * target_value_prefix_phi).sum(-1)
        return value_prefix_loss

    def scalar_value_loss(self, value, target_value):
        '''Compute value prefix loss given vectorized value and scalar target_value
        '''
        target_value_phi = phi(target_value.squeeze(-1), self.value_support)
        value_loss = -(torch.log_softmax(value, dim=-1) * target_value_phi).sum(-1)
        return value_loss


register('muzero-mlp', MuzeroMLPPolicy)
