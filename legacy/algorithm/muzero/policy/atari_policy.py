from einops import rearrange
from typing import List, Union
import numpy as np
import torch

from legacy.algorithm.muzero.model.EfficientZeroNet import EfficientZeroNet
from legacy.algorithm.muzero.policy.base_policy import MuzeroBasePolicy
from legacy.algorithm.muzero.utils.scalar_transform import DiscreteSupport, inverse_scalar_transform, scalar_transform, phi
from legacy.algorithm.muzero.utils.image_transform import Transforms
from legacy.algorithm.muzero.utils.utils import prepare_observation_lst, LinearSchedule
from api.policy import register


class MuzeroAtariPolicy(MuzeroBasePolicy):

    def __init__(
            self,
            act_dim: int,
            action_space,
            obs_shape: Union[List[int], int],
            discount: float = 0.997,
            # tree search
            use_mcts: bool = True,
            value_delta_max: float = 0.01,
            num_simulations: int = 50,
            root_dirichlet_alpha: float = 0.3,
            root_exploration_fraction: float = 0.25,
            pb_c_base: int = 19652,
            pb_c_init: float = 1.25,
            num_threads: int = 16,
            # network initialization
            init_zero: bool = True,
            # frame skip & stack observation
            frame_skip: int = 4,
            stacked_observations: int = 4,
            gray_scale: bool = False,
            # value prefix
            lstm_hidden_size: int = 512,
            use_value_prefix: bool = True,
            value_prefix_horizon: int = 5,
            # siamese
            proj_hid: int = 1024,
            proj_out: int = 1024,
            pred_hid: int = 512,
            pred_out: int = 1024,
            # image augmentation
            use_augmentation: bool = True,
            augmentation: List[str] = ['shift', 'intensity'],
            # reanalyze
            reanalyze_ratio_schedule=LinearSchedule(v_init=0., v_end=0., t_end=1),
            td_steps: int = 5,
            num_unroll_steps: int = 5,
            # value & reward transform
            value_support: DiscreteSupport = DiscreteSupport(-300, 300, delta=1),
            reward_support: DiscreteSupport = DiscreteSupport(-300, 300, delta=1),
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
            seed: int = 0,
            **kwargs):

        discount = discount**frame_skip

        self.obs_shape = obs_shape
        self.act_dim = act_dim
        if action_space is None:
            import gym.spaces
            action_space = gym.spaces.Discrete(self.act_dim)
        self.action_space = action_space
        self.frame_skip = frame_skip
        self.stacked_observations = stacked_observations
        self.gray_scale = gray_scale
        self.image_channels = 1 if self.gray_scale else 3

        # efficient zero neural network
        self.bn_mt = 0.1
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        if self.gray_scale:
            self.channels = 32
        self.reduced_channels_reward = 16  # x36 Number of channels in reward head
        self.reduced_channels_value = 16  # x36 Number of channels in value head
        self.reduced_channels_policy = 16  # x36 Number of channels in policy head
        self.resnet_fc_reward_layers = [
            32
        ]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [
            32
        ]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [
            32
        ]  # Define the hidden layers in the policy head of the prediction network
        self.downsample = True
        self.init_zero = init_zero
        self.state_norm = False

        # value prefix
        self.lstm_hidden_size = lstm_hidden_size

        # reward & value transform
        self.value_support = value_support
        self.reward_support = reward_support

        # image augmentation
        self.use_augmentation = use_augmentation
        self.augmentation = augmentation

        neural_network = EfficientZeroNet(self.obs_shape,
                                          self.act_dim,
                                          self.action_space,
                                          self.blocks,
                                          self.channels,
                                          self.reduced_channels_reward,
                                          self.reduced_channels_value,
                                          self.reduced_channels_policy,
                                          self.resnet_fc_reward_layers,
                                          self.resnet_fc_value_layers,
                                          self.resnet_fc_policy_layers,
                                          self.reward_support.size,
                                          self.value_support.size,
                                          self.downsample,
                                          self.inverse_value_transform,
                                          self.inverse_reward_transform,
                                          self.lstm_hidden_size,
                                          bn_mt=self.bn_mt,
                                          proj_hid=proj_hid,
                                          proj_out=proj_out,
                                          pred_hid=pred_hid,
                                          pred_out=pred_out,
                                          init_zero=self.init_zero,
                                          state_norm=self.state_norm,
                                          mcts_use_sampled_actions=mcts_use_sampled_actions,
                                          mcts_num_sampled_actions=mcts_num_sampled_actions)

        super(MuzeroAtariPolicy, self).__init__(
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
            stacked_observations=stacked_observations,
            # value_prefix
            use_value_prefix=use_value_prefix,
            value_prefix_horizon=value_prefix_horizon,
            lstm_hidden_size=lstm_hidden_size,
            # image augmentation
            use_augmentation=use_augmentation,
            # reanalyze
            reanalyze_ratio_schedule=reanalyze_ratio_schedule,
            td_steps=td_steps,
            num_unroll_steps=num_unroll_steps,
            # actor exploration
            visit_softmax_temperature_fn=visit_softmax_temperature_fn,
            warm_up_version=warm_up_version,
            # consistency
            proj_out=proj_out,
            pred_out=pred_out,
            # model
            neural_network=neural_network,
            rollout_update_interval=rollout_update_interval,
            reanalyze_update_interval=reanalyze_update_interval,
            # sampled muzero
            mcts_use_sampled_actions=mcts_use_sampled_actions,
            mcts_num_sampled_actions=mcts_num_sampled_actions,
            # others
            seed=seed,
            **kwargs)

        if self.use_augmentation:
            self.transform = Transforms(self.augmentation,
                                        image_shape=(self.obs_shape[1], self.obs_shape[2])).transform

    def inverse_reward_transform(self, reward_logits):
        return inverse_scalar_transform(reward_logits, self.reward_support, rescale=True)

    def inverse_value_transform(self, value_logits):
        return inverse_scalar_transform(value_logits, self.value_support, rescale=True)

    # def rollout_process_observation(self, obs):
    #     """Note that EfficientZero converts numpy array to string to save memory storage, which partially changes the observations
    #     """
    #     obs = np.swapaxes(obs.obs, 1, 3)  # [b, w, h, s, c] -> [b, s, h, w, c]
    #     assert (len(obs.shape) == 5), (obs.shape)  # obs is of shape (B, S, H, W, C)
    #     obs = prepare_observation_lst(obs)
    #     assert (obs.shape[1:] == self.obs_shape), (obs.shape, self.obs_shape)  # self.obs_shape is (s*c, w, h), (12, 96, 96) in this case
    #     assert obs.max() - obs.min() > 100
    #     obs = obs / 255.0
    #     return obs  # np.array, float, shape [bs, s*c, w, h]

    # def analyze_process_observation(self, obs):
    #     """To save cpu-gpu data transfer time, only take necessary observations and transfer uint8 instead of float.
    #     """
    #     # [t, b, w, h, s, c] -> [t, b, s, h, w, c]
    #     obs = torch.permute(obs.obs, [0, 1, 4, 3, 2, 5])
    #     assert len(obs.shape) == 6, obs.shape
    #     a = obs[0, :, :-1, :, :, :].transpose(0, 1)  # [s-1, b, h, w, c]
    #     b = obs[:, :, -1, :, :, :]  # [t, b, h, w, c]
    #     obs = torch.cat([a, b], 0)
    #     obs = rearrange(obs, 't b h w c -> t b c h w', c=self.image_channels)
    #     return obs.float().contiguous()
    # obs = np.swapaxes(obs.obs, 2, 4)
    # assert (len(obs.shape) == 6), (obs.shape)  # obs is of shape (T, B, S, H, W, C)
    # obs = np.concatenate([obs[0, :, :-1, :, :, :].transpose(1, 0, 2, 3, 4), obs[:, :, -1, :, :, :]],
    #                      axis=0)  # (T', B, H, W, C)
    # obs = torch.from_numpy(obs)
    # obs = rearrange(obs, 't b h w c -> t b c h w', c=self.image_channels)
    # return obs.to(dtype=torch.uint8)

    def process_stack_observation(self, obs: torch.Tensor):
        """transform shape to do initial inference
        """
        obs = rearrange(obs,
                        's b c h w -> b (s c) h w',
                        s=self.stacked_observations,
                        c=self.image_channels,
                        h=self.obs_shape[1],
                        w=self.obs_shape[2]) / 255.0
        return obs

    def scalar_reward_loss(self, value_prefix, target_value_prefix):
        '''Compute value prefix loss given vectorized value_prefix and scalar target_value_prefix
        '''
        transformed_target_value_prefix = scalar_transform(target_value_prefix.squeeze(-1),
                                                           self.reward_support)
        target_value_prefix_phi = phi(transformed_target_value_prefix, self.reward_support)
        value_prefix_loss = -(torch.log_softmax(value_prefix, dim=-1) * target_value_prefix_phi).sum(-1)
        return value_prefix_loss

    def scalar_value_loss(self, value, target_value):
        '''Compute value prefix loss given vectorized value and scalar target_value
        '''
        transformed_target_value = scalar_transform(target_value.squeeze(-1), self.value_support)
        target_value_phi = phi(transformed_target_value, self.value_support)
        value_loss = -(torch.log_softmax(value, dim=-1) * target_value_phi).sum(-1)
        return value_loss


register('muzero-atari', MuzeroAtariPolicy)
