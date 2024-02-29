import gym
import itertools
import math
import torch

import numpy as np
import torch.nn as nn

from legacy.algorithm.modules import mlp
from legacy.algorithm.muzero.model.base_net import BaseNet, renormalize
from legacy.algorithm.muzero.utils.act import ACTLayer


def init_fc(k, p):
    if 'weight' in k and len(p.data.shape) >= 2:
        # filter out layer norm weights
        nn.init.orthogonal_(p.data, gain=math.sqrt(2))
    if 'bias' in k:
        nn.init.zeros_(p.data)


class MLPBase(nn.Module):

    def __init__(self, input_dim, hidden_layers, output_dim, init_zero=False):
        super().__init__()

        self.mlp = mlp([input_dim] + hidden_layers, activation=nn.ReLU, layernorm=True)
        if output_dim == -1:
            self.fc = lambda x: x
        else:
            self.fc = nn.Linear(hidden_layers[-1], output_dim)
        for k, p in self.mlp.named_parameters():
            init_fc(k, p)
        if output_dim != -1:
            for k, p in self.fc.named_parameters():
                init_fc(k, p)
        if init_zero:
            self.fc.weight.data.fill_(0.)
            self.fc.bias.data.fill_(0.)

    def forward(self, x):
        return self.fc(self.mlp(x))


class ResidualBlock(nn.Module):
    """A ResNet v2 style pre-activation residual block modified with fully-connected layers. 
    Used by Sampled MuZero.
    Reference: http://arxiv.org/abs/2104.06303. Appendix A.
    """

    def __init__(self, hidden_dim, activation=nn.ReLU):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = activation()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = activation()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        for fc in [self.fc1, self.fc2]:
            for k, p in fc.named_parameters():
                init_fc(k, p)

    def forward(self, x):
        residual = x
        x1 = self.fc1(self.act1(self.norm1(x)))
        x2 = self.fc2(self.act2(self.norm2(x1)))
        return residual + x2


class ResidualTower(nn.Module):
    """A residual tower consists of N residual blocks.
    """

    def __init__(self, num_blocks, hidden_dim, activation=nn.ReLU):
        super().__init__()

        blocks = [ResidualBlock(hidden_dim, activation=activation) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class RepresentationNetwork(nn.Module):

    def __init__(self, obs_dim, state_dim, num_blocks):
        super().__init__()

        self.obs_block = nn.Sequential(nn.Linear(obs_dim, state_dim), nn.LayerNorm(state_dim), nn.Tanh())
        self.res_tower = ResidualTower(num_blocks, state_dim)

    def forward(self, obs):
        x = self.obs_block(obs)
        state = self.res_tower(x)
        return state


class DynamicsNetwork(nn.Module):

    def __init__(self, state_dim, act_dim, reward_support_size, init_zero, reward_fc_layers, num_blocks):
        super().__init__()

        self.action_block = nn.Sequential(nn.Linear(act_dim, state_dim), nn.LayerNorm(state_dim), nn.ReLU())
        self.res_tower = ResidualTower(num_blocks, state_dim)
        self.reward_fc = MLPBase(state_dim, reward_fc_layers, reward_support_size, init_zero=init_zero)

    def forward(self, state, action):
        action_emb = self.action_block(action)
        next_state = self.res_tower(state + action_emb)
        reward = self.reward_fc(next_state)
        return next_state, reward


class PredictionNetwork(nn.Module):

    def __init__(self, state_dim, action_space, value_support_size, init_zero, value_fc_layers,
                 policy_fc_layers):
        super().__init__()

        self.action_space = action_space

        self.value_fc = MLPBase(state_dim, value_fc_layers, value_support_size, init_zero=init_zero)
        self.policy_fc = MLPBase(state_dim, policy_fc_layers, -1, init_zero=False)
        self.act_layer = ACTLayer(action_space, policy_fc_layers[-1])

    def forward(self, state):
        value = self.value_fc(state)
        action_sampler = self.act_layer(self.policy_fc(state))
        return action_sampler, value


class MuzeroMLPNet(BaseNet):

    def __init__(
        self,
        obs_dim,
        act_dim,
        action_space,
        state_dim,
        value_fc_layers,
        policy_fc_layers,
        reward_fc_layers,
        reward_support_size,
        value_support_size,
        inverse_value_transform,
        inverse_reward_transform,
        init_zero=True,
        proj_hid=256,
        proj_out=256,
        pred_hid=64,
        pred_out=256,
        num_blocks=10,
        mcts_use_sampled_actions=False,
        mcts_num_sampled_actions=0,
    ):
        super(MuzeroMLPNet, self).__init__(act_dim,
                                           action_space,
                                           inverse_value_transform,
                                           inverse_reward_transform,
                                           1,
                                           mcts_use_sampled_actions=mcts_use_sampled_actions,
                                           mcts_num_sampled_actions=mcts_num_sampled_actions)

        self.obs_dim = obs_dim
        self.state_dim = state_dim

        self.representation_network = RepresentationNetwork(
            obs_dim=obs_dim,
            state_dim=state_dim,
            num_blocks=num_blocks,
        )

        dynamics_act_dim = -1
        if isinstance(self.action_space, gym.spaces.Discrete):
            dynamics_act_dim = self.action_space.n
        elif isinstance(self.action_space, gym.spaces.Box):
            dynamics_act_dim = self.action_space.shape[0]
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            dynamics_act_dim = np.prod(self.action_space.shape)
        else:
            raise NotImplementedError(
                f"Action embedding for {self.action_space} is not implemented. Please specifcy `dynamics_act_dim` for {self.action_space} here."
            )

        self.dynamics_network = DynamicsNetwork(
            state_dim=state_dim,
            act_dim=dynamics_act_dim,
            reward_support_size=reward_support_size,
            init_zero=init_zero,
            reward_fc_layers=reward_fc_layers,
            num_blocks=num_blocks,
        )

        self.prediction_network = PredictionNetwork(
            state_dim=state_dim,
            action_space=action_space,
            value_support_size=value_support_size,
            init_zero=init_zero,
            value_fc_layers=value_fc_layers,
            policy_fc_layers=policy_fc_layers,
        )

        self.projection_in_dim = self.state_dim
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.projection = nn.Sequential(nn.Linear(self.projection_in_dim, self.proj_hid),
                                        nn.BatchNorm1d(self.proj_hid), nn.ReLU(),
                                        nn.Linear(self.proj_hid,
                                                  self.proj_hid), nn.BatchNorm1d(self.proj_hid), nn.ReLU(),
                                        nn.Linear(self.proj_hid, self.proj_out),
                                        nn.BatchNorm1d(self.proj_out))
        self.projection_head = nn.Sequential(
            nn.Linear(self.proj_out, self.pred_hid),
            nn.BatchNorm1d(self.pred_hid),
            nn.ReLU(),
            nn.Linear(self.pred_hid, self.pred_out),
        )

    def prediction(self, state):
        action_sampler, value = self.prediction_network(state)
        return action_sampler, value

    def representation(self, observation):
        state = self.representation_network(observation)
        return state

    def dynamics(self, state, reward_hidden, action):
        if isinstance(self.action_space, gym.spaces.Discrete):
            # transform to onehot vector
            action = torch.scatter(torch.zeros(*action.shape[:-1],
                                               self.action_space.n).to(action).to(dtype=torch.float32),
                                   dim=-1,
                                   src=torch.ones_like(action).to(dtype=torch.float32),
                                   index=action.long())
        elif isinstance(self.action_space, gym.spaces.Box):
            pass
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            # normalize each dimension to [-1, 1]
            action = action.float().reshape(*action.shape[:-len(self.action_space.shape)],
                                            np.prod(self.action_space.shape)) / torch.from_numpy(
                                                (self.action_space.nvec - 1)).reshape(
                                                    *([
                                                        1,
                                                    ] * (len(action.shape) - len(self.action_space.shape))),
                                                    -1).to(device=action.device, dtype=torch.float32) * 2 - 1.
        else:
            raise NotImplementedError(f"Action embedding for {self.action_space} is not implemented.")
        next_state, reward = self.dynamics_network(state, action)

        return next_state, reward_hidden, reward

    def get_params_mean(self):
        raise NotImplementedError

    def project(self, hidden_state, with_grad=True):
        # only the branch of proj + pred can share the gradients
        hidden_state = hidden_state.view(-1, self.projection_in_dim)
        proj = self.projection(hidden_state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head(proj)
            return proj
        else:
            return proj.detach()
