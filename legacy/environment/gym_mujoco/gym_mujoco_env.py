from typing import List
import copy
import gym
import numpy as np

import api.environment as env_base
import api.env_utils as env_utils


class GymMuJoCoEnvironment(env_base.Environment):

    def __init__(self,
                 scenario: str,
                 action_squash_type=None,
                 seed=None,
                 discretize=False,
                 discrete_num_bins=7,
                 **kwargs):
        self.__env = gym.make(scenario, **kwargs)
        try:
            self.__env.seed(seed)
        except AttributeError:
            pass

        self.__action_squash_type = action_squash_type
        assert action_squash_type in ['clip', 'tanh', None]

        self.action_high = self.__env.action_space.high
        self.action_low = self.__env.action_space.low
        self.action_scale = (self.action_high - self.action_low) / 2
        self.action_center = (self.action_high + self.action_low) / 2

        self.discretize = discretize
        self.discrete_num_bins = discrete_num_bins
        if self.discretize:
            assert (self.discrete_num_bins > 1), (
                f"Expected use more than one bins when using discretized action space, but get {self.discrete_num_bins}"
            )
            self.__action_space = env_utils.DiscreteActionSpace(
                gym.spaces.MultiDiscrete([
                    self.discrete_num_bins,
                ] * self.__env.action_space.shape[0]))
        else:
            self.__action_space = env_utils.ContinuousActionSpace(self.__env.action_space)

        self.__episode_return = np.zeros(1, dtype=np.float32)
        self.__episode_length = np.zeros(1, dtype=np.int32)

    @property
    def agent_count(self):
        return 1

    @property
    def action_spaces(self):
        return [self.__action_space]

    def reset(self, seed=None):
        try:
            self.__env.seed(seed)
            # FIXME: gym 0.22 acceptd `seed` argument in reset and deleted `self.seed`
            obs = self.__env.reset()
        except AttributeError:
            obs, _ = self.__env.reset(seed=seed)
        self.__episode_length[:] = 0
        self.__episode_return[:] = 0
        return [
            env_base.StepResult(dict(obs=obs), np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.uint8),
                                dict())
        ]

    def step(self, actions: List[env_utils.ContinuousAction]) -> List[env_base.StepResult]:
        assert len(actions) == 1

        a = actions[0].x
        if self.discretize:
            a = 2 * a / (self.discrete_num_bins - 1) - 1
            a = a * self.action_scale + self.action_center
        # do not squash action if elf.__action_squash_type is None
        elif self.__action_squash_type == 'tanh':
            a = np.tanh(a) * self.action_scale + self.action_center
        elif self.__action_squash_type == 'clip':
            a = np.clip(a, -1, 1) * self.action_scale + self.action_center

        obs, r, done, *truncated, _ = self.__env.step(a)
        if len(truncated) > 0:
            done = done or truncated[0]
        self.__episode_length += 1
        self.__episode_return += r
        info = dict(episode_length=copy.deepcopy(self.__episode_length),
                    episode_return=copy.deepcopy(self.__episode_return))
        return [
            env_base.StepResult(dict(obs=obs), np.array([float(r)], dtype=np.float32),
                                np.array([bool(done)], dtype=np.uint8), info)
        ]

    def render(self):
        self.__env.render()

    def seed(self, seed=None):
        self.__env.seed(seed)
