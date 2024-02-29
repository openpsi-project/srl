"""A simplified environment of Aeroplane Chess useful for testing.
"""
from typing import List
import gym
import numpy as np
import scipy.stats

import api.environment as env_base
import api.env_utils as env_utils


class AerochessEnvironment(env_base.Environment):
    """Our simple environment has a very simple and predictable behaviour:
    - There are `n` players (configurable) all starting at location 0.
    - In each step, each players goes forward by `a_i in [1, 6]` steps while `a` is the action provided.
    - Each player's observation is a vector showing all players' current location.
    - Each player's reward is its rank (`n` for first, and 1 for last) after the step.
    - Reaching location `length` will end the game for the player.
    - No killing is considered.
    """

    def __init__(self, length=10, n=1, max_steps=None):
        self.__length = length
        self.__n = n
        self.__space = env_utils.DiscreteActionSpace(gym.spaces.Discrete(6))
        self.__locations = None
        self.__steps = None
        self.max_steps = max_steps

    @property
    def agent_count(self):
        return self.__n

    @property
    def observation_spaces(self):
        return {}

    @property
    def action_spaces(self):
        return [self.__space for _ in range(self.__n)]

    def reset(self):
        self.__locations = np.zeros(self.__n, dtype=np.int64)
        self.__steps = 0
        return [
            env_base.StepResult(obs={"obs": self.__locations},
                                reward=np.array([0], dtype=np.float32),
                                done=np.array([False], dtype=np.uint8),
                                info={}) for _ in range(self.__n)
        ]

    def step(self, actions: List[env_utils.DiscreteAction]):
        delta = np.array([a.x.item() for a in actions], dtype=np.int64)
        assert np.all(delta > 0), delta
        self.__locations = np.minimum(self.__locations + delta, self.__length)
        rewards = self.__n - scipy.stats.rankdata(self.__locations, method='min') + 1

        self.__steps += 1
        if not self.max_steps:
            return [
                env_base.StepResult(obs={"obs": self.__locations},
                                    reward=np.array([r], dtype=np.float32),
                                    done=np.array([loc >= self.__length], dtype=np.uint8),
                                    info={}) for r, loc in zip(rewards, self.__locations)
            ]
        else:
            if self.__steps < self.max_steps:
                return [
                    env_base.StepResult(obs={"obs": self.__locations},
                                        reward=np.array([r], dtype=np.float32),
                                        done=np.array([False], dtype=np.uint8),
                                        info={}) for r, loc in zip(rewards, self.__locations)
                ]
            else:
                # step = 3, the third step, finish step
                return [
                    env_base.StepResult(obs={"obs": self.__locations},
                                        reward=np.array([r], dtype=np.float32),
                                        done=np.array([True], dtype=np.uint8),
                                        info={}) for r, loc in zip(rewards, self.__locations)
                ]


env_base.register("aerochess", AerochessEnvironment)
