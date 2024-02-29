from typing import List
import dataclasses
import numpy as np
import random

from onpolicy.envs.hanabi.Hanabi_Env import HanabiEnv
from api import environment
from api.environment import Action, StepResult, ActionSpace
from api.env_utils import DiscreteActionSpace


@dataclasses.dataclass
class HanabiConfig:
    hanabi_name: str
    num_agents: int
    use_obs_instead_of_state: bool = False


class HanabiEnvironment(environment.Environment):
    """Wrapper of Hanabi Env.
    """

    def __init__(self, hanabi_name, num_agents, use_obs_instead_of_state=False, seed=None):
        """Creates an environment with the given game configuration.
        """
        self._seed = seed or random.randint(0, 20000)
        self.__agent_count = num_agents
        self.__env = HanabiEnv(HanabiConfig(hanabi_name, num_agents, use_obs_instead_of_state),
                               seed=self._seed)
        self.__observation_spaces = [{"obs": self.__env.observation_space}]

        self.__curr_agent = None
        self.__recent_rewards = None

    @property
    def agent_count(self) -> int:
        return self.__agent_count

    @property
    def observation_spaces(self) -> List[dict]:
        return self.__observation_spaces

    @property
    def action_spaces(self) -> List[ActionSpace]:
        return [DiscreteActionSpace(aspc) for aspc in self.__env.action_space]

    def reset(self, choose=True):
        obs, state, available_action = self.__env.reset()
        self.__recent_rewards = [0 for _ in range(self.__agent_count)]
        self.__curr_agent = 0  # potential mismatch with original agent index.

        step_results = [None for _ in range(self.agent_count)]
        step_results[self.__curr_agent] = StepResult(
            obs=dict(obs=np.array(obs), state=np.array(state), available_action=np.array(available_action)),
            reward=np.zeros(1),
            done=np.zeros(1, dtype=np.int8),
            info=dict(),
        )
        self.__curr_agent = (self.__curr_agent + 1) % self.agent_count
        return step_results

    def step(self, actions: List[Action]) -> List[StepResult]:
        [action] = [int(action.x.item()) for action in actions if action is not None]
        obs, share_obs, rewards, done, infos, available_actions = self.__env.step([action])
        # Onpolicy-Hanabi return rewards of shape(num_agents, 1)
        self.__recent_rewards.append(rewards[0][0])
        target_players = [self.__curr_agent] if not done else [(self.__curr_agent + i) % self.__agent_count
                                                               for i in range(self.__agent_count)]
        step_results = [None for _ in range(self.__agent_count)]
        self.__curr_agent = (self.__curr_agent + 1) % self.agent_count
        for curr_agent in target_players:
            self.__recent_rewards.pop(0)
            step_results[curr_agent] = StepResult(
                obs=dict(obs=np.array(obs),
                         state=np.array(share_obs),
                         available_action=np.array(available_actions)),
                reward=np.array([sum(self.__recent_rewards)], dtype=np.float32),
                done=np.array([done], dtype=np.uint8),
                info=dict(score=np.array([infos["score"]])),
            )

        return step_results

    def seed(self, seed):
        self.__env.seed(seed)


class SingleAgentHanabiEnvironment(environment.Environment):
    """Wrapper of Hanabi Env.
    Act as single agent.
    """

    def __init__(self, hanabi_name, num_agents, use_obs_instead_of_state=False, seed=None):
        """Creates an environment with the given game configuration.
        """
        self._seed = seed or random.randint(0, 20000)
        self.__agent_count = 1
        self.__env = HanabiEnv(HanabiConfig(hanabi_name, num_agents, use_obs_instead_of_state),
                               seed=self._seed)
        self.__observation_spaces = [{"obs": self.__env.observation_space}]
        self.__action_spaces = [self.__env.action_space]

        self.__step_count = None
        self.__reward_sum = None
        self.__available_action = None
        self.__obs = None
        self.__state = None

    @property
    def env(self):
        return self.__env

    @property
    def agent_count(self) -> int:
        return self.__agent_count

    @property
    def observation_spaces(self) -> List[dict]:
        return self.__observation_spaces

    @property
    def action_spaces(self) -> List[ActionSpace]:
        return self.__action_spaces

    def reset(self, choose=True):
        obs, state, available_action = self.__env.reset()
        self.__step_count = 0
        self.__reward_sum = 0
        self.__obs = obs
        self.__state = state
        self.__available_action = available_action

        step_results = [
            StepResult(
                obs=dict(obs=np.array(obs),
                         state=np.array(state),
                         available_action=np.array(available_action)),
                reward=np.zeros(1),
                done=np.zeros(1, dtype=np.int8),
                info=dict(),
            )
        ]
        return step_results

    def step(self, actions: List[Action]) -> List[StepResult]:
        [action] = [int(action.x.item()) for action in actions if action is not None]
        if self.__available_action[action] == 0:
            # invalid action
            step_results = [
                StepResult(obs=dict(obs=np.array(self.__obs),
                                    state=np.array(self.__state),
                                    available_action=np.array(self.__available_action)),
                           reward=np.array([-self.__reward_sum - 1], dtype=np.float32),
                           done=np.array([True], dtype=np.uint8),
                           info=dict(score=np.array([self.__env.state.score()]),
                                     episode_length=np.array([self.__step_count])))
            ]
            return step_results
        obs, state, rewards, done, infos, available_action = self.__env.step([action])
        self.__step_count += 1
        self.__reward_sum += rewards[0][0]
        self.__obs = obs
        self.__state = state
        self.__available_action = available_action
        step_results = [
            StepResult(obs=dict(obs=np.array(obs),
                                state=np.array(state),
                                available_action=np.array(available_action)),
                       reward=np.array([rewards[0][0]], dtype=np.float32),
                       done=np.array([done], dtype=np.uint8),
                       info=dict(score=np.array([infos["score"]]),
                                 episode_length=np.array([self.__step_count])))
        ]

        return step_results

    def seed(self, seed):
        self.__env.seed(seed)
