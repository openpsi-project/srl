from typing import Union
import gym
import numpy as np

from base.namedarray import NamedArray
import api.environment as environment


# @namedarray
class DiscreteAction(NamedArray, environment.Action):

    def __init__(self, x: np.ndarray):
        super(DiscreteAction, self).__init__(x=x)

    def __eq__(self, other):
        assert isinstance(other, DiscreteAction), \
            "Cannot compare DiscreteAction to object of class{}".format(other.__class__.__name__)
        return self.key == other.key

    def __hash__(self):
        return hash(self.x.item())

    @property
    def key(self):
        return self.x.item()


class DiscreteActionSpace(environment.ActionSpace):

    def __init__(self,
                 space: Union[gym.spaces.Discrete, gym.spaces.MultiDiscrete],
                 shared=False,
                 n_agents=-1):
        """Discrete Action Space Wrapper.
        Args:
            space: Action space of one agent. can be (gym.spaces.Discrete) or (gym.spaces.MultiDiscrete)
            shared: concatenate action space for multiple agents.
            n_agents: number of agents. Effective only if shared=True.
        """
        self.__shared = shared
        self.__n_agents = n_agents
        if shared and n_agents == -1:
            raise ValueError("n_agents must be given to a shared action space.")
        self.__space = space
        assert isinstance(space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)), type(space)

    @property
    def n(self):
        return self.__space.n if isinstance(self.__space, gym.spaces.Discrete) else self.__space.nvec

    def sample(self, available_action: np.ndarray = None) -> DiscreteAction:
        if available_action is None:
            if isinstance(self.__space, gym.spaces.Discrete):
                if self.__shared:
                    x = np.array([[self.__space.sample()] for _ in range(self.__n_agents)], dtype=np.int32)
                else:
                    x = np.array([self.__space.sample()], dtype=np.int32)
            else:
                if self.__shared:
                    x = np.array([self.__space.sample() for _ in range(self.__n_agents)], dtype=np.int32)
                else:
                    x = np.array(self.__space.sample(), dtype=np.int32)
            return DiscreteAction(x)
        else:
            if self.__shared:
                assert available_action.shape == (self.__n_agents, self.__space.n)
                x = []
                for agent_idx in range(self.__n_agents):
                    a_x = self.__space.sample()
                    while not available_action[agent_idx, a_x]:
                        a_x = self.__space.sample()
                    x.append([a_x])
                x = np.array(x, dtype=np.int32)
            else:
                assert available_action.shape == (self.__space.n,)
                x = self.__space.sample()
                while not available_action[x]:
                    x = self.__space.sample()
                x = np.array([x], dtype=np.int32)
            return DiscreteAction(x)


class ContinuousAction(NamedArray, environment.Action):

    def __init__(self, x=np.ndarray):
        super(ContinuousAction, self).__init__(x=x)

    def __eq__(self, other):
        assert isinstance(other, ContinuousAction), \
            "Cannot compare ContinuousAction to object of class{}".format(other.__class__.__name__)
        return self.key == other.key

    @property
    def key(self):
        return self.x


class ContinuousActionSpace(environment.ActionSpace):

    def __init__(self, space: gym.spaces.Box, shared=False, n_agents=-1):
        """Continuous Action Space wrapper.
        Args:
            space: Action space of a single agent. Must be of type gym.spaces.Box.
            shared: concatenate action space for multiple agents.
            n_agents: number of agents. Effective only if shared=True.
        """
        self.__shared = shared
        self.__n_agents = n_agents
        if shared and n_agents == -1:
            raise ValueError("n_agents must be given to a shared action space.")
        self.__space = space
        assert isinstance(space, gym.spaces.Box) and len(space.shape) == 1, type(space)

    @property
    def n(self):
        return self.__space.shape[0]

    def sample(self) -> ContinuousAction:
        if self.__shared:
            x = np.stack([self.__space.sample() for _ in range(self.__n_agents)])
        else:
            x = self.__space.sample()
        return ContinuousAction(x)
