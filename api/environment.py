"""Abstraction of the RL environment and related concepts.

This is basically a clone of the gym interface. The reasons of replicating are:
- Allow easy changing of APIs when necessary.
- Avoid hard dependency on gym.
"""
from typing import List, Union, Dict, Type
import dataclasses
import importlib
import numpy as np

from api import config as config


class Action:
    pass


class ActionSpace:

    def sample(self, *args, **kwargs) -> Action:
        raise NotImplementedError()


class DataAugmenter:
    """DataAugmenter pre-process the generated sample before it is sent to trainers.. Defined per environment.
    """

    def process(self, sample):
        """Relabel sample. Operation should be in-place.
        Args:
            sample (algorithm.trainer.SampleBatch) Sample to be augmented.
        Return:
            augmented_sample (algorithm.trainer.SampleBatch).
        """
        raise NotImplementedError()


class NullAugmenter(DataAugmenter):

    def process(self, sample):
        return sample


@dataclasses.dataclass
class StepResult:
    """Step result for a single agent. In multi-agent scenario, env.step() essentially returns
    List[StepResult].
    """
    obs: Dict
    reward: np.ndarray
    done: np.ndarray
    info: Dict
    truncated: np.ndarray = np.zeros(shape=(1,), dtype=np.uint8)


class Environment:

    @property
    def agent_count(self) -> int:
        raise NotImplementedError()

    @property
    def observation_spaces(self) -> List[dict]:
        """Return a list of observation spaces for all agents.

        Each element in self.observation_spaces is a Dict, which contains
        shapes of observation entries specified by the key.
        Example:
        -------------------------------------------------------------
        self.observation_spaces = [{
            'observation_self': (10, ),
            'box_obs': (9, 15),
        }, {
            'observation_self': (20, ),
            'box_obs': (9, 15),
        }]
        -------------------------------------------------------------
        Observation spaces of different agents can be different.
        In this case, policies *MUST* be *DIFFERENT*
        among agents with different observation dimension.
        """
        raise NotImplementedError()

    @property
    def action_spaces(self) -> List[ActionSpace]:
        """Return a list of action spaces for all agents.

        Each element in self.action_spaces is an instance of
        env_base.ActionSpace, which is basically a wrapped Dict.
        The Dict contains shapes of action entries specified by the key.
        **We force each action entry to be either gym.spaces.Discrete
        or gym.spaces.Box.**
        Example:
        -------------------------------------------------------------
        self.action_spaces = [
            SomeEnvActionSpace(dict(move_x=Discrete(10), move_y=Discrete(10), cursur=Box(2))),
            SomeEnvActionSpace(dict(cursur=Box(2)))
        ]
        -------------------------------------------------------------
        Action spaces of different agents can be different.
        In this case, policies *MUST* be *DIFFERENT*
        among agents with different action output.
        """
        raise NotImplementedError()

    def reset(self) -> List[StepResult]:
        """Reset the environment, and returns a list of step results for all agents.

        Returns:
            List[StepResult]: StepResult with valid Observations only.
        """
        raise NotImplementedError()

    def step(self, actions: List[Action]) -> List[StepResult]:
        """ Consume actions and advance one env step.

        Args:
            actions (List[Action]): Actions of all agents.

        Returns:
            step result (StepResult): An object with 4 members:
            - obs (namedarray): It contains observations, available actions, masks, etc.
            - reward (numpy.ndarray): A numpy array with shape [1].
            - done (numpy.ndarray): A numpy array with shape [1],
                indicating whether an episode is done or an agent is dead.
            - info (namedarray): Customized namedarray recording required summary infos.
        """
        raise NotImplementedError()

    def render(self) -> None:
        pass

    def seed(self, seed):
        """Set a random seed for the environment.

        Args:
            seed (Any): The seed to be set. It could be int,
            str or any other types depending on the implementation of
            the specific environment. Defaults to None.

        Returns:
            Any: The new seed.
        """
        raise NotImplementedError()

    def set_curriculum_stage(self, stage_name: str):
        """Set the environment to be in a certain stage.
        Args:
            stage_name: name of the stage to be set.
        """
        raise NotImplementedError()


ALL_ENVIRONMENT_CLASSES = {}
ALL_ENVIRONMENT_MODULES = {}
ALL_AUGMENTER_CLASSES = {}


def register(name, env_class: Union[Type, str], module=None):
    """Register a environment. If env_class is string, the module is registered implicitly. The corresponding
    module is only imported when the environment is created.
    Args:
        name: A reference name of the environment. Use this name in your experiment configuration.
        env_class: Class of the environment. If passed as string, its source module is required.
        module: String, the module path the find the env_class, ignored when env_class is not a string.

    Raises:
        KeyError: if name is already registered.

    Examples:
        # codespace/implementation/this_is_my_env.py
        class ThisIsMyEnv(api.environment.Environment):
        ...
        register("this-is-my-env", ThisIsMyEnv)
        # OR
        register("this-is-my-env", "ThisIsMyEnv", "codespace.implementation.this_is_my_env")

    """
    if name in ALL_ENVIRONMENT_CLASSES:
        raise KeyError(f"Environment {name} already registered as {ALL_ENVIRONMENT_CLASSES[name]}. "
                       f"But got another register with env_class={env_class} and module={module}")
    if isinstance(env_class, str):
        assert module is not None, "For safe registration, specify module in api.environment.register."
        ALL_ENVIRONMENT_MODULES[name] = module
    ALL_ENVIRONMENT_CLASSES[name] = env_class


def register_relabler(name, relabeler_class):
    ALL_AUGMENTER_CLASSES[name] = relabeler_class


def make(cfg: Union[str, config.Environment]) -> Environment:
    env_type_ = cfg if isinstance(cfg, str) else cfg.type_
    if isinstance(cfg, str):
        cfg = config.Environment(type_=cfg)
    if isinstance(ALL_ENVIRONMENT_CLASSES[env_type_], str):
        if env_type_ not in ALL_ENVIRONMENT_MODULES:
            raise RuntimeError("Module is not registered correctly for safe registration.")
        m_ = importlib.import_module(ALL_ENVIRONMENT_MODULES[env_type_])
        ALL_ENVIRONMENT_CLASSES[env_type_] = getattr(m_, ALL_ENVIRONMENT_CLASSES[env_type_])
    cls = ALL_ENVIRONMENT_CLASSES[env_type_]
    return cls(**cfg.args)


register_relabler("NULL", NullAugmenter)


def make_augmenter(cfg: Union[str, config.DataAugmenter]) -> DataAugmenter:
    augmenter_type = cfg if isinstance(cfg, str) else cfg.type_
    cls = ALL_AUGMENTER_CLASSES[augmenter_type]
    return cls(**cfg.args)
