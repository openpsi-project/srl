"""Simple wrapper around the Atari environments provided by gym.
"""
from typing import List, Union, Optional
import collections
import gym
import logging
import numpy as np
import os
import time

from .atari_wrappers import make_atari
import api.environment as env_base
import api.env_utils as env_utils

logger = logging.getLogger("env-atari")

_HAS_DISPLAY = len(os.environ.get("DISPLAY", "").strip()) > 0


class AtariEnvironment(env_base.Environment):

    def __init__(
        self,
        game_name,
        render: bool = False,
        pause: bool = False,
        noop_max: int = 0,
        episode_life: bool = False,
        clip_reward: bool = False,
        frame_skip: int = 1,
        stacked_observations: Union[int, None] = None,
        max_episode_steps: int = 108000,
        gray_scale: bool = False,
        obs_shape: Union[List[int], None] = None,
        scale: bool = False,
        seed: Optional[int] = None,
        obs_include_last_action: bool = False,
        stacked_last_actions: int = 1,
        obs_include_last_reward: bool = False,
        full_action_space: bool = False,
        epsilon: Optional[float] = None,
    ):
        """Atari environment

        DQN training configuration:
            - noop_max: 30
            - episode_life: True
            - clip_reward: True
            - frame_skip: 4
            - stacked_observations: 4
            - max_episode_steps: 108000
            - gray_scale: True
            - obs_shape: (84, 84)

        R2D2 training configuration:
            - noop_max: 30
            - episode_life: False
            - clip_reward: False
            - frame_skip: 4
            - stacked_observations: 4
            - max_episode_steps: 108000
            - gray_scale: True
            - obs_shape: (84, 84)
            - obs_include_last_action: True
            - obs_include_last_reward: True

        Parameters
        ----------
        noop_max: int
            upon reset, do no-op action for a number of steps in [1, noop_max]
        episode_life: bool
            terminal upon loss of life
        clip_reward: bool
            reward -> sign(reward)
        frame_skip: int
            repeat the action for `frame_skip` steps and return max of last two frames
        max_episode_steps: int
            episode length
        gray_scale: bool
            use gray image observation
        obs_shape: list
            resize observation to `obs_shape`
        scale: bool
            scale frames to [0, 1]
        obs_include_last_action:
            include one-hot action in observation dict
        stacked_last_actions:
            stack k latest one-hot actions
        obs_include_last_reward:
            include last reward in observation dict
        """
        self.game_name = game_name
        self.__render = render
        self.__pause = pause
        self.__env = make_atari(
            env_id=game_name,
            seed=seed,
            noop_max=noop_max,
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            episode_life=episode_life,
            obs_shape=obs_shape,
            gray_scale=gray_scale,
            clip_reward=clip_reward,
            stacked_observations=stacked_observations,
            scale=scale,
            full_action_space=full_action_space,
        )
        self.__frame_skip = frame_skip

        self.__obs_include_last_action = obs_include_last_action
        self.__stacked_last_actions = stacked_last_actions
        self.__last_actions_queue = collections.deque(maxlen=stacked_last_actions)
        self.__obs_include_last_reward = obs_include_last_reward

        self.__step_count = np.zeros(1, dtype=np.int32)
        self.__episode_return = np.zeros(1, dtype=np.float32)

        self.__epsilon = epsilon

    @property
    def agent_count(self) -> int:
        return 1  # We are a simple Atari environment here.

    @property
    def observation_spaces(self):
        base_space = {"obs": self.__env.observation_space.shape}
        if self.__obs_include_last_action:
            base_space['action'] = (self.__stacked_last_actions * self.__env.action_space.n,)
        if self.__obs_include_last_reward:
            base_space['reward'] = (1,)
        return [base_space]

    @property
    def action_spaces(self):
        return [env_utils.DiscreteActionSpace(self.__env.action_space)]

    def _get_obs(self, frame, action, reward):
        self.__last_action[:] = 0
        if action is not None:
            self.__last_action[action] = 1
        self.__last_actions_queue.append(self.__last_action)

        obs = dict(obs=frame)
        if self.__epsilon is not None:
            obs['epsilon'] = np.array([self.__epsilon], dtype=np.float32)
        if self.__obs_include_last_action:
            obs['action'] = np.concatenate(list(self.__last_actions_queue), -1)
        if self.__obs_include_last_reward:
            obs['reward'] = np.array([reward], dtype=np.float32)
        return obs

    def reset(self) -> List[env_base.StepResult]:
        self.__step_count[:] = 0
        self.__episode_return[:] = 0
        self.__last_action = np.zeros((self.__env.action_space.n,), dtype=np.uint8)
        for _ in range(self.__stacked_last_actions):
            self.__last_actions_queue.append(self.__last_action)

        frame = self.__env.reset()

        return [
            env_base.StepResult(obs=self._get_obs(frame, None, 0),
                                reward=np.array([0.0], dtype=np.float32),
                                done=np.array([False], dtype=np.uint8),
                                info=dict(episode_length=self.__step_count.copy(),
                                          episode_return=self.__episode_return.copy()))
        ]

    def step(self, actions: List[env_utils.DiscreteAction]) -> List[env_base.StepResult]:

        assert len(actions) == 1, len(actions)
        action = int(actions[0].x)

        frame, reward, done, info = self.__env.step(action)

        self.__step_count += self.__frame_skip
        self.__episode_return += reward

        if self.__render:
            logger.info("Step %d: reward=%.2f, done=%d", self.__step_count, reward, done)
            if _HAS_DISPLAY:
                self.render()
                if self.__pause:
                    input()
                else:
                    time.sleep(0.05)

        return [
            env_base.StepResult(
                obs=self._get_obs(frame, action, reward),
                reward=np.array([reward], dtype=np.float32),
                done=np.array([done], dtype=np.uint8),
                info=dict(episode_length=self.__step_count.copy(),
                          episode_return=self.__episode_return.copy()),
            )
        ]

    def render(self) -> None:
        self.__env.render()

    def seed(self, seed=None):
        self.__env.seed(seed)
        return seed


if __name__ == '__main__':
    import psutil
    import multiprocessing
    import time

    # code or function for which memory
    # has to be monitored
    def app():
        config = dict(
            full_action_space=False,
            noop_max=30,
            frame_skip=4,
            stacked_observations=4,
            gray_scale=True,
            obs_shape=(84, 84),
            scale=False,
            obs_include_last_action=True,
            obs_include_last_reward=True,
            epsilon=0.1,
        )
        env = AtariEnvironment('PongNoFrameskip-v4', **config)
        step_cnt = 0
        import time
        srs = env.reset()
        tik = time.perf_counter()
        for _ in range(100):
            done = False
            srs = env.reset()
            while not done:
                srs = env.step([sp.sample() for sp, sr in zip(env.action_spaces, srs)])
                done = all(sr.done[0] for sr in srs if sr is not None)
                step_cnt += 1
                if step_cnt % 100 == 0:
                    print(f"FPS: {step_cnt * 4 / (time.perf_counter() - tik)}")

    p = multiprocessing.Process(target=app)
    p.start()

    # Get the process ID of the current Python process
    main_process = psutil.Process()

    for _ in range(100):
        time.sleep(5)
        # Get a list of child processes for the main process
        child_processes = main_process.children(recursive=True)

        # Find the child process with the highest memory usage
        mem = sum([process.memory_info().rss / 1024**2 for process in child_processes])
        print(f"Memory: {mem}")