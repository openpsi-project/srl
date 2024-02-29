from typing import List
from gfootball.env import create_environment
import copy
import getpass
import numpy as np
import os
import shutil

import api.environment
import api.env_utils
import base.cluster

_HAS_DISPLAY = len(os.environ.get("DISPLAY", "").strip()) > 0


class FootballEnvironment(api.environment.Environment):
    """A wrapper of google football environment
    """

    def __copy_videos(self):
        for file in os.listdir(self.__tmp_log_dir):
            shutil.move(os.path.join(self.__tmp_log_dir, file), os.path.join(self.__log_dir, file))

    def __del__(self):
        self.__env.close()
        self.__copy_videos()

    def set_curriculum_stage(self, stage_name: str):
        self.__copy_videos()
        if stage_name is None:
            raise ValueError()
        if self.__env_name == stage_name:
            return
        self.__env_name = stage_name
        self.__log_dir = os.path.join(os.path.dirname(self.__log_dir), self.__env_name)
        os.makedirs(self.__log_dir, exist_ok=True)

        self.__tmp_log_dir = base.cluster.get_random_tmp()
        kwargs = dict(env_name=stage_name,
                      representation=self.__representation,
                      number_of_left_players_agent_controls=self.control_left,
                      number_of_right_players_agent_controls=self.control_right,
                      write_video=self.__write_video,
                      write_full_episode_dumps=self.__write_full_episode_dumps,
                      logdir=self.__tmp_log_dir,
                      dump_frequency=self.__dump_frequency,
                      render=False)
        self.__env.close()
        self.__env = create_environment(**kwargs)
        self.seed(None)

    def seed(self, seed):
        self.__env.seed(seed)

    def __init__(self, seed=None, share_reward=False, shared=False, **kwargs):
        self.__shared = shared
        if shared:
            share_reward = True
        self.__env_name = kwargs["env_name"]
        self.__representation = kwargs["representation"]
        self.control_left = kwargs.get("number_of_left_players_agent_controls", 1)
        self.control_right = kwargs.get("number_of_right_players_agent_controls", 0)
        self.__render = kwargs.get("render", False)
        self.__write_video = kwargs.get("write_video", False)
        self.__write_full_episode_dumps = kwargs.get("write_full_episode_dumps", False)
        self.__dump_frequency = kwargs.get("dump_frequency", 1)
        self.__log_dir = os.path.join(kwargs.get("logdir", f"/home/{getpass.getuser()}/fb_replay"),
                                      self.__env_name)
        os.makedirs(self.__log_dir, exist_ok=True)
        if "logdir" in kwargs:
            kwargs.pop("logdir")
        self.__tmp_log_dir = base.cluster.get_random_tmp()
        self.__env = create_environment(**kwargs, logdir=self.__tmp_log_dir)
        self.seed(seed)
        self.__share_reward = share_reward
        self.__env_agents = self.control_left + self.control_right
        self.__space = api.env_utils.DiscreteActionSpace(self.__env.action_space[0],
                                                         shared=self.__shared,
                                                         n_agents=self.__env_agents)
        self.__step_count = np.zeros(1, dtype=np.int32)
        self.__episode_return = np.zeros((self.__env_agents, 1), dtype=np.float32)

    @property
    def agent_count(self) -> int:
        return self.__env_agents if not self.__share_reward else 1

    @property
    def action_spaces(self) -> List[api.environment.ActionSpace]:
        return [self.__space for _ in range(self.agent_count)]

    def reset(self):
        obs = self.__env.reset()
        self.__step_count[:] = self.__episode_return[:] = 0
        obs, _ = self.__post_process_obs_and_rew(obs, np.zeros(self.__env_agents))
        if not self.__shared:
            return [
                api.environment.StepResult(obs=dict(obs=obs[i]),
                                           reward=np.array([0.0], dtype=np.float64),
                                           done=np.array([False], dtype=np.uint8),
                                           info=dict()) for i in range(self.agent_count)
            ]
        else:
            obs = np.array(obs, dtype=np.float32)
            state = np.concatenate([obs[..., :-18], obs[..., -7:]], -1)
            return [
                api.environment.StepResult(obs=dict(obs=obs, state=state),
                                           reward=np.zeros((self.__env_agents, 1), dtype=np.float32),
                                           done=np.zeros((self.__env_agents, 1), dtype=np.uint8),
                                           info=dict())
            ]

    def __post_process_obs_and_rew(self, obs, reward):
        if self.__env_agents == 1:
            obs = obs[np.newaxis, :]
            reward = [reward]
        if self.__representation == "extracted":
            obs = np.swapaxes(obs, 1, 3)
        if self.__representation in ("simple115", "simple115v2"):
            obs[obs == -1] = 0
        if self.__share_reward:
            left_reward = np.mean(reward[:self.control_left])
            right_reward = np.mean(reward[self.control_left:])
            reward = np.array([left_reward] * self.control_left + [right_reward] * self.control_right)
        return obs, reward

    def step(self, actions: List[api.env_utils.DiscreteAction]) -> List[api.environment.StepResult]:
        assert len(actions) == self.agent_count, len(actions)
        if not self.__shared:
            actions = [a.x.item() for a in actions]
        else:
            actions = [int(actions[0].x[i]) for i in range(len(actions[0].x))]
        obs, reward, done, info = self.__env.step(actions)
        obs, reward = self.__post_process_obs_and_rew(obs, reward)
        self.__step_count += 1
        self.__episode_return += reward[:, np.newaxis]
        if not self.__shared:
            return [
                api.environment.StepResult(obs=dict(obs=obs[i]),
                                           reward=np.array([reward[i]], dtype=np.float64),
                                           done=np.array([done], dtype=np.uint8),
                                           info=dict(episode_length=copy.deepcopy(self.__step_count),
                                                     episode_return=copy.deepcopy(self.__episode_return[i])))
                for i in range(self.agent_count)
            ]
        else:
            obs = np.array(obs, dtype=np.float32)
            state = np.concatenate([obs[..., :-18], obs[..., -7:]], -1)
            reward = np.array([[float(reward[i])] for i in range(len(reward))], dtype=np.float32)
            done = np.array([[done] for _ in range(len(reward))], dtype=np.uint8)
            info = dict(episode_length=copy.deepcopy(self.__step_count),
                        episode_return=copy.deepcopy(self.__episode_return[0]))
            return [
                api.environment.StepResult(obs=dict(obs=obs, state=state),
                                           reward=reward,
                                           done=done,
                                           info=info)
            ]

    def render(self) -> None:
        self.__env.render()


if __name__ == "__main__":
    import psutil
    import multiprocessing
    import time

    # code or function for which memory
    # has to be monitored
    def app():
        env = FootballEnvironment(
            env_name="11_vs_11_stochastic",
            representation="simple115v2",
            number_of_left_players_agent_controls=11,
            number_of_right_players_agent_controls=0,
            shared=True,
        )
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
                    print(f"FPS: {step_cnt / (time.perf_counter() - tik)}")

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
