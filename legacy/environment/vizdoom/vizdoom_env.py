from typing import List
import gym
import numpy as np

from .doom.action_space import Discretized
from .doom.doom_utils import DOOM_ENVS, make_doom_env
import api.environment as env_base
import api.env_utils


class VizDoomEnvironment(env_base.Environment):

    def __init__(self, scenario_name, seed=None, skip_frames=4, **kwargs):
        self.__env = make_doom_env(scenario_name, skip_frames=skip_frames, **kwargs)
        self.__env_spec = None
        self.__skip_frames = skip_frames
        for spec in DOOM_ENVS:
            if spec.name == scenario_name:
                self.__env_spec = spec
                break
        if isinstance(self.__env.action_space, gym.spaces.Tuple) and all(
                isinstance(space, gym.spaces.Discrete) or isinstance(space, Discretized)
                for space in self.__env.action_space.spaces):
            self.__action_space = gym.spaces.MultiDiscrete(
                [space.n for space in self.__env.action_space.spaces])
        elif isinstance(self.__env.action_space, gym.spaces.Discrete):
            self.__action_space = self.__env.action_space
        else:
            raise NotImplementedError("Compositional action space not supported yet.")
        self.__env.seed(seed)

    @property
    def action_spaces(self) -> List[env_base.ActionSpace]:
        return [api.env_utils.DiscreteActionSpace(self.__action_space) for _ in range(self.agent_count)]

    @property
    def agent_count(self) -> int:
        return self.__env_spec.num_agents

    def step(self, actions: List[env_base.Action]) -> List[env_base.StepResult]:
        env_action = [tuple(map(int, tuple(action.x))) for action in actions]
        for i, a in enumerate(env_action):
            if len(a) == 1:
                env_action[i] = a[0]
        if self.agent_count == 1:
            env_action = env_action[0]
        obs, reward, done, truncated, env_info = self.__env.step(env_action)
        self.__ep_len += 1
        self.__ep_ret += np.array(reward)
        if self.agent_count == 1:
            obs = [obs]
            reward = [reward]
            done = [done]
            truncated = [truncated]
            env_info = [env_info]
        for i, o in enumerate(obs):
            if not isinstance(o, dict):
                assert o.dtype == np.uint8
                obs[i] = {'obs': o}
        for o in obs:
            o['obs'] = np.transpose(o['obs'], (2, 0, 1)).astype(np.uint8)

        # for i, (t, d) in enumerate(zip(truncated, done)):
        #     if (self.__env_spec.default_timeout > 0
        #             and self.__ep_len * self.__skip_frames >= self.__env_spec.default_timeout):
        #         assert d and not t
        #         truncated[i] = True
        #         done[i] = False

        return [
            env_base.StepResult(
                obs=obs[i],
                reward=np.array([reward[i]], dtype=np.float32),
                done=np.array([done[i]], dtype=np.uint8),
                truncated=np.array([truncated[i]], dtype=np.uint8),
                info=dict(**({
                    f"episode_return_{i}": self.__ep_ret[i:i + 1].copy()
                    for i in range(self.agent_count)
                } if self.agent_count > 1 else dict(episode_return=self.__ep_ret.copy())),
                          raw_score=np.array([env_info[i]['true_objective']], dtype=np.float32)
                          if 'true_objective' in env_info[i] else np.zeros(1, dtype=np.float32),
                          episode_length=self.__ep_len.copy())) for i in range(self.agent_count)
        ]

    def reset(self) -> List[env_base.StepResult]:
        self.__ep_ret = np.zeros(self.agent_count)
        self.__ep_len = np.zeros(1)

        obs, _ = self.__env.reset()
        if self.agent_count == 1:
            obs = [obs]
        for i, o in enumerate(obs):
            if not isinstance(o, dict):
                assert o.dtype == np.uint8
                obs[i] = {'obs': o}
        for o in obs:
            o['obs'] = np.transpose(o['obs'], (2, 0, 1)).astype(np.uint8)
        return [
            env_base.StepResult(obs=obs[i],
                                reward=np.zeros(1, dtype=np.float32),
                                done=np.array([False], dtype=np.uint8),
                                truncated=np.array([False], dtype=np.uint8),
                                info=None) for i in range(self.agent_count)
        ]

    def render(self, *args, **kwargs):
        return self.__env.render(*args, **kwargs)

    def seed(self, seed=None):
        return self.__env.seed(seed)

    def close(self):
        return self.__env.close()


if __name__ == "__main__":
    # single-agent case
    scenario_name = 'doom_benchmark'
    # scenario_name = 'doom_battle2'
    # scenario_name = 'doom_duel'
    for spec in DOOM_ENVS:
        if spec.name == scenario_name:
            env_spec = spec
            break
    num_agents = env_spec.num_agents

    env = VizDoomEnvironment(scenario_name)
    import time
    tik = time.time()
    all_env_steps = 0
    for _ in range(200):
        srs = env.reset()
        step_cnt = 0
        done = False
        assert len(srs) == num_agents
        for sr in srs:
            assert isinstance(sr.obs, dict)
            assert sr.obs['obs'].shape == (3, 72, 128), sr.obs['obs'].shape
            assert sr.obs['obs'].dtype == np.uint8
            assert isinstance(sr.info, dict) or sr.info is None
            assert sr.reward.shape == (1,)
            assert sr.done.shape == (1,)
            assert sr.truncated.shape == (1,)
        for spec in DOOM_ENVS:
            if spec.name == scenario_name:
                env_spec = spec
                break
        while not done:
            srs = env.step([space.sample() for space in env.action_spaces])
            for sr in srs:
                assert isinstance(sr.obs, dict)
                assert sr.obs['obs'].shape == (3, 72, 128), sr.obs['obs'].shape
                assert sr.obs['obs'].dtype == np.uint8
                assert isinstance(sr.info, dict)
                assert sr.reward.shape == (1,)
                assert sr.done.shape == (1,)
                assert sr.truncated.shape == (1,)
            step_cnt += 1
            done = all(sr.done for sr in srs)
        print(step_cnt, srs[0].info)
        all_env_steps += step_cnt
    print(all_env_steps / (time.time() - tik))

    env.close()
    del env
