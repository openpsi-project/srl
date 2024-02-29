"""Wrapper around the SMAC environments provided by https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/envs/starcraft2/StarCraft2_Env.py.
"""
from typing import List
import copy
import gym
import logging
import numpy as np

from legacy.environment.smac.smac_env_ import StarCraft2Env
import api.environment as env_base
import api.env_utils as env_utils
import base.numpy_utils

logger = logging.getLogger("env-smac")


def get_smac_shapes(map_name, use_state_agent=True, agent_specific_obs=False, agent_specific_state=False):
    env = StarCraft2Env(map_name, use_state_agent=use_state_agent)
    return env.get_obs_size(agent_specific_obs), env.get_state_size(
        agent_specific_state), env.get_total_actions(), env.n_agents


class SMACAction(env_utils.DiscreteAction):
    pass


class SMACActionSpace(env_utils.DiscreteActionSpace):
    pass


#
# Below are deprecated definitions of SMAC observation/episode_info. Kept here as reminder.
#
# @namedarray
# class SMACAgentSpecificObs(env_base.Observation):
#     obs_allies: np.ndarray
#     obs_enemies: np.ndarray
#     obs_move: np.ndarray
#     obs_self: np.ndarray
#     obs_mask: np.ndarray
#
#
# @namedarray
# class SMACAgentSpecificState(env_base.Observation):
#     state_allies: np.ndarray
#     state_enemies: np.ndarray
#     state_self: np.ndarray
#     state_mask: np.ndarray
#     state_move: Optional[np.ndarray] = None
#
#
# @namedarray
# class SMACObservation(env_base.Observation):
#     local_obs: Union[np.ndarray, SMACAgentSpecificObs]
#     state: Union[np.ndarray, SMACAgentSpecificState]
#     available_action: np.ndarray
#
#
# @namedarray
# class SMACEpisodeInfo(env_base.EpisodeInfo):
#     episode_length: np.ndarray = np.zeros(1, dtype=np.float32)
#     episode_return: np.ndarray = np.zeros(1, dtype=np.float32)
#     win: np.ndarray = np.zeros(1, dtype=np.float32)

SMAC_EPISODE_INFO_FIELDS = ["episode_length", "episode_return", "win"]


class SMACEnvironment(env_base.Environment):

    def __init__(self,
                 map_name,
                 save_replay=False,
                 agent_specific_obs=False,
                 agent_specific_state=False,
                 shared=False,
                 use_truncate=True,
                 discount_rate=None,
                 **kwargs):
        self.map_name = map_name
        self.__shared = shared
        self.__save_replay = save_replay
        self.__agent_specific_obs = agent_specific_obs
        self.__agent_specific_state = agent_specific_state
        self.__use_truncate = use_truncate

        if not shared and discount_rate is None:
            raise ValueError("Discount rate must be given if not shared!")
        self.__discount_rate = discount_rate

        for i in range(10):
            try:
                self.__env = StarCraft2Env(map_name=map_name, **kwargs)
                self.__act_space = SMACActionSpace(gym.spaces.Discrete(self.__env.n_actions), self.__shared,
                                                   self.__env.n_agents)
                _, _, available_action, _, _ = self.__env.reset()
                if self.__shared:
                    action = self.__act_space.sample(available_action).x
                else:
                    actions = [
                        self.__act_space.sample(available_action[i]) for i in range(len(available_action))
                    ]
                    action = np.stack([
                        action.x if action is not None else np.zeros(1, dtype=np.float32)
                        for action in actions
                    ],
                                      axis=0)
                self.__env.step(action)
                break
            except Exception as e:
                print(f"Failed to start SC2 Environment due to {e}, retrying {i}")
        else:
            raise RuntimeError("Failed to start SC2.")

        self.__obs_shapes = self.__env.get_obs_size(agent_specific=self.__agent_specific_obs)
        self.__state_shapes = self.__env.get_state_size(agent_specific=self.__agent_specific_state)

        self.__rewards_to_go = np.zeros((self.__env.n_agents, 1), dtype=np.float32)

        if self.__agent_specific_obs:
            self.__obs_split_shapes = copy.deepcopy(self.__obs_shapes)
            self.__obs_split_shapes.pop('obs_mask')
        if self.__agent_specific_state:
            self.__state_split_shapes = copy.deepcopy(self.__state_shapes)
            self.__state_split_shapes.pop('state_mask')

        self.__obs_space = dict(local_obs=self.__obs_shapes,
                                state=self.__state_shapes,
                                available_action=self.__act_space.n)

    @property
    def agent_count(self) -> int:
        # consider smac as a single-agent environment if the action/observation spaces are shared
        return self.__env.n_agents if not self.__shared else 1

    @property
    def observation_spaces(self) -> List[dict]:
        return [self.__obs_space for _ in range(self.agent_count)]

    @property
    def action_spaces(self):
        return [self.__act_space for _ in range(self.agent_count)]

    def reset(self) -> List[env_base.StepResult]:
        self.__rewards_to_go[:] = 0

        local_obs, state, available_action, obs_mask, state_mask = self.__env.reset()

        is_alive = np.expand_dims(1 - self.__env.death_tracker_ally, -1)

        if self.__agent_specific_obs:
            local_obs = dict(**base.numpy_utils.split_to_shapes(local_obs, self.__obs_split_shapes, -1),
                             obs_mask=obs_mask)
        if self.__agent_specific_state:
            state = dict(**base.numpy_utils.split_to_shapes(state, self.__state_split_shapes, -1),
                         state_mask=state_mask)

        if self.__shared:
            return [
                env_base.StepResult(obs=dict(local_obs=local_obs,
                                             state=state,
                                             available_action=available_action,
                                             is_alive=is_alive),
                                    reward=np.zeros((len(available_action), 1), dtype=np.float32),
                                    done=np.zeros((len(available_action), 1), dtype=np.uint8),
                                    info=dict())
            ]
        else:
            agent_obs = [
                dict(local_obs=local_obs[i], state=state[i], available_action=available_action[i])
                for i in range(self.agent_count)
            ]

            return [
                env_base.StepResult(obs=agent_obs[i],
                                    reward=np.array([0.0], dtype=np.float32),
                                    done=np.array([False], dtype=np.uint8),
                                    info=dict()) for i in range(self.agent_count)
            ]

    def step(self, actions: List[SMACAction]) -> List[env_base.StepResult]:
        assert len(actions) == self.agent_count, len(actions)
        if self.__shared:
            assert len(actions) == 1
            actions = actions[0].x
        else:
            actions = np.stack(
                [action.x if action is not None else np.zeros(1, dtype=np.float32) for action in actions],
                axis=0)

        local_obs, state, rewards, dones, infos, available_action, obs_mask, state_mask = self.__env.step(
            actions)

        is_alive = np.expand_dims(1 - self.__env.death_tracker_ally, -1)

        episode_finish = dones.all()
        if not self.__shared:
            if dones.any():
                # Accumulate rewards for dead agents and return `None` until episode finishes.
                self.__rewards_to_go *= self.__discount_rate
                self.__rewards_to_go += rewards * dones
            if dones.all():
                # Return a `done` step with accumulated reward when episode finishes.
                rewards = self.__rewards_to_go.copy()

        truncateds = np.zeros_like(dones)
        if self.__use_truncate and infos[0].get('episode_limit', False):
            truncateds[:] = 1
            dones[:] = 0

        if self.__agent_specific_obs:
            local_obs = dict(**base.numpy_utils.split_to_shapes(local_obs, self.__obs_split_shapes, -1),
                             obs_mask=obs_mask)
        if self.__agent_specific_state:
            state = dict(**base.numpy_utils.split_to_shapes(state, self.__state_split_shapes, -1),
                         state_mask=state_mask)

        if self.__save_replay and np.all(dones):
            self.__env.save_replay()

        if self.__shared:
            assert rewards.shape == (len(available_action), 1) and dones.shape == (len(available_action),
                                                                                   1), (rewards.shape,
                                                                                        dones.shape)
            info = {
                k: np.array([float(infos[0][k])], dtype=np.float32)
                if k in infos[0] else np.zeros(1, dtype=np.float32)
                for k in SMAC_EPISODE_INFO_FIELDS
            }
            dones[:] = dones.all()
            truncateds[:] = truncateds.all()
            obs = dict(local_obs=local_obs, state=state, available_action=available_action, is_alive=is_alive)
            return [env_base.StepResult(obs, rewards, dones, info, truncateds)]
        else:
            episode_infos = [{
                k: np.array([float(info[k])], dtype=np.float32) if k in info else np.zeros(1,
                                                                                           dtype=np.float32)
                for k in SMAC_EPISODE_INFO_FIELDS
            } for info in infos]
            for info in episode_infos:
                info['max_reward_to_go'] = np.array([float(max(self.__rewards_to_go))], dtype=np.float32)
            agent_obs = [
                dict(local_obs=local_obs[i], state=state[i], available_action=available_action[i])
                for i in range(self.agent_count)
            ]
            return [
                env_base.StepResult(*x) if episode_finish or not dones[i] else None
                for i, x in enumerate(zip(agent_obs, rewards, dones, episode_infos, truncateds))
            ]

    def render(self) -> None:
        raise NotImplementedError(
            'Rendering the SMAC environment is by default disabled. Please save replay instead.')

    def seed(self, seed):
        self.__env.seed(seed)
        return seed

    def close(self):
        self.__env.close()


if __name__ == "__main__":
    import psutil
    import multiprocessing
    import time

    # code or function for which memory
    # has to be monitored
    def app():
        env = SMACEnvironment(map_name='27m_vs_30m', shared=False, discount_rate=0.99)
        step_cnt = 0
        import time
        srs = env.reset()
        tik = time.perf_counter()
        for _ in range(100):
            done = False
            srs = env.reset()
            while not done:
                srs = env.step([
                    sp.sample(sr.obs['available_action']) if sr is not None else None
                    for sp, sr in zip(env.action_spaces, srs)
                ])
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
