from typing import Dict, List
import copy
import gym
import numpy as np

from api import environment
import api.env_utils

import mujoco_py
assert mujoco_py.__version__ == '1.50.1.68', (
    "To reproduce results in the original paper, "
    "mujoco_py version 1.50.1.68 is required. "
    "Please follow instructions in "
    "https://github.com/openai/mujoco-py/tree/1.50.1.0 to finish installation.")

EPISODE_INFO_FIELDS = [
    "max_box_move_prep", "max_box_move", "num_box_lock_prep", "num_box_lock", "max_ramp_move_prep",
    "max_ramp_move", "num_ramp_lock_prep", "num_ramp_lock", "episode_return", "hider_return", "seeker_return"
]


class HideAndSeekEnvironment(environment.Environment):

    def __init__(self, scenario_name, seed=None, max_n_agents=5, **kwargs):
        self.__env_config = copy.deepcopy(dict(max_n_agents=max_n_agents, **kwargs))
        self.__scenario = scenario_name
        self.__max_n_agents = max_n_agents + 1  # max others + self

        if scenario_name == "box_locking":
            from legacy.environment.hide_and_seek.scenarios.box_locking import make_env
            self.__num_agents = self.__env_config['n_agents']
            self.__env = make_env(**self.__env_config)
            self.ordered_obs_keys = ['agent_qpos_qvel', 'box_obs', 'ramp_obs', 'observation_self']
            self.__obs_space = ['mask_aa_obs', 'mask_ab_obs', 'mask_ar_obs', None]
        elif scenario_name == "blueprint_construction":
            from legacy.environment.hide_and_seek.scenarios.blueprint_construction import make_env
            self.__num_agents = self.__env_config['n_agents']
            self.__env = make_env(**self.__env_config)
            self.ordered_obs_keys = [
                'agent_qpos_qvel', 'box_obs', 'ramp_obs', 'construction_site_obs', 'observation_self'
            ]
            self.__obs_space = [None, None, None, None, None]
        elif scenario_name == "hide_and_seek":
            from legacy.environment.hide_and_seek.scenarios.hide_and_seek import make_env
            self.__num_seekers = self.__env_config['n_seekers']
            self.__num_hiders = self.__env_config['n_hiders']
            self.__env = make_env(**self.__env_config)
            self.__num_agents = self.__env_config['n_seekers'] + self.__env_config['n_hiders']
            self.ordered_obs_keys = [
                'agent_qpos_qvel', 'box_obs', 'ramp_obs', 'foodict_obsbs', 'observation_self', 'lidar'
            ]
            self.__obs_space = ['mask_aa_obs', 'mask_ab_obs', 'mask_ar_obs', 'mask_af_obs', None, None]
        else:
            raise NotImplementedError

        self.__hider_return = self.__seeker_return = 0
        self.__episode_return = 0

        self.__act_space = gym.spaces.MultiDiscrete([11, 11, 11, 2, 2])
        self.seed(seed)

    @property
    def agent_count(self) -> int:
        return self.__num_agents

    @property
    def action_spaces(self) -> List[dict]:
        return [api.env_utils.DiscreteActionSpace(self.__act_space) for _ in range(self.agent_count)]

    def reset(self):
        self.__hider_return = self.__seeker_return = 0
        self.__episode_return = 0
        dict_obs = self.__env.reset()
        if 'lidar' in dict_obs.keys():
            dict_obs['lidar'] = np.transpose(dict_obs['lidar'], (0, 2, 1))
        dict_obs = {
            **dict_obs, 'mask_ar_obs_spoof': np.ones(
                (self.__num_hiders + self.__num_seekers, self.__env_config['n_ramps']), dtype=np.float32)
        }
        # dict_obs = self._pad_agent(dict_obs)
        return [
            environment.StepResult(obs={
                k: v[i]
                for k, v in dict_obs.items()
            },
                                   reward=np.array([0.0], dtype=np.float32),
                                   done=np.array([False], dtype=np.uint8),
                                   info=None,
                                   truncated=np.array([False], dtype=np.uint8))
            for i in range(self.agent_count)
        ]

    def step(self, actions: List[api.env_utils.DiscreteAction]) -> List[environment.StepResult]:
        if self.__scenario == 'hide_and_seek':
            actions = actions[:self.__num_hiders + self.__num_seekers]
        action_movement = np.array(
            [[int(action.x[0].item()),
              int(action.x[1].item()),
              int(action.x[2].item())] for action in actions])
        action_pull = np.array([int(action.x[3].item()) for action in actions])
        action_glueall = np.array([int(action.x[4].item()) for action in actions])
        actions_env = {
            'action_movement': action_movement,
            'action_pull': action_pull,
            'action_glueall': action_glueall
        }

        dict_obs, rewards, done, info = self.__env.step(actions_env)
        # rewards = np.append(
        #     rewards,
        #     np.zeros((self.__max_n_agents - (self.__num_hiders + self.__num_seekers)), dtype=np.float32))
        rewards = np.expand_dims(np.array(rewards), -1)
        truncateds = np.array([[done] for _ in range(self.agent_count)], dtype=np.uint8)
        dones = np.zeros_like(truncateds, dtype=np.uint8)
        if 'lidar' in dict_obs.keys():
            dict_obs['lidar'] = np.transpose(dict_obs['lidar'], (0, 2, 1))

        episode_infos = [{
            k: np.array([float(info[k])], dtype=np.float32) if k in info else np.zeros(1, dtype=np.float32)
            for k in EPISODE_INFO_FIELDS
        } for _ in range(self.agent_count)]
        if self.__scenario == 'hide_and_seek':
            self.__hider_return += rewards[0]
            self.__seeker_return += rewards[self.__num_hiders + self.__num_seekers - 1]
            for info in episode_infos:
                info['hider_return'][:] = self.__hider_return
                info['seeker_return'][:] = self.__seeker_return
        else:
            self.__episode_return += rewards[0]
            for info in episode_infos:
                info['episode_return'][:] = self.__episode_return
        dict_obs = {
            **dict_obs, 'mask_ar_obs_spoof': np.ones(
                (self.__num_hiders + self.__num_seekers, self.__env_config['n_ramps']), dtype=np.float32)
        }
        # dict_obs = self._pad_agent(dict_obs)

        agent_obs = [{k: v[i] for k, v in dict_obs.items()} for i in range(self.agent_count)]
        # TODO: variable number of agents
        # TODO: truncated episode
        return [environment.StepResult(*x) for x in zip(agent_obs, rewards, dones, episode_infos, truncateds)]

    def render(self, *args, **kwargs):
        return self.__env.render(*args, **kwargs)

    def seed(self, seed=None):
        if seed is None:
            self.__env.seed(np.random.randint(4294967295))
        else:
            self.__env.seed(seed)
        return seed

    def close(self):
        self.__env.close()


if __name__ == "__main__":
    _HNS_RANDOMWALL_CONFIG = {
        'max_n_agents': 5,
        # grab
        'grab_box': True,
        'grab_out_of_vision': False,
        'grab_selective': False,
        'grab_exclusive': False,
        # lock
        'lock_box': True,
        'lock_ramp': True,
        'lock_type': 'all_lock_team_specific',
        'lock_out_of_vision': False,
        # horizon
        'n_substeps': 15,
        'horizon': 240,
        'prep_fraction': 0.4,
        # map
        'scenario': 'randomwalls',
        'n_rooms': 4,
        'rew_type': 'joint_zero_sum',
        'random_room_number': True,
        'prob_outside_walls': 0.5,
        'restrict_rect': [-6.0, -6.0, 12.0, 12.0],
        'hiders_together_radius': 0.5,
        'seekers_together_radius': 0.5,
        # box
        'n_boxes': [3, 9],
        'n_elongated_boxes': [3, 9],
        'box_only_z_rot': True,
        'boxid_obs': False,
        # ramp
        'n_ramps': 2,
        # food
        'n_food': 0,
        'max_food_health': 40,
        'food_radius': 0.5,
        'food_box_centered': True,
        'food_together_radius': 0.25,
        'food_respawn_time': 5,
        'food_rew_type': 'joint_mean',
        # observations
        'n_lidar_per_agent': 30,
        'visualize_lidar': False,
        'prep_obs': True
    }
    nh = np.random.randint(1, 4)
    ns = np.random.randint(1, 4)
    env = HideAndSeekEnvironment("hide_and_seek", n_hiders=nh, n_seekers=ns, **_HNS_RANDOMWALL_CONFIG)
    done = [False]
    _ = env.reset()
    step = 0
    while not all(done):
        rs = env.step([sp.sample() for sp in env.action_spaces])
        assert len(rs) == env.agent_count
        done = [bool(r.truncated) for r in rs]
        step += 1
        print(rs[0].obs)
    # print({k: v.shape for k, v in rs[0].obs.items()})
    print(step, nh, ns, env.agent_count)
