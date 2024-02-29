"""
Wrapper around the Overcooked environments
Note: It is suggested to set ring_size to 1 if rendering.
"""
from typing import List
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_trajectory import DEFAULT_TRAJ_KEYS, TIMESTEP_TRAJ_KEYS
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import gym
import imageio
import numpy as np
import pickle
import os
import time

from api import env_utils
from api import environment
from api.environment import ActionSpace

# adapted from https://github.com/Stanford-ILIAD/PantheonRL/blob/35b31201e75c85dccedf22a2f0ebd9d0d83afe83/overcookedgym/overcooked.py

LAYOUT_LIST = [
    'asymmetric_advantages', 'asymmetric_advantages_tomato', 'bonus_order_test', 'bottleneck',
    'centre_objects', 'centre_pots', 'coordination_ring', 'corridor', 'counter_circuit',
    'counter_circuit_o_1order', 'cramped_corridor', 'cramped_room', 'cramped_room_o_3orders',
    'cramped_room_single', 'cramped_room_tomato', 'five_by_five', 'forced_coordination',
    'forced_coordination_tomato', 'inverse_marshmallow_experiment', 'large_room', 'long_cook_time',
    'm_shaped_s', 'marshmallow_experiment', 'marshmallow_experiment_coordination', 'mdp_test',
    'multiplayer_schelling', 'pipeline', 'scenario1_s', 'scenario2', 'scenario2_s', 'scenario3', 'scenario4',
    'schelling', 'schelling_s', 'simple_o', 'simple_o_t', 'simple_tomato', 'small_corridor',
    'soup_coordination', 'tutorial_0', 'tutorial_1', 'tutorial_2', 'tutorial_3', 'unident',
    'you_shall_not_pass'
]

DEFAULT_ENV_PARAMS = {"horizon": 400}

OvercookedAction = env_utils.DiscreteAction
OvercookedActionSpace = env_utils.DiscreteActionSpace


class OvercookedEnvironment(environment.Environment):

    def __init__(self,
                 layout_name=LAYOUT_LIST[1],
                 sparse_reward=False,
                 render=False,
                 render_interval=1000,
                 env_idx=-1,
                 **kwargs):
        assert layout_name in LAYOUT_LIST
        self.layout_name = layout_name
        self.mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name)
        self.__env = OvercookedEnv.from_mdp(self.mdp, **DEFAULT_ENV_PARAMS)
        self.mdp = self.__env.mdp
        self.__space = OvercookedActionSpace(gym.spaces.Discrete(len(Action.ALL_ACTIONS)))
        self.__step_count = np.zeros(1, dtype=np.int32)
        self.__episode_return = np.zeros(1, dtype=np.float32)
        self.__render = render
        self.__render_interval = render_interval
        self.__env_idx = env_idx
        self.init_traj()
        self.__traj_num = 0
        self.featurize_fn = None
        self.__sparse_reward = sparse_reward

    def seed(self, seed=None):
        self.__env.seed(seed)
        return seed

    @property
    def agent_count(self) -> int:
        return self.__env.mdp.num_players

    @property
    def observation_spaces(self) -> List[dict]:
        dummy_mdp = self.__env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_mdp, dummy_state)[0].shape
        high = np.ones(obs_shape) * max(dummy_mdp.soup_cooking_time, dummy_mdp.num_items_for_soup, 5)
        # TODO(th): implement obs space
        return {'obs': gym.spaces.Box(high * 0, high, dtype=np.float32)}
        # return {}

    @property
    def action_spaces(self) -> List[ActionSpace]:
        return [self.__space for _ in range(self.agent_count)]

    def init_traj(self):
        self.__traj = {k: [] for k in DEFAULT_TRAJ_KEYS}
        for key in TIMESTEP_TRAJ_KEYS:
            self.__traj[key].append([])

    @property
    def __should_render(self):
        return self.__render and (self.__traj_num % self.__render_interval == 0) and (self.__env_idx == 0)

    def render(self):
        try:
            save_dir = f'~/overcooked_replay/{self.layout_name}/traj_num_{self.__traj_num}'
            save_dir = os.path.expanduser(save_dir)
            StateVisualizer().display_rendered_trajectory(self.__traj,
                                                          img_directory_path=save_dir,
                                                          ipython_display=False)
            for img_path in os.listdir(save_dir):
                img_path = save_dir + '/' + img_path
                if 'gif' in img_path:
                    os.remove(img_path)
            imgs = []
            imgs_dir = os.listdir(save_dir)
            imgs_dir = sorted(imgs_dir, key=lambda x: int(x.split('.')[0]))
            for img_path in imgs_dir:
                img_path = save_dir + '/' + img_path
                imgs.append(imageio.imread(img_path))
            imageio.mimsave(save_dir + f'/reward_{self.__traj["ep_returns"][0][0]}.gif', imgs, duration=0.05)
            imgs_dir = os.listdir(save_dir)
            for img_path in imgs_dir:
                img_path = save_dir + '/' + img_path
                if 'png' in img_path:
                    os.remove(img_path)
        except Exception as e:
            print('failed to render traj: ', e)

    def reset(self) -> List[environment.StepResult]:
        if self.__step_count > 0 and self.__should_render:
            self.__traj['ep_returns'].append(self.__episode_return)
            self.__traj['ep_lengths'].append(self.__step_count)
            self.__traj["mdp_params"].append(self.mdp.mdp_params)
            self.__traj["env_params"].append(self.__env.env_params)
            self.render()
            self.init_traj()
            self.__traj_num += 1

        else:
            self.init_traj()
            self.__traj_num += 1

        # TODO(th): is regen_mdp needed?
        self.__env.reset(regen_mdp=False)

        if self.featurize_fn is None:
            for _ in range(99):
                try:
                    mlp = MediumLevelActionManager.from_pickle_or_compute(self.mdp,
                                                                          NO_COUNTERS_PARAMS,
                                                                          force_compute=False)
                    self.featurize_fn = lambda x: self.mdp.featurize_state(x, mlp)
                    break
                except pickle.UnpicklingError as e:
                    time.sleep(1)
            else:
                raise e

        ordered_features = self.featurize_fn(self.__env.state)

        self.__step_count[:] = self.__episode_return[:] = 0
        return [
            environment.StepResult(obs=dict(obs=ordered_features[i]),
                                   reward=np.array([0.0], dtype=np.float64),
                                   done=np.array([False], dtype=np.int8),
                                   info=dict()) for i in range(self.agent_count)
        ]

    def step(self, actions: List[Action]) -> List[environment.StepResult]:
        assert len(actions) == self.agent_count, len(actions)
        state = self.__env.state
        actions = [Action.INDEX_TO_ACTION[a.x.item()] for a in actions]
        next_state, reward, done, info = self.__env.step(actions)

        if self.__should_render:
            self.__traj['ep_states'][0].append(state)
            self.__traj['ep_actions'][0].append(actions)
            self.__traj["ep_rewards"][0].append(reward)
            self.__traj["ep_dones"][0].append(done)
            self.__traj["ep_infos"][0].append(info)

        obs = self.featurize_fn(next_state)
        # if self.agent_count == 1:
        #     obs = obs[np.newaxis, :]
        #     reward = [reward]
        self.__step_count += 1
        self.__episode_return += reward

        # reward reshape
        if self.__sparse_reward:
            reward = info['sparse_r_by_agent']
        else:
            reward = np.array(info['shaped_r_by_agent']) + 10 * np.array(info['sparse_r_by_agent'])

        return [
            environment.StepResult(obs=dict(obs=obs[i]),
                                   reward=np.array([reward[i]], dtype=np.float64),
                                   done=np.array([done], dtype=np.uint8),
                                   info=dict(episode_length=self.__step_count,
                                             episode_return=self.__episode_return))
            for i in range(self.agent_count)
        ]
