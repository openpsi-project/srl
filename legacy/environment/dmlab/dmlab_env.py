from typing import List, Optional, Tuple
import logging
import numpy as np

from .lab.dmlab_env_ import make_dmlab_env
import api.environment as env_base
import api.env_utils
import time

logger = logging.getLogger("DMLab")

ALL_LEVEL_NAMES = [
    'rooms_collect_good_objects_train', 'rooms_exploit_deferred_effects_train',
    'rooms_select_nonmatching_object', 'rooms_watermaze', 'rooms_keys_doors_puzzle',
    'language_select_described_object', 'language_select_located_object', 'language_execute_random_task',
    'language_answer_quantitative_question', 'lasertag_one_opponent_small', 'lasertag_three_opponents_small',
    'lasertag_one_opponent_large', 'lasertag_three_opponents_large', 'natlab_fixed_large_map',
    'natlab_varying_map_regrowth', 'natlab_varying_map_randomized', 'skymaze_irreversible_path_hard',
    'skymaze_irreversible_path_varied', 'psychlab_arbitrary_visuomotor_mapping',
    'psychlab_continuous_recognition', 'psychlab_sequential_comparison', 'psychlab_visual_search',
    'explore_object_locations_small', 'explore_object_locations_large', 'explore_obstructed_goals_small',
    'explore_obstructed_goals_large', 'explore_goal_locations_small', 'explore_goal_locations_large',
    'explore_object_rewards_few', 'explore_object_rewards_many'
]


class DMLabEnvironment(env_base.Environment):

    def __init__(self, spec_name: str, rank: int, seed: Optional[int] = None, **kwargs):
        """Initialize the DeepMind Lab environment wrapped by sample factory.

        The environment has been seeded internally with the rank.
        Check lab/dmlab_env_.py for available kwargs.

        Args:
            spec_name (str): The spec name to run. Check lab/dmlab30.py for the list of available specs.
            rank (int): The rank of the environment. Specified by actor worker index and ring size.
                Used to set the seed and the level or task id in multi-task training.
                NOTE: Rank should be the same in a env ring.
        """
        self.__env = make_dmlab_env(spec_name, rank=rank, **kwargs)
        self.__level_name = self.__env.level.split('/')[-1]
        self.seed(seed)

        self.__raw_score_key = f"z_{ALL_LEVEL_NAMES.index(self.level_name):02d}_{self.level_name}_dmlab_raw_score"
        self.__ep_len_key = f"z_{ALL_LEVEL_NAMES.index(self.level_name):02d}_{self.level_name}_len"

        self.__first_reset = True
        self.__rank = rank

    @property
    def level_name(self):
        return self.__level_name

    @property
    def action_spaces(self) -> List[env_base.ActionSpace]:
        return [api.env_utils.DiscreteActionSpace(self.__env.action_space)]

    @property
    def agent_count(self) -> int:
        return 1

    def step(self, actions: List[env_base.Action]) -> List[env_base.StepResult]:
        action = int(actions[0].x)
        obs, reward, done, truncated, info_ = self.__env.step(action)
        if info_.get('episode_extra_stats', None):
            info = {
                k: np.array([float(v)], dtype=np.float32)
                for k, v in info_.get('episode_extra_stats').items()
            }
        else:
            info = {
                self.__raw_score_key: np.zeros(1, dtype=np.float32),
                self.__ep_len_key: np.zeros(1, dtype=np.float32)
            }
        return [
            env_base.StepResult(obs=obs,
                                reward=np.array([float(reward)], dtype=np.float32),
                                done=np.array([bool(done)], dtype=np.uint8),
                                truncated=np.array([bool(truncated)], dtype=np.uint8),
                                info={})
        ]

    def reset(self) -> List[env_base.StepResult]:
        # NOTE: we don't need to log episode return and length here, because the environment itself contains
        # if self.__first_reset:
        #     print(f"first reset, sleeping for {0.01*self.__rank}", flush=True)
        #     time.sleep(0.01*self.__rank)
        #     print(f"rank {self.__rank} sleep over")
        #     self.__first_reset = False

        obs, _ = self.__env.reset()
        return [
            env_base.StepResult(obs=obs,
                                reward=np.array([0], dtype=np.float32),
                                done=np.array([0], dtype=np.uint8),
                                truncated=np.array([0], dtype=np.uint8),
                                info={})
        ]

    def seed(self, seed=None):
        return self.__env.seed(seed)

    def close(self):
        return self.__env.close()

    def render(self, *args, **kwargs):
        return self.__env.render(*args, **kwargs)


if __name__ == '__main__':
    level_names = []
    envs = [DMLabEnvironment('dmlab_watermaze', rank=i) for i in range(10)]
    level_names.append(envs[0].level_name)

    st = time.monotonic()
    for _ in range(10):
        for env in envs:
            env.reset()
    print("sequential reset 10:", time.monotonic() - st)

    def reset_10_times(env):
        for _ in range(10):
            env.reset()

    import threading
    ts = [threading.Thread(target=reset_10_times, args=(env,)) for env in envs]

    for t in ts:
        t.start()
    st = time.monotonic()

    for t in ts:
        t.join()
    print("parallel reset 10:", time.monotonic() - st)

    st = time.monotonic()
    for _ in range(100):
        for env in envs:
            env.step([space.sample() for space in env.action_spaces])
    print("sequential step 100:", time.monotonic() - st)

    def step_100_times(env):
        for _ in range(100):
            env.step([space.sample() for space in env.action_spaces])

    import threading
    ts = [threading.Thread(target=step_100_times, args=(env,)) for env in envs]

    for t in ts:
        t.start()
    st = time.monotonic()

    for t in ts:
        t.join()
    print("parallel step 100:", time.monotonic() - st)

    # for ep_i in range(1000):
    #     srs = env.reset()

    #     step_cnt = 0
    #     done = False
    #     assert len(srs) == 1
    #     for sr in srs:
    #         assert isinstance(sr.obs, dict)
    #         assert sr.obs['obs'].shape == (3, 72, 96), sr.obs['obs'].shape
    #         assert sr.obs['INSTR'].shape == (16,), sr.obs['INSTR'].shape
    #         assert isinstance(sr.info, dict) or sr.info is None
    #         assert sr.reward.shape == (1,)
    #         assert sr.done.shape == (1,)
    #         assert sr.truncated.shape == (1,)

    #     print(step_cnt, srs[0].info)
    #     count = 0
    #     import time
    #     st = time.monotonic()
    #     while not done:
    #         count += 1
    #         srs = env.step([space.sample() for space in env.action_spaces])
    #         for sr in srs:
    #             assert isinstance(sr.obs, dict)
    #             assert sr.obs['obs'].shape == (3, 72, 96), sr.obs['obs'].shape
    #             assert sr.obs['INSTR'].shape == (16,), sr.obs['INSTR'].shape
    #             assert isinstance(sr.info, dict) or sr.info is None
    #             assert sr.reward.shape == (1,)
    #             assert sr.done.shape == (1,)
    #             assert sr.truncated.shape == (1,)
    #         step_cnt += 1
    #         done = all(sr.done for sr in srs)
    #         # print(step_cnt, srs[0].info)

    #     print(count, time.monotonic() - st, float(count/(time.monotonic()-st)))

    # env.close()
    # del env
    # print(level_names)
