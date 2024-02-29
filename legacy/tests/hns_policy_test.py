import gym
import numpy as np
import unittest

from base.testing import *

from legacy.algorithm.ppo.actor_critic_policies.actor_critic_policy import PPORolloutAnalyzedResult
from legacy.algorithm.ppo.game_policies.hns_policy import HNSPolicy, HNSPolicyState
from api.policy import RolloutRequest, RolloutResult
from api.trainer import SampleBatch
from legacy.algorithm.ppo.mappo import MultiAgentPPO
from base.namedarray import recursive_aggregate, NamedArray
import api.env_utils
import base.namedarray
import api.policy
import api.config


class HNSPolicyTest(unittest.TestCase):

    def setUp(self):
        self.observation_space = {
            'agent_qpos_qvel': (5, 10),
            'box_obs': (9, 15),
            'lidar': (1, 30),
            'mask_aa_obs': (5,),
            'mask_aa_obs_spoof': (5,),
            'mask_ab_obs': (9,),
            'mask_ab_obs_spoof': (9,),
            'mask_ar_obs': (2,),
            'mask_ar_obs_spoof': (2,),
            'observation_self': (10,),
            'ramp_obs': (2, 15)
        }
        self.action_space = api.env_utils.DiscreteActionSpace(gym.spaces.MultiDiscrete([11, 11, 11, 2, 2]))
        cfg = api.config.Policy(type_='hns',
                                args=dict(obs_space=self.observation_space, act_dims=[11, 11, 11, 2, 2]))
        self.policy = api.policy.make(cfg)

    def _check_rollout_result(self, result):
        self.assertIsInstance(result.action, api.env_utils.DiscreteAction)
        self.assertEqual(result.action.x.shape[-1], 5)
        self.assertIsInstance(result.policy_state, HNSPolicyState)
        self.assertIsInstance(result.analyzed_result.value, np.ndarray)
        self.assertIsInstance(result.analyzed_result.log_probs, np.ndarray)
        self.assertEqual(result.analyzed_result.value.shape[-1], 1)
        self.assertEqual(result.analyzed_result.log_probs.shape[-1], 1)

    def test_rollout(self):
        bs = 7
        request = RolloutRequest(obs=base.namedarray.recursive_aggregate(
            [base.namedarray.from_dict(self._sample_obs()) for _ in range(bs)], np.stack),
                                 on_reset=np.ones((bs, 1), dtype=np.uint8),
                                 is_evaluation=np.random.randint(0, 2, (bs, 1)),
                                 policy_state=NamedArray(actor_hx=np.random.randn(bs, 1, 512),
                                                         critic_hx=np.random.randn(bs, 1, 512)),
                                 client_id=np.random.randn(bs, 1),
                                 request_id=np.random.randn(bs, 1),
                                 received_time=np.random.randn(bs, 1))
        rollout_result = self.policy.rollout(request)
        self.assertIsInstance(rollout_result, RolloutResult)
        self._check_rollout_result(rollout_result)

        for _ in range(20):
            request = RolloutRequest(obs=base.namedarray.recursive_aggregate(
                [base.namedarray.from_dict(self._sample_obs()) for _ in range(bs)], np.stack),
                                     on_reset=np.ones((bs, 1), dtype=np.uint8),
                                     is_evaluation=np.random.randint(0, 2, (bs, 1)),
                                     client_id=np.random.randn(bs, 1),
                                     request_id=np.random.randn(bs, 1),
                                     received_time=np.random.randn(bs, 1),
                                     policy_state=rollout_result.policy_state)
            rollout_result = self.policy.rollout(request)
            self.assertIsInstance(rollout_result, RolloutResult)
            self._check_rollout_result(rollout_result)

    def _sample_obs(self):
        obs = {}
        for k, shape in self.observation_space.items():
            if 'mask' not in k:
                obs[k] = np.random.randn(*shape)
            else:
                obs[k] = np.random.randint(0, 2, shape).astype(np.uint8)
        return obs

    def _sample_info(self):
        info = {
            "max_box_move_prep": np.zeros(1, dtype=np.float32),
            "max_box_move": np.zeros(1, dtype=np.float32),
            "num_box_lock_prep": np.zeros(1, dtype=np.float32),
            "num_box_lock": np.zeros(1, dtype=np.float32),
            "max_ramp_move_prep": np.zeros(1, dtype=np.float32),
            "max_ramp_move": np.zeros(1, dtype=np.float32),
            "num_ramp_lock_prep": np.zeros(1, dtype=np.float32),
            "num_ramp_lock": np.zeros(1, dtype=np.float32),
            "episode_return": np.zeros(1, dtype=np.float32),
            "hider_return": np.zeros(1, dtype=np.float32),
            "seeker_return": np.zeros(1, dtype=np.float32),
        }
        for k in info.keys():
            if k != 'episode_return':
                info[k] = np.random.rand(1)
        return info

    def _make_traj(self, traj_len):
        bs = 3
        request = RolloutRequest(obs=base.namedarray.recursive_aggregate(
            [base.namedarray.from_dict(self._sample_obs()) for _ in range(bs)], np.stack),
                                 on_reset=np.zeros((bs, 1), dtype=np.uint8),
                                 is_evaluation=np.random.randint(0, 2, (bs, 1)),
                                 policy_state=NamedArray(actor_hx=np.random.randn(bs, 1, 512),
                                                         critic_hx=np.random.randn(bs, 1, 512)),
                                 client_id=np.random.randn(bs, 1),
                                 request_id=np.random.randn(bs, 1),
                                 received_time=np.random.randn(bs, 1))
        rollout_result = self.policy.rollout(request)

        on_reset = np.zeros((traj_len + 1, bs, 1), dtype=np.uint8)
        dt_mask = np.random.randint(0, 2, (traj_len, bs, 1)).astype(np.uint8)
        done = on_reset[1:] * dt_mask
        truncated = on_reset[1:] * (1 - dt_mask)
        return SampleBatch(
            obs=recursive_aggregate([request.obs for _ in range(traj_len)], np.stack),
            on_reset=on_reset[:-1],
            done=done,
            truncated=truncated,
            action=recursive_aggregate([rollout_result.action for _ in range(traj_len)], np.stack),
            reward=np.random.randn(traj_len, bs, 1) * (1 - on_reset[1:]),
            info_mask=np.zeros((traj_len, bs, 1), dtype=np.uint8),
            policy_state=recursive_aggregate([request.policy_state for _ in range(traj_len)], np.stack),
            analyzed_result=recursive_aggregate([rollout_result.analyzed_result for _ in range(traj_len)],
                                                np.stack),
        )

    def test_analyze(self):
        traj_len = 11
        algo = MultiAgentPPO(self.policy)
        # algo.distributed(rank=0, world_size=1, init_method="tcp://localhost:7777")

        sample = self._make_traj(traj_len)
        for _ in range(5):
            training_step_result = algo.step(sample)


if __name__ == '__main__':
    unittest.main()
