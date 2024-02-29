import mock
import numpy as np
import torch
import unittest

from api.environment import StepResult
from api.env_utils import DiscreteAction
from api.policy import RolloutRequest, RolloutResult
from api.trainer import SampleBatch
from base.namedarray import NamedArray, recursive_aggregate, recursive_apply
from legacy.algorithm.q_learning.deep_q_learning import DeepQLearning
from legacy.algorithm.q_learning.game_policies.atari_dqn_policy import AtariDQNPolicy, DQNPolicyState

torch.set_num_threads(32)


class Observation(NamedArray):

    def __init__(self, obs, action, reward, epsilon):
        super().__init__(obs=obs, action=action, reward=reward, epsilon=epsilon)


def make_request(bs, with_hx=True):
    epsilon = np.random.rand(bs, 1)
    pixel = np.random.randint(0, 255, (bs, 4, 84, 84)).astype(np.uint8)
    act = np.random.randint(0, 6, (bs,))
    action = np.zeros((bs, 1), dtype=np.float32)
    action[act] = 1
    reward = np.random.randn(bs, 1)
    on_reset = np.random.randint(0, 2, (bs, 1))
    is_evaluation = np.random.randint(0, 2, (bs, 1))
    policy_state = DQNPolicyState(
        np.random.randn(bs, 1, 512 * 2) if with_hx else None, np.random.randn(bs, 1))
    return RolloutRequest(obs=Observation(pixel, action, reward, epsilon),
                          policy_state=policy_state,
                          is_evaluation=is_evaluation,
                          on_reset=on_reset,
                          step_count=None,
                          client_id=None,
                          request_id=None,
                          received_time=None)


def make_sample_batch(
    ep_len,
    bs,
    with_hx=True,
):
    epsilon = np.random.rand(bs, 1)
    pixel = np.random.randint(0, 255, (ep_len, bs, 4, 84, 84)).astype(np.uint8)
    on_reset = np.random.randint(0, 2, (ep_len + 1, bs, 1))
    action = DiscreteAction(np.random.randint(0, 6, (ep_len, bs, 1)))
    act = np.random.randint(0, 6, (ep_len * bs,))
    last_action = np.zeros((ep_len * bs, 1), dtype=np.float32)
    last_action[act] = 1
    last_action = last_action.reshape(ep_len, bs, 1)
    last_reward = np.random.randn(ep_len, bs, 1)
    policy_state = DQNPolicyState(
        np.random.randn(ep_len, bs, 1, 512 * 2) if with_hx else None, np.random.randn(ep_len, bs, 1))
    _mask = np.random.randint(0, 2, (ep_len, bs, 1))
    done = on_reset[1:] * _mask
    truncated = on_reset[1:] * (1 - _mask)
    return SampleBatch(
        obs=Observation(pixel, last_action, last_reward, epsilon),
        on_reset=on_reset[:-1],
        done=done,
        truncated=truncated,
        action=action,
        reward=np.random.randn(ep_len, bs, 1),
        policy_state=policy_state,
        info_mask=np.zeros((ep_len, bs, 1), dtype=np.uint8),
    )


class AtariDqnPolicyTest(unittest.TestCase):

    def setUp(self):
        self.policy_configs = [
            dict(act_dim=6, dueling=False, use_double_q=True, chunk_len=10),
            dict(act_dim=6, dueling=True, use_double_q=False, chunk_len=10),
            dict(act_dim=6, dueling=True, use_double_q=False, num_rnn_layers=1, chunk_len=10),
            dict(act_dim=6, dueling=True, use_double_q=True, num_rnn_layers=1, chunk_len=10),
            dict(act_dim=6, dueling=True, use_double_q=True, chunk_len=10),
            dict(act_dim=6, dueling=True, use_double_q=True, chunk_len=10, rnn_include_last_action=True),
            dict(act_dim=6, dueling=True, use_double_q=True, chunk_len=10, rnn_include_last_reward=True),
            dict(act_dim=6,
                 dueling=True,
                 use_double_q=True,
                 chunk_len=10,
                 rnn_include_last_reward=True,
                 rnn_include_last_action=True),
        ]
        self.policy_collection = [AtariDQNPolicy(**config) for config in self.policy_configs]

    def testRollout(self):
        for p, config in zip(self.policy_collection, self.policy_configs):
            req = make_request(bs=8, with_hx=config.get('num_rnn_layers', 0) > 0)
            result = p.rollout(req)
            self.assertIsInstance(result, RolloutResult)
            self.assertIsInstance(result.action, DiscreteAction)
            self.assertIsInstance(result.action.x, np.ndarray)
            self.assertIsInstance(result.policy_state, DQNPolicyState)
            self.assertEqual(result.action.x.shape, (8, 1))
            if config.get('num_rnn_layers', 0) == 0:
                self.assertIs(result.policy_state.hx, None)
            else:
                self.assertFalse((np.abs(result.policy_state.hx - req.policy_state.hx) <= 1e-4).all())

    def testAnalyze(self):
        for p, config in zip(self.policy_collection, self.policy_configs):
            bi = np.random.randint(0, 20)
            bo = np.random.randint(1, 10)
            trainer = DeepQLearning(p, use_soft_update=False, bootstrap_steps=bo, burn_in_steps=bi)
            sb = make_sample_batch(
                bi + 10,
                4,
                with_hx=config.get('num_rnn_layers', 0) > 0,
            )
            trainer.step(sb)


if __name__ == "__main__":
    unittest.main()