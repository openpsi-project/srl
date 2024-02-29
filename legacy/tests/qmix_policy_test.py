from unittest import mock
import numpy as np
import torch
import unittest

from api.environment import StepResult
from api.env_utils import DiscreteAction
from api.policy import RolloutRequest, RolloutResult
from api.trainer import SampleBatch
from base.namedarray import NamedArray, recursive_aggregate, recursive_apply
from legacy.algorithm.q_learning.deep_q_learning import DeepQLearning
from legacy.algorithm.q_learning.qmix.qmix import QtotPolicy, PolicyState

torch.set_num_threads(32)

obs_dim = 115
state_dim = 115 - 11
hidden_dim = 128
n_agents = 3
act_dim = 19


class Observation(NamedArray):

    def __init__(self, obs, state):
        super().__init__(obs=obs, state=state)


def make_request(bs):
    obs = np.random.randn(bs, n_agents, obs_dim)
    state = np.random.randn(bs, n_agents, state_dim)
    on_reset = np.random.randint(0, 2, (bs, n_agents, 1))
    is_evaluation = np.random.randint(0, 2, (bs, n_agents, 1))
    policy_state = PolicyState(np.random.randn(bs, 1, n_agents, hidden_dim), np.random.randn(bs, 1))
    return RolloutRequest(obs=Observation(obs, state),
                          policy_state=policy_state,
                          is_evaluation=is_evaluation,
                          on_reset=on_reset,
                          step_count=None,
                          client_id=None,
                          request_id=None,
                          received_time=None)


def make_sample_batch(ep_len, bs):
    obs = np.random.randn(ep_len, bs, n_agents, obs_dim)
    state = np.random.randn(ep_len, bs, n_agents, state_dim)
    on_reset = np.random.randint(0, 2, (ep_len + 1, bs, n_agents, 1))
    action = DiscreteAction(np.random.randint(0, act_dim, (ep_len, bs, n_agents, 1)))
    policy_state = PolicyState(np.random.randn(ep_len, bs, 1, n_agents, hidden_dim),
                               np.random.randn(ep_len, bs, 1))
    _mask = np.random.randint(0, 2, (ep_len, bs, n_agents, 1))
    done = on_reset[1:] * _mask
    truncated = on_reset[1:] * (1 - _mask)
    sample = SampleBatch(
        obs=Observation(obs, state),
        on_reset=on_reset[:-1],
        done=done,
        truncated=truncated,
        action=action,
        reward=np.random.randn(ep_len, bs, n_agents, 1),
        policy_state=policy_state,
        info_mask=np.zeros((ep_len, bs, n_agents, 1), dtype=np.uint8),
    )
    sample.clear_metadata()
    return sample


class FootballQmixPolicyTest(unittest.TestCase):

    def setUp(self):
        from legacy.algorithm.q_learning.game_policies.football_qmix import FootballQMixPolicy
        seed = 1
        self.policy_configs = [
            dict(
                env_name="academy_3_vs_1_with_keeper",
                chunk_len=10,  # chunk length requires modification for different map
                use_double_q=True,
                mixer_type='vdn',
                epsilon_start=1.0,
                epsilon_finish=0.05,
                epsilon_anneal_time=5000,
                q_i_config=dict(hidden_dim=128, num_dense_layers=2, rnn_type="gru", num_rnn_layers=1),
                mixer_config=dict(
                    popart=False,
                    hidden_dim=64,
                    num_hypernet_layers=2,
                    hypernet_hidden_dim=64,
                ),
                state_use_all_local_obs=False,
                state_concate_all_local_obs=False,
                seed=seed,
            ),
        ]
        self.policy_collection = [FootballQMixPolicy(**config) for config in self.policy_configs]

    def testRollout(self):
        for p, config in zip(self.policy_collection, self.policy_configs):
            req = make_request(bs=8)
            res = p.rollout(req)
            ps = res.policy_state
            act = res.action.x
            self.assertIsInstance(ps, PolicyState)
            self.assertEqual(act.shape, (8, n_agents, 1))

    def testAnalyze(self):
        for p, config in zip(self.policy_collection, self.policy_configs):
            bi = np.random.randint(0, 20)
            bo = np.random.randint(1, 10)
            trainer = DeepQLearning(p, use_soft_update=False, bootstrap_steps=bo, burn_in_steps=bi)
            sb = make_sample_batch(bi + 10, 4)
            trainer.step(sb)


if __name__ == "__main__":
    unittest.main()