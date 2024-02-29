import mock
import numpy as np
import torch
import unittest

from api.environment import StepResult
from api.env_utils import DiscreteAction
from api.policy import RolloutRequest, RolloutResult
from api.trainer import SampleBatch
from base.namedarray import NamedArray, recursive_aggregate, recursive_apply
from legacy.algorithm.ppo.mappo import MultiAgentPPO
from legacy.algorithm.ppo.game_policies.vizdoom_policy import VizDoomActorCriticPolicy

torch.set_num_threads(32)

action_dims = [3, 3, 2, 2, 11]
measurement_dim = 5
pixel_shape = (3, 72, 128)


class Observation(NamedArray):

    def __init__(self, obs, measurements=None):
        super().__init__(obs=obs, measurements=measurements)


def make_request(bs, with_hx=True, with_measurements=False):
    pixel = np.random.randint(0, 255, (bs, *pixel_shape)).astype(np.uint8)
    measurements = np.random.rand(bs, measurement_dim) if with_measurements else None
    on_reset = np.random.randint(0, 2, (bs, 1))
    is_evaluation = np.random.randint(0, 2, (bs, 1))
    policy_state = NamedArray(hx=np.random.randn(bs, 1, 512)) if with_hx else None
    return RolloutRequest(obs=Observation(pixel, measurements),
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
    with_measurements=False,
):
    pixel = np.random.randint(0, 255, (ep_len, bs, *pixel_shape)).astype(np.uint8)
    measurements = np.random.rand(ep_len, bs, measurement_dim) if with_measurements else None
    action = []
    for act_dim in action_dims:
        action.append(np.random.randint(0, act_dim, (
            ep_len,
            bs,
        )))
    action = DiscreteAction(np.stack(action, -1))
    on_reset = np.random.randint(0, 2, (ep_len + 1, bs, 1))
    policy_state = NamedArray(hx=np.random.randn(ep_len, bs, 1, 512)) if with_hx else None
    _mask = np.random.randint(0, 2, (ep_len, bs, 1))
    done = on_reset[1:] * _mask
    truncated = on_reset[1:] * (1 - _mask)
    return SampleBatch(
        obs=Observation(pixel, measurements),
        on_reset=on_reset[:-1],
        done=done,
        truncated=truncated,
        action=action,
        reward=np.random.randn(ep_len, bs, 1) * (1 - on_reset[1:]),
        log_probs=np.random.randn(ep_len, bs, 1),
        value=np.random.randn(ep_len, bs, 1) * (1 - done),
        policy_state=policy_state,
        info_mask=np.zeros((ep_len, bs, 1), dtype=np.uint8),
    )


class VizDoomPolicyTest(unittest.TestCase):

    def setUp(self):
        self.policy_configs = [
            dict(obs_shapes=dict(obs=pixel_shape, measurements=(measurement_dim,)),
                 num_rnn_layers=1,
                 popart=False),
            dict(obs_shapes=dict(obs=pixel_shape), num_rnn_layers=0),
            dict(obs_shapes=dict(obs=pixel_shape, measurements=(measurement_dim,)),
                 num_rnn_layers=1,
                 layernorm=True),
            dict(obs_shapes=dict(obs=pixel_shape, measurements=(measurement_dim,)),
                 num_dense_layers=1,
                 layernorm=True),
            dict(obs_shapes=dict(obs=pixel_shape, measurements=(measurement_dim,)),
                 num_dense_layers=1,
                 layernorm=True,
                 use_symmetric_kl=True),
        ]
        self.policy_collection = [
            VizDoomActorCriticPolicy(action_dims=action_dims, **config) for config in self.policy_configs
        ]

    def testRollout(self):
        for p, config in zip(self.policy_collection, self.policy_configs):
            req = make_request(
                bs=8,
                with_hx=config.get('num_rnn_layers', 1) > 0,
                with_measurements=('measurements' in config.get('obs_shapes', {})),
            )
            result = p.rollout(req)
            self.assertIsInstance(result, RolloutResult)
            self.assertIsInstance(result.action, DiscreteAction)
            self.assertIsInstance(result.action.x, np.ndarray)
            self.assertEqual(result.action.x.shape, (8, len(action_dims)))
            if config.get('num_rnn_layers', 1) == 0:
                self.assertIs(result.policy_state, None)
            else:
                self.assertEqual(result.policy_state.shape, req.policy_state.shape)

    def testAnalyze(self):
        for p, config in zip(self.policy_collection, self.policy_configs):
            bi = np.random.randint(0, 20)
            bo = np.random.randint(1, 10)
            trainer = MultiAgentPPO(p,
                                    use_soft_update=False,
                                    bootstrap_steps=bo,
                                    burn_in_steps=bi,
                                    popart=config.get('popart', True))
            sb = make_sample_batch(
                ep_len=bi + 10 + bo,
                bs=4,
                with_hx=config.get('num_rnn_layers', 1) > 0,
                with_measurements=('measurements' in config.get('obs_shapes', {})),
            )
            trainer.step(sb)


if __name__ == "__main__":
    unittest.main()