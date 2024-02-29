from unittest import mock
import numpy as np
import unittest

from base.testing import *

from api.env_utils import DiscreteAction
from api.policy import RolloutRequest
from api.trainer import SampleBatch
from base.namedarray import recursive_aggregate, recursive_apply
from legacy.algorithm.ppo.phasic_policy_gradient import _PPGLocalCache, PPGPhase1AnalyzedResult
from legacy.algorithm.ppo.actor_critic_policies.actor_critic_policy import ActorCriticPolicy, ActorCriticSeparate, PPORolloutAnalyzedResult
import base.namedarray


class ActorCriticTest(unittest.TestCase):

    def setUp(self) -> None:
        self.hidden_dim = 64
        self.test_dims = [
            dict(obs_dim={"obs": 30}, state_dim={"state": 41}, action_dim=19, rnn_layers=0),
            dict(obs_dim={"obs": 30}, state_dim={"state": 41}, action_dim=19, rnn_layers=1),
            dict(obs_dim={"obs": 30}, state_dim={"state": 41}, action_dim=[10, 5, 4], rnn_layers=0),
            dict(obs_dim={"obs": 30}, state_dim={"state": 41}, action_dim=[10, 5, 4], rnn_layers=1),
            dict(obs_dim={
                "obs": 10,
                "vec": (2, 10)
            },
                 state_dim={
                     "state": 41,
                     "vec": (2, 10)
                 },
                 action_dim=19,
                 rnn_layers=0),
            dict(obs_dim={
                "obs": 10,
                "vec": (2, 10)
            },
                 state_dim={
                     "state": 41,
                     "vec": (2, 10)
                 },
                 action_dim=19,
                 rnn_layers=1),
            dict(obs_dim={
                "obs": 10,
                "image": (2, 10, 10)
            },
                 state_dim={
                     "state": 41,
                     "vec": (2, 10)
                 },
                 action_dim=19,
                 rnn_layers=0),
            dict(obs_dim={
                "obs": 10,
                "image": (2, 10, 10)
            },
                 state_dim={
                     "state": 41,
                     "vec": (2, 10)
                 },
                 action_dim=19,
                 rnn_layers=1),
            dict(obs_dim={
                "obs": 10,
                "image": (2, 10, 10, 10)
            },
                 state_dim={
                     "state": 41,
                     "vec": (2, 10)
                 },
                 action_dim=19,
                 rnn_layers=0),
            dict(obs_dim={
                "obs": 10,
                "image": (2, 10, 10, 10)
            },
                 state_dim={
                     "state": 41,
                     "vec": (2, 10)
                 },
                 action_dim=19,
                 rnn_layers=1),
        ]

    def make_policy(self,
                    obs_dim,
                    state_dim,
                    action_dim,
                    rnn_layers,
                    shared_backbone=False,
                    auxiliary_head=False,
                    continuous_action=False):
        return ActorCriticPolicy(
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            value_dim=1,
            hidden_dim=self.hidden_dim,
            num_rnn_layers=rnn_layers,
            chunk_len=10,
            seed=0,
            shared_backbone=shared_backbone,
            auxiliary_head=auxiliary_head,
            continuous_action=continuous_action,
        )

    def rollout_request(self,
                        obs_dim,
                        state_dim,
                        action_dim,
                        rnn_layers,
                        is_eval=False,
                        on_reset=False,
                        shared=False):
        if shared:
            policy_state = base.namedarray.NamedArray(hx=np.random.random((rnn_layers, self.hidden_dim)))
        else:
            policy_state = base.namedarray.NamedArray(actor_hx=np.random.random(
                (rnn_layers, self.hidden_dim)),
                                                      critic_hx=np.random.random(
                                                          (rnn_layers, self.hidden_dim)))
        obs_dim = {k: v if isinstance(v, int) else v for k, v in obs_dim.items()}
        state_dim = {
            k: v if isinstance(v, int) else v
            for k, v in state_dim.items() if k not in obs_dim.keys()
        }
        action_dim = [action_dim] if isinstance(action_dim, int) else action_dim
        return RolloutRequest(obs=base.namedarray.from_dict(
            dict(**{
                k: np.random.random(v)
                for k, v in obs_dim.items()
            },
                 **{
                     k: np.random.random(v)
                     for k, v in state_dim.items()
                 },
                 available_action=np.random.randint(2, size=sum(action_dim)))),
                              policy_state=policy_state,
                              is_evaluation=np.array([is_eval], dtype=np.uint8),
                              on_reset=np.array([on_reset], dtype=np.uint8))

    def test_traceable(self):
        for dims in self.test_dims:
            for continuous_action in [True, False]:
                if not isinstance(dims["action_dim"], int) and continuous_action:
                    continue
                for shared_backbone in [True, False]:
                    policy = self.make_policy(**dims,
                                              continuous_action=continuous_action,
                                              shared_backbone=shared_backbone)
                    batch = recursive_aggregate([
                        self.sample_batch(**dims, sample_steps=10, shared=shared_backbone) for _ in range(3)
                    ], lambda x: np.stack(x, axis=1))
                    batch = recursive_apply(
                        batch, lambda x: torch.from_numpy(x).to(device=policy.device, dtype=torch.float32))
                    policy.trace_by_sample_batch(batch)

    def sample_batch(self, obs_dim, state_dim, action_dim, sample_steps, rnn_layers, shared=False):
        if shared:
            policy_state = base.namedarray.NamedArray(hx=np.random.random((rnn_layers, self.hidden_dim)))
        else:
            policy_state = base.namedarray.NamedArray(actor_hx=np.random.random(
                (rnn_layers, self.hidden_dim)),
                                                      critic_hx=np.random.random(
                                                          (rnn_layers, self.hidden_dim)))
        obs_dim = {k: v if isinstance(v, int) else v for k, v in obs_dim.items()}
        state_dim = {
            k: v if isinstance(v, int) else v
            for k, v in state_dim.items() if k not in obs_dim.keys()
        }
        action_dim = [action_dim] if isinstance(action_dim, int) else action_dim

        return recursive_aggregate([
            SampleBatch(obs=base.namedarray.from_dict(
                dict(**{
                    k: np.random.random(v)
                    for k, v in obs_dim.items()
                },
                     **{
                         k: np.random.random(v)
                         for k, v in state_dim.items()
                     },
                     available_action=np.random.randint(2, size=sum(action_dim)))),
                        policy_state=policy_state,
                        on_reset=np.array([False], dtype=np.uint8),
                        action=DiscreteAction(
                            np.array([np.random.randint(a) for a in action_dim]).astype(np.int32)),
                        analyzed_result=PPORolloutAnalyzedResult(
                            value=np.random.random(1,).astype(np.int32),
                            log_probs=np.random.random(1,).astype(np.int32),
                        ),
                        reward=np.array([0], dtype=np.float32).astype(np.int32),
                        info=None) for _ in range(sample_steps)
        ], np.stack)

    def cache_entries(self, obs_dim, state_dim, action_dim, sample_steps, rnn_layers, shared=False):
        batch = recursive_aggregate([
            self.sample_batch(obs_dim, state_dim, action_dim, sample_steps, rnn_layers, shared)
            for _ in range(3)
        ], lambda x: np.stack(x, axis=1))
        return _PPGLocalCache.CacheEntry(sample=SampleBatch(
            obs=batch.obs,
            policy_state=batch.policy_state,
            on_reset=batch.on_reset,
            info_mask=batch.on_reset,
            value=np.random.randn(sample_steps, 1),
        ),
                                         action_dists=None)

    def test_attributes(self):
        self.assertRaises(AttributeError,
                          self.make_policy,
                          **self.test_dims[0],
                          auxiliary_head=True,
                          shared_backbone=True)
        self.assertRaises(AttributeError,
                          self.make_policy,
                          **self.test_dims[2],
                          auxiliary_head=True,
                          shared_backbone=True)

    def test_reset_rollout(self):
        for dims in self.test_dims:
            self.tearDown()
            a = 1 if isinstance(dims["action_dim"], int) else len(dims["action_dim"])
            policy1 = self.make_policy(**dims)
            request1 = recursive_aggregate([self.rollout_request(**dims) for _ in range(3)], np.stack)
            result1 = policy1.rollout(request1)
            self.assertEqual(result1.action.x.shape, (3, a))
            policy1a = self.make_policy(**dims, auxiliary_head=True)
            request1a = recursive_aggregate([self.rollout_request(**dims) for _ in range(3)], np.stack)
            result1a = policy1a.rollout(request1a)
            self.assertEqual(result1a.action.x.shape, (3, a))
            policy2 = self.make_policy(**dims, shared_backbone=True)
            request2 = recursive_aggregate([self.rollout_request(**dims, shared=True) for _ in range(3)],
                                           np.stack)
            result2 = policy2.rollout(request2)
            self.assertEqual(result2.action.x.shape, (3, a))
            if a == 1:
                policy2 = self.make_policy(**dims, auxiliary_head=True, continuous_action=True)
                request2 = recursive_aggregate([self.rollout_request(**dims) for _ in range(3)], np.stack)
                result2 = policy2.rollout(request2)
                self.assertEqual(result2.action.x.shape, (3, dims["action_dim"]))

    @mock.patch("torch.distributions.Categorical.sample",
                mock.MagicMock(return_value=torch.LongTensor([[0, 0, 0]])))
    def test_rollout_sample_eval(self):
        action_prob_max_at_3 = [0, 0, 0, 0.7, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        action_prob_max_at_4 = [0, 0, 0, 0.1, 0.7, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        action_prob_max_at_5 = [0, 0, 0, 0.1, 0.1, 0.7, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        mock_action_probs = torch.Tensor([[action_prob_max_at_3, action_prob_max_at_4, action_prob_max_at_5]])
        self.assertEqual(mock_action_probs.size(), torch.Size((1, 3, 19)))
        with mock.patch(
                "legacy.algorithm.ppo.actor_critic_policies.actor_critic_policy.ActorCriticSeparate.forward",
                mock.MagicMock(return_value=((torch.log(mock_action_probs),), (torch.randn(1, 3, 1),),
                                             (torch.randn(1, 3, self.hidden_dim),
                                              torch.randn(1, 3, self.hidden_dim))))):

            policy = self.make_policy(**self.test_dims[0])
            requests = recursive_aggregate([self.rollout_request(**self.test_dims[0]) for _ in range(3)],
                                           np.stack)

            results = policy.rollout(requests)
            self.assertEqual(results.action.x.shape, (3, 1))
            self.assertEqual(results.analyzed_result.log_probs.shape, (3, 1))
            self.assertTrue((results.action.x == 0).all())

            requests = recursive_aggregate(
                [self.rollout_request(**self.test_dims[0], is_eval=True) for _ in range(3)], np.stack)
            results = policy.rollout(requests)
            self.assertEqual(results.action.x.shape, (3, 1))
            self.assertEqual(results.analyzed_result.log_probs.shape, (3, 1))
            self.assertTrue((results.action.x == np.array([[3], [4], [5]])).all())

            policy = self.make_policy(**self.test_dims[2])
            action_prob_max_at_3_2_3 = [
                0, 0, 0, 0.7, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0.1, 0.5, 0.4, 0, 0, 0, 0.1, 0.9
            ]
            action_prob_max_at_4_3_2 = [
                0, 0, 0, 0.1, 0.7, 0.1, 0.1, 0, 0, 0, 0.3, 0.1, 0.2, 0.4, 0, 0, 0, 0.9, 0.1
            ]
            action_prob_max_at_5_2_0 = [
                0, 0, 0, 0.1, 0.1, 0.7, 0.1, 0, 0, 0, 0.3, 0.1, 0.4, 0.2, 0, 0.9, 0, 0., 0.1
            ]
            mock_action_probs = torch.Tensor(
                [[action_prob_max_at_3_2_3, action_prob_max_at_4_3_2, action_prob_max_at_5_2_0]])
            self.assertEqual(mock_action_probs.size(), torch.Size((1, 3, 19)))
        with mock.patch(
                "legacy.algorithm.ppo.actor_critic_policies.actor_critic_policy.ActorCriticSeparate.forward",
                mock.MagicMock(return_value=((torch.log(mock_action_probs),), (torch.randn(1, 3, 1),),
                                             (torch.randn(1, 3, self.hidden_dim),
                                              torch.randn(1, 3, self.hidden_dim))))):
            requests = recursive_aggregate(
                [self.rollout_request(**self.test_dims[2], is_eval=False, on_reset=True) for _ in range(3)],
                np.stack)
            results = policy.rollout(requests)
            self.assertEqual(results.action.x.shape, (3, 3))
            self.assertEqual(results.analyzed_result.log_probs.shape, (3, 1))
            self.assertTrue((results.action.x == 0).all())

            requests = recursive_aggregate(
                [self.rollout_request(**self.test_dims[2], is_eval=True, on_reset=True) for _ in range(3)],
                np.stack)
            results = policy.rollout(requests)
            self.assertEqual(results.action.x.shape, (3, 3))
            self.assertEqual(results.analyzed_result.log_probs.shape, (3, 1))
            self.assertTrue((results.action.x == np.array([[3, 2, 3], [4, 3, 2], [5, 2, 0]])).all())

    def test_ppo_analyze(self):
        for dims in self.test_dims:
            self.tearDown()
            for continuous_action in [True, False]:
                if not isinstance(dims["action_dim"], int) and continuous_action:
                    continue
                for shared_backbone in [True, False]:
                    policy = self.make_policy(**dims,
                                              continuous_action=continuous_action,
                                              shared_backbone=shared_backbone)
                    batch = recursive_aggregate([
                        self.sample_batch(**dims, sample_steps=10, shared=shared_backbone) for _ in range(3)
                    ], lambda x: np.stack(x, axis=1))
                    batch = recursive_apply(
                        batch, lambda x: torch.from_numpy(x).to(device=policy.device, dtype=torch.float32))
                    self.assertRaises(RuntimeError, policy.analyze, batch, target="ppg_ppo_phase")
                    analyzed_result = policy.analyze(batch, target="ppo")
                    self.assertEqual(analyzed_result.state_values.size(), torch.Size((10, 3, 1)))
                    self.assertEqual(analyzed_result.entropy.size(), torch.Size((10, 3, 1)))
                    self.assertEqual(analyzed_result.new_action_log_probs.size(), torch.Size((10, 3, 1)))
                    self.assertEqual(analyzed_result.old_action_log_probs.size(), torch.Size((10, 3, 1)))

    def test_ppg_phase1_analyze(self):
        for dims in self.test_dims:
            self.tearDown()
            for continuous_action in [True, False]:
                if not isinstance(dims["action_dim"], int) and continuous_action:
                    continue
                policy = self.make_policy(**dims, continuous_action=continuous_action, auxiliary_head=True)
                batch = recursive_aggregate([self.sample_batch(**dims, sample_steps=10) for _ in range(3)],
                                            lambda x: np.stack(x, axis=1))
                batch = recursive_apply(
                    batch, lambda x: torch.from_numpy(x).to(device=policy.device, dtype=torch.float32))
                self.assertRaises(RuntimeError, policy.analyze, batch, target="ppo")
                analyzed_result = policy.analyze(batch, target="ppg_ppo_phase")
                assert isinstance(analyzed_result, PPGPhase1AnalyzedResult)
                self.assertEqual(analyzed_result.reward.size(), torch.Size((10, 3, 1)))
                self.assertEqual(analyzed_result.aux_values.size(), torch.Size((10, 3, 1)))
                self.assertEqual(analyzed_result.state_values.size(), torch.Size((10, 3, 1)))
                self.assertEqual(analyzed_result.entropy.size(), torch.Size((10, 3, 1)))
                self.assertEqual(analyzed_result.new_action_log_probs.size(), torch.Size((10, 3, 1)))
                self.assertEqual(analyzed_result.old_action_log_probs.size(), torch.Size((10, 3, 1)))

                entry = self.cache_entries(**dims, shared=False, sample_steps=10)
                batch = recursive_apply(
                    entry.sample, lambda x: torch.from_numpy(x).to(device=policy.device, dtype=torch.float32))

                analyzed_result = policy.analyze(batch, target="ppg_aux_phase")
                self.assertEqual(analyzed_result.auxiliary_value.size(), torch.Size((10, 3, 1)))
                self.assertEqual(analyzed_result.predicted_value.size(), torch.Size((10, 3, 1)))
                if continuous_action:
                    for dist in analyzed_result.action_dists:
                        self.assertEqual(dist.loc.size(), torch.Size((10, 3, 19)))
                        self.assertEqual(dist.scale.size(), torch.Size((10, 3, 19)))
                else:
                    action_dims = [dims["action_dim"]] if isinstance(dims["action_dim"],
                                                                     int) else dims["action_dim"]
                    for dist, d in zip(analyzed_result.action_dists, action_dims):
                        self.assertEqual(dist.logits.size(), torch.Size((10, 3, d)))


if __name__ == '__main__':
    unittest.main()
