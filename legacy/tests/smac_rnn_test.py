from absl import flags
from unittest.mock import MagicMock
import collections
import numpy as np
import unittest

from base.testing import *

from api.environment import StepResult
from api.policy import RolloutRequest, RolloutResult
from api.trainer import SampleBatch
from base.namedarray import recursive_aggregate, recursive_apply
from legacy.algorithm.ppo.actor_critic_policies.actor_critic_policy import PPORolloutAnalyzedResult
from legacy.algorithm.ppo.mappo import MultiAgentPPO, SampleAnalyzedResult
from legacy.algorithm.ppo.game_policies.smac_rnn import SMACPolicy, SMACPolicyState
from legacy.environment.smac.smac_env import SMACAction
import base.namedarray

map_name = '3s5z_vs_3s6z'
_OBS_DIM = 268
_STATE_DIM = 318
_ACT_DIM = 15
_AGENT_COUNT = 8
_TOTAL_STEPS = 40

_AGENT_SPECIFIC_OBS_SHAPE = collections.OrderedDict([('obs_allies', (7, 23)), ('obs_enemies', (9, 8)),
                                                     ('obs_move', (1, 4)), ('obs_self', (31,)),
                                                     ('obs_mask', (17,))])
_AGENT_SPECIFIC_STATE_SHAPE = collections.OrderedDict([('state_allies', (7, 26)), ('state_enemies', (9, 11)),
                                                       ('state_move', (1, 4)), ('state_self', (33,)),
                                                       ('state_mask', (17,))])

_SHARED = True
_AGENT_SPECIFIC_OBS = False
_AGENT_SPECIFIC_STATE = False

_HIDDEN_DIM = 128


class SMACNetTest(unittest.TestCase):

    def _get_local_obs(self, shape_prefix):
        if _AGENT_SPECIFIC_OBS:
            local_obs = {}
            for k, shape in _AGENT_SPECIFIC_OBS_SHAPE.items():
                if 'mask' not in k:
                    local_obs[k] = np.random.randn(*shape_prefix, *shape)
                else:
                    local_obs[k] = np.random.randint(0, 2, (*shape_prefix, *shape))
            return local_obs
        else:
            return np.random.randn(*shape_prefix, _OBS_DIM)

    def _get_state(self, shape_prefix):
        if _AGENT_SPECIFIC_STATE:
            state = {}
            for k, shape in _AGENT_SPECIFIC_STATE_SHAPE.items():
                if 'mask' not in k:
                    state[k] = np.random.randn(*shape_prefix, *shape)
                else:
                    state[k] = np.random.randint(0, 2, (*shape_prefix, *shape))
            return state
        else:
            return np.random.randn(*shape_prefix, _STATE_DIM)

    def make_env(self):
        ############### mock environment ###############
        env = MagicMock()
        shape_prefix = (_AGENT_COUNT,) if _SHARED else ()
        env.agent_count = 1 if _SHARED else _AGENT_COUNT
        env.step = MagicMock(return_value=[
            StepResult(obs=dict(local_obs=self._get_local_obs(shape_prefix),
                                state=self._get_state(shape_prefix),
                                available_action=np.random.randint(0, 2, (*shape_prefix, _ACT_DIM))),
                       reward=np.random.randn(*shape_prefix, 1),
                       done=np.zeros((*shape_prefix, 1), dtype=np.uint8),
                       info=dict(episode_length=np.random.randn(1),
                                 episode_return=np.random.randint(0, 2, (1,))))
            for _ in range(env.agent_count)
        ])
        env.reset = MagicMock(return_value=[
            StepResult(obs=dict(local_obs=self._get_local_obs(shape_prefix),
                                state=self._get_state(shape_prefix),
                                available_action=np.random.randint(0, 2, (*shape_prefix, _ACT_DIM))),
                       reward=np.zeros((*shape_prefix, 1), dtype=np.float32),
                       done=np.zeros((*shape_prefix, 1), dtype=np.uint8),
                       info=dict()) for _ in range(env.agent_count)
        ])
        env.step_count = 0
        env.seed = MagicMock()

        ############### real environment ###############
        # env = SMACEnvironment(map_name,
        #                       shared=_SHARED,
        #                       agent_specific_obs=_AGENT_SPECIFIC_OBS,
        #                       agent_specific_state=_AGENT_SPECIFIC_STATE)
        return env

    def setUp(self):
        self.ring_size = 2
        self.envs = envs = [self.make_env() for _ in range(self.ring_size)]
        for i, env in enumerate(self.envs):
            env.seed(i)
        self.envs_done = [False for _ in range(self.ring_size)]
        self.agent_count = self.envs[0].agent_count

        self.policy = SMACPolicy(map_name,
                                 _HIDDEN_DIM,
                                 2,
                                 shared=_SHARED,
                                 agent_specific_obs=_AGENT_SPECIFIC_OBS,
                                 agent_specific_state=_AGENT_SPECIFIC_STATE)

    def _check_step_results(self, step_results):
        self.assertEqual(len(step_results), self.agent_count)
        step_result = np.random.choice(step_results)
        self.assertIsInstance(step_result.obs, dict)
        if not _AGENT_SPECIFIC_OBS:
            self.assertEqual(step_result.obs["local_obs"].shape[-1], _OBS_DIM)
        else:
            for k, shape in _AGENT_SPECIFIC_OBS_SHAPE.items():
                self.assertEqual(step_result.obs["local_obs"][k].shape[-len(shape):], shape)

        if not _AGENT_SPECIFIC_STATE:
            self.assertEqual(step_result.obs["state"].shape[-1], _STATE_DIM)
        else:
            for k, shape in _AGENT_SPECIFIC_STATE_SHAPE.items():
                self.assertEqual(step_result.obs["state"][k].shape[-len(shape):], shape)
        self.assertEqual(step_result.obs["available_action"].shape[-1], _ACT_DIM)
        self.assertEqual(step_result.reward.shape[-1], 1)
        self.assertEqual(step_result.done.shape[-1], 1)

        if _SHARED:
            if not _AGENT_SPECIFIC_OBS:
                self.assertEqual(step_result.obs["local_obs"].shape[-2], _AGENT_COUNT)
            else:
                for k, shape in _AGENT_SPECIFIC_OBS_SHAPE.items():
                    self.assertEqual(step_result.obs["local_obs"][k].shape[-len(shape) - 1], _AGENT_COUNT)

            if not _AGENT_SPECIFIC_STATE:
                self.assertEqual(step_result.obs["state"].shape[-2], _AGENT_COUNT)
            else:
                for k, shape in _AGENT_SPECIFIC_STATE_SHAPE.items():
                    self.assertEqual(step_result.obs["state"][k].shape[-len(shape) - 1], _AGENT_COUNT)
            self.assertEqual(step_result.obs["available_action"].shape[-2], _AGENT_COUNT)
            self.assertEqual(step_result.reward.shape[-2], _AGENT_COUNT)
            self.assertEqual(step_result.done.shape[-2], _AGENT_COUNT)

    def _check_rollout_results(self, rollout_results):
        self.assertTrue(len(rollout_results.analyzed_result.log_probs) % self.agent_count == 0)
        bs = rollout_results.analyzed_result.log_probs.shape[0]
        self.assertEqual(rollout_results.action.x.shape[0], bs)
        self.assertEqual(rollout_results.analyzed_result.log_probs.shape[0], bs)
        self.assertEqual(rollout_results.policy_state.actor_hx.shape[0], bs)
        self.assertEqual(rollout_results.policy_state.critic_hx.shape[0], bs)

        self.assertEqual(rollout_results.action.x.shape[-1], 1)
        self.assertEqual(rollout_results.analyzed_result.log_probs.shape[-1], 1)

        if _SHARED:
            self.assertEqual(rollout_results.action.x.shape[-2], _AGENT_COUNT)
            self.assertEqual(rollout_results.analyzed_result.log_probs.shape[-2], _AGENT_COUNT)
            self.assertEqual(rollout_results.policy_state.actor_hx.shape[-3:], (_AGENT_COUNT, 1, _HIDDEN_DIM))
            self.assertEqual(rollout_results.policy_state.critic_hx.shape[-3:],
                             (_AGENT_COUNT, 1, _HIDDEN_DIM))
        else:
            self.assertEqual(rollout_results.policy_state.actor_hx.shape[-2:], (1, _HIDDEN_DIM))
            self.assertEqual(rollout_results.policy_state.critic_hx.shape[-2:], (1, _HIDDEN_DIM))

    def test_rollout(self):
        envs = self.envs
        policy = self.policy
        policy.eval_mode()
        requests = []
        for env in envs:
            step_results = env.reset()
            env.available_actions = [step.obs["available_action"] for step in step_results]
            requests += [
                RolloutRequest(obs=base.namedarray.from_dict(step_result.obs),
                               policy_state=None,
                               is_evaluation=np.random.randint(0, 2,
                                                               step_result.reward.shape).astype(np.uint8),
                               on_reset=np.random.randint(0, 2, step_result.reward.shape).astype(np.uint8))
                for step_result in step_results
            ]

        requests = recursive_aggregate(requests, np.stack)
        rollout_results = self.policy.rollout(requests)
        self.assertEqual(rollout_results.analyzed_result.log_probs.shape[0], len(envs) * self.agent_count)
        self._check_rollout_results(rollout_results)

        while not all(self.envs_done):
            requests = []
            for i, env in enumerate(envs):
                if not self.envs_done[i]:
                    cur_rollout_results = rollout_results[:self.agent_count]
                    rollout_results = rollout_results[self.agent_count:]
                    actions = [cur_rollout_results[i].action for i in range(self.agent_count)]
                    for action, available_action in zip(actions, env.available_actions):
                        self.assertTrue((action.x % 1 == 0).all())
                        bs = available_action.shape[0]
                        if _SHARED:
                            self.assertTrue(
                                (available_action[np.arange(bs),
                                                  action.x.squeeze(-1).astype(np.int32)] == 1).all())
                        else:
                            self.assertTrue(available_action[int(action.x.item())] == 1)
                    step_results = env.step(actions)

                    # comment this block if the environment is not a mock
                    ##########################################################
                    env.step_count += 1
                    num_done_agents = env.step_count // (_TOTAL_STEPS // self.agent_count)
                    for agent_idx, step in enumerate(step_results):
                        step.done[:] = (agent_idx < num_done_agents)
                    ##########################################################

                    env.available_actions = [step.obs["available_action"] for step in step_results]
                    self._check_step_results(step_results)
                    self.envs_done[i] = self.envs_done[i] or all(
                        [result.done.all() for result in step_results])
                    if not self.envs_done[i]:
                        requests += [
                            RolloutRequest(
                                obs=base.namedarray.from_dict(step_result.obs),
                                policy_state=cur_rollout_results[i].policy_state,
                                is_evaluation=np.random.randint(0, 2,
                                                                step_result.reward.shape).astype(np.uint8),
                                on_reset=np.random.randint(0, 2, step_result.reward.shape).astype(np.uint8))
                            for i, step_result in enumerate(step_results)
                        ]

            if len(requests) > 0:
                requests = recursive_aggregate(requests, np.stack)
                rollout_results = self.policy.rollout(requests)
                self._check_rollout_results(rollout_results)

    def test_others(self):
        policy = self.policy
        policy.parameters()
        sd = policy.get_checkpoint()
        policy.load_checkpoint(sd)

    def _get_sample_batch(self):
        T, B = 11, 20
        shape_prefix = (T, B, _AGENT_COUNT) if _SHARED else (T, B)
        on_reset = np.random.randint(0, 2, (T + 1, B, _AGENT_COUNT,
                                            1)) if _SHARED else np.random.randint(0, 2, (T + 1, B, 1))
        rnn_shape = (T, B, _AGENT_COUNT, 1, _HIDDEN_DIM) if _SHARED else (T, B, 1, _HIDDEN_DIM)
        return SampleBatch(obs=base.namedarray.from_dict(
            dict(local_obs=self._get_local_obs(shape_prefix),
                 state=self._get_state(shape_prefix),
                 available_action=np.random.randint(0, 2, (*shape_prefix, _ACT_DIM)),
                 is_alive=np.random.randint(0, 2, (*shape_prefix, 1)))),
                           policy_state=SMACPolicyState(np.random.randn(*rnn_shape),
                                                        np.random.randn(*rnn_shape)),
                           on_reset=on_reset[:-1],
                           done=on_reset[1:],
                           action=SMACAction(np.random.randint(0, _ACT_DIM, (*shape_prefix, 1))),
                           analyzed_result=PPORolloutAnalyzedResult(
                               log_probs=np.random.randn(*shape_prefix, 1),
                               value=np.random.randn(*shape_prefix, 1),
                           ),
                           info_mask=np.zeros(shape=(*shape_prefix, 1), dtype=np.int8),
                           reward=np.random.randn(*shape_prefix, 1) * (1 - on_reset[1:]),
                           on_episode_closure=on_reset[1:],
                           info=base.namedarray.from_dict({}))

    def test_analyze(self):
        policy = self.policy
        envs = self.envs
        policy.train_mode()
        T, B = 11, 20
        bootstrap = 1
        for _ in range(2):
            sample = self._get_sample_batch()
            tensor_sample = recursive_apply(
                sample, lambda x: torch.from_numpy(x).to(device=self.policy.device, dtype=torch.float32))
            analyze_result = self.policy.analyze(tensor_sample[:-bootstrap])
            self.assertIsInstance(analyze_result, SampleAnalyzedResult)
            if _SHARED:
                self.assertEqual(analyze_result.new_action_log_probs.shape,
                                 (T - bootstrap, B, _AGENT_COUNT, 1))
                self.assertEqual(analyze_result.old_action_log_probs.shape,
                                 (T - bootstrap, B, _AGENT_COUNT, 1))
                self.assertEqual(analyze_result.state_values.shape, (T - bootstrap, B, _AGENT_COUNT, 1))
                self.assertEqual(analyze_result.entropy.shape, (T - bootstrap, B, _AGENT_COUNT, 1))
            else:
                self.assertEqual(analyze_result.new_action_log_probs.shape, (T - bootstrap, B, 1))
                self.assertEqual(analyze_result.old_action_log_probs.shape, (T - bootstrap, B, 1))
                self.assertEqual(analyze_result.state_values.shape, (T - bootstrap, B, 1))
                self.assertEqual(analyze_result.entropy.shape, (T - bootstrap, B, 1))

            trainer = MultiAgentPPO(self.policy)
            trainer.step(sample)

    def tearDown(self):
        for env in self.envs:
            env.close()


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    unittest.main()
