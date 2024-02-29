from unittest import mock
from unittest.mock import patch
import numpy as np
import socket
import unittest

from api import config
from api.policy import RolloutResult
from base.testing import *
from base.namedarray import recursive_aggregate
from api.env_utils import DiscreteAction as AerochessAction
from distributed.base.monitoring import DummyMonitor
from distributed.system.actor_worker import ActorWorker, _EnvTarget, Agent, AgentState
from api.environment import StepResult
from distributed.system.inference_stream import NameResolvingInferenceServer, NameResolvingInferenceClient
from distributed.system.sample_stream import NameResolvingSampleConsumer
import api.testing.aerochess_env
import distributed.base.name_resolve


def make_test_agent(
    inference_client,
    sample_producer,
    sample_steps=10,
    bootstrap_steps=1,
    send_after_done=False,
    send_full_trajectory=False,
    pad_trajectory=False,
    deterministic_action=False,
    send_concise_info=False,
    stack_frames=0,
    env_max_num_steps=20,
    index=0,
    burn_in_steps=0,
):
    return Agent(inference_client=inference_client,
                 sample_producer=sample_producer,
                 sample_steps=sample_steps,
                 bootstrap_steps=bootstrap_steps,
                 burn_in_steps=burn_in_steps,
                 send_after_done=send_after_done,
                 send_full_trajectory=send_full_trajectory,
                 trajectory_postprocessor='null',
                 pad_trajectory=pad_trajectory,
                 deterministic_action=deterministic_action,
                 send_concise_info=send_concise_info,
                 stack_frames=stack_frames,
                 env_max_num_steps=env_max_num_steps,
                 index=index)


def make_step_result(base_shape=(), fill_value=None, truncate=False, done=False, info=None):
    if truncate and done:
        raise ValueError("Done and truncate cannot be True at the same time.")
    if fill_value is None:
        obs = dict(obs=np.random.random((*base_shape, 1, 10)))
    else:
        obs = dict(obs=np.full((*base_shape, 1, 10), fill_value=fill_value))
    reward = np.random.random((*base_shape, 1))
    return StepResult(
        obs=dict(obs=obs),
        reward=reward,
        done=(np.ones_like(reward) * done).astype(np.uint8),
        info=info,
        truncated=(np.ones_like(reward) * truncate).astype(np.uint8),
    )


class AgentTest(unittest.TestCase):

    def setUp(self) -> None:
        distributed.base.name_resolve.reconfigure("memory")
        self.inference_client = mock.MagicMock()
        self.inference_client.default_policy_state = np.random.randn(10)
        self.inference_client.consume_result = mock.MagicMock(return_value=[
            RolloutResult(action=AerochessAction(np.array([39], dtype=np.int32)), policy_state=np.zeros(10))
        ])
        self.sample_producer = mock.MagicMock()
        self.sample_producer.post = mock.MagicMock(side_effect=lambda x: x)
        self._uninference()

    def _inference(self):
        self.inference_client.is_ready = mock.MagicMock(return_value=True)

    def _uninference(self):
        self.inference_client.is_ready = mock.MagicMock(return_value=False)

    def _get_agent_action(self, ag: Agent):
        self.assertEqual(ag.state, AgentState.WAITING_FOR_ACTION)
        self._inference()
        self.assertTrue(ag.ready_to_step())
        self._uninference()
        self.assertEqual(ag.state, AgentState.UNCOMSUMED_ACTION_READY)
        ag.consume_inference_result()
        self.assertEqual(ag.state, AgentState.READY_FOR_OBSERVATION)
        return ag.get_action()

    def test_agent_trivial(self):
        ag = make_test_agent(self.inference_client, self.sample_producer)

        # Upon reset
        self.assertEqual(ag.state, AgentState.READY_FOR_OBSERVATION)
        self.assertTrue(ag.ready_to_reset())
        ag.observe(make_step_result())
        self.assertEqual(ag.state, AgentState.WAITING_FOR_ACTION)
        self.inference_client.post_request.assert_called_once()
        self.assertFalse(ag.ready_to_step())

        # Agent will be ready when inference is done.
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        self.inference_client.reset_mock()

        # Another step.
        ag.observe(make_step_result())
        self.inference_client.post_request.assert_called_once()
        self.assertFalse(ag.ready_to_step())
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        self.inference_client.reset_mock()

        # After done the agent is ready to reset
        ag.observe(make_step_result(done=True))
        self.assertEqual(ag.state, AgentState.READY_FOR_OBSERVATION)
        self.assertTrue(ag.ready_to_reset())
        self.inference_client.post_request.assert_not_called()  # Not posting done steps.

    def test_step_count(self):
        ag = make_test_agent(self.inference_client, self.sample_producer)
        self.assertTrue(ag.ready_to_reset())
        for i in range(10):
            ag.observe(make_step_result())
            np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
            np.testing.assert_equal(self.inference_client.post_request.call_args[0][0].step_count, i)
            self.inference_client.reset_mock()

        ag.observe(make_step_result(done=True))
        for i in range(10):
            ag.observe(make_step_result())
            np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))

            np.testing.assert_equal(self.inference_client.post_request.call_args[0][0].step_count, i)
            self.inference_client.reset_mock()

    def test_agent_send_batch(self):
        ag = make_test_agent(self.inference_client, self.sample_producer, sample_steps=10, bootstrap_steps=2)
        ag.observe(make_step_result())
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        for i in range(12):  # in total 13 step_results, the final one is cached are not put into memory yet
            ag.observe(make_step_result(done=(i == 10)))
            self.sample_producer.assert_not_called()
            if i == 10:
                self.assertEqual(ag.state, AgentState.READY_FOR_OBSERVATION)
                self.assertEqual(self.inference_client.post_request.call_count, 11)
            else:
                np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        self.assertEqual(ag.state, AgentState.READY_FOR_OBSERVATION)
        self.sample_producer.post.assert_called_once()
        sample = self.sample_producer.post.mock_calls[0].args[0]
        np.testing.assert_array_equal(sample.action.x[:11], np.full((11, 1), fill_value=39, dtype=np.int32))
        np.testing.assert_array_equal(sample.action.x[-1], np.full((1,), fill_value=0, dtype=np.int32))

    def test_agent_send_trajectory(self):
        ag = make_test_agent(self.inference_client,
                             self.sample_producer,
                             bootstrap_steps=1,
                             send_full_trajectory=True,
                             pad_trajectory=True,
                             env_max_num_steps=30)
        ag.observe(make_step_result())
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        for i in range(20):
            ag.observe(make_step_result())
            self.sample_producer.post.assert_not_called()
            np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        ag.observe(make_step_result(done=True))  # not send at this step
        ag.observe(make_step_result())
        self.sample_producer.post.assert_called_once()
        self.assertEqual(self.sample_producer.post.call_args[0][0].length(dim=0), 31)
        self.sample_producer.reset_mock()

        ag = make_test_agent(self.inference_client,
                             self.sample_producer,
                             bootstrap_steps=1,
                             send_full_trajectory=True,
                             pad_trajectory=False)
        ag.observe(make_step_result())
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        for i in range(20):
            ag.observe(make_step_result())
            self.sample_producer.post.assert_not_called()
            np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        ag.observe(make_step_result(done=True))  # not send at this step
        ag.observe(make_step_result())
        self.sample_producer.post.assert_called_once()
        self.assertEqual(self.sample_producer.post.call_args[0][0].length(dim=0), 22)
        # 22 = 1 reset step + 20 undone steps + 1 done step

    def test_deterministic(self):
        ag = make_test_agent(self.inference_client, self.sample_producer, deterministic_action=False)
        ag.observe(make_step_result())
        self.assertFalse(self.inference_client.post_request.call_args[0][0].is_evaluation.any())
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        self.inference_client.reset_mock()
        ag.observe(make_step_result())
        self.assertFalse(self.inference_client.post_request.call_args[0][0].is_evaluation.any())
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        self.inference_client.reset_mock()

        ag = make_test_agent(self.inference_client, self.sample_producer, deterministic_action=True)
        ag.observe(make_step_result())
        self.assertTrue(self.inference_client.post_request.call_args[0][0].is_evaluation.all())
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))

        ag.observe(make_step_result())
        self.assertTrue(self.inference_client.post_request.call_args[0][0].is_evaluation.all())
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))

        self.inference_client.reset_mock()

    def test_skip_agent(self):
        ag = make_test_agent(self.inference_client, self.sample_producer)
        ag.observe(None)
        self.inference_client.post_request.assert_not_called()

        ag.observe(make_step_result())
        self.inference_client.post_request.assert_called_once()
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        self.inference_client.consume_result.assert_called_once()
        self.inference_client.reset_mock()
        ag.observe(None)
        self.inference_client.post_request.assert_not_called()
        ag.observe(make_step_result(done=True))
        self.inference_client.post_request.assert_not_called()

        # Async agent reset
        ag.observe(None)
        ag.observe(make_step_result())
        self.inference_client.post_request.assert_called_once()
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        ag.observe(make_step_result())
        self.assertTrue(self.inference_client.post_request.call_args_list[0][0][0].on_reset.all())
        self.assertFalse(self.inference_client.post_request.call_args_list[1][0][0].on_reset.any())
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))

        # Some corner cases
        self.inference_client.reset_mock()
        ag.observe(make_step_result(done=True))
        ag.observe(None)
        ag.observe(make_step_result(done=True))
        self.inference_client.post_request.assert_not_called()

    def test_post_last_step(self):
        ag = make_test_agent(self.inference_client,
                             self.sample_producer,
                             send_full_trajectory=True,
                             send_concise_info=True,
                             pad_trajectory=False)
        first_step = make_step_result()
        ag.observe(first_step)
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        for _ in range(17):
            ag.observe(make_step_result())
            np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))

        info = dict(episode_length=np.array([93]), episode_return=np.array([3.1416]))
        ag.observe(make_step_result(done=True, info=info))
        self.assertEqual(ag.state, AgentState.READY_FOR_OBSERVATION)
        ag.observe(None)
        self.assertEqual(ag.state, AgentState.READY_FOR_OBSERVATION)
        second_step = make_step_result()
        ag.observe(second_step)
        self.assertEqual(ag.state, AgentState.WAITING_FOR_ACTION)
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        info2 = dict(episode_length=np.array([13]), episode_return=np.array([111]))
        ag.observe(make_step_result(done=True, info=info2))
        ag.observe(make_step_result())

        self.assertEqual(self.sample_producer.post.call_count, 2)
        recv_sample1 = self.sample_producer.post.call_args_list[0][0][0]
        self.assertEqual(recv_sample1.length(dim=0), 1)
        np.testing.assert_allclose(first_step.obs["obs"], recv_sample1.obs[0]["obs"])
        for k in info.keys():
            np.testing.assert_allclose(info[k], recv_sample1.info[0][k])
        recv_sample2 = self.sample_producer.post.call_args_list[1][0][0]
        self.assertEqual(recv_sample2.length(dim=0), 1)
        np.testing.assert_allclose(second_step.obs["obs"], recv_sample2.obs[0]["obs"])
        for k in info.keys():
            np.testing.assert_allclose(info2[k], recv_sample2.info[0][k])

        self.sample_producer.reset_mock()

        ag = make_test_agent(self.inference_client,
                             self.sample_producer,
                             send_full_trajectory=False,
                             sample_steps=2,
                             send_concise_info=True,
                             pad_trajectory=False)
        first_step = make_step_result()
        info = dict(episode_length=np.array([93]), episode_return=np.array([3.1416]))
        ag.observe(first_step)
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        for _ in range(17):
            ag.observe(make_step_result())
            np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))

        ag.observe(make_step_result(done=True, info=info))
        second_step = make_step_result()
        second_info = dict(episode_length=np.array([32]), episode_return=np.array([2.236]))

        ag.observe(second_step)
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        for _ in range(17):
            ag.observe(make_step_result())
            np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))

        ag.observe(make_step_result(done=True, info=second_info))
        ag.observe(None)
        ag.observe(make_step_result())
        np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))

        ag.observe(make_step_result(done=True, info=None))
        ag.observe(make_step_result())
        self.sample_producer.post.assert_called_once()
        recv_sample = self.sample_producer.post.call_args_list[0][0][0]
        np.testing.assert_allclose(first_step.obs["obs"], recv_sample.obs[0]["obs"])
        for k in info.keys():
            np.testing.assert_allclose(info[k], recv_sample.info[0][k])
        np.testing.assert_allclose(second_step.obs["obs"], recv_sample.obs[1]["obs"])
        for k in second_info.keys():
            np.testing.assert_allclose(second_info[k], recv_sample.info[1][k])

    def test_worker_truncation(self):
        ag = make_test_agent(self.inference_client,
                             self.sample_producer,
                             sample_steps=10,
                             bootstrap_steps=0,
                             send_full_trajectory=False,
                             send_concise_info=False,
                             pad_trajectory=False)
        for i in range(4):
            ag.observe(make_step_result())
            np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        ag.observe(make_step_result(), truncate=True)
        self.assertEqual(ag.state, AgentState.WAITING_FOR_ACTION)
        self._inference()
        self.assertTrue(ag.ready_to_reset())
        ag.consume_inference_result()
        np.testing.assert_array_equal(ag.get_action().x, np.array([39], dtype=np.int32))
        self._uninference()
        self.assertEqual(ag.state, AgentState.READY_FOR_OBSERVATION)

        for i in range(6):
            ag.observe(make_step_result())
            np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))

        self.sample_producer.post.assert_called_once()
        recv_sample = self.sample_producer.post.call_args_list[0][0][0]
        self.assertEqual(recv_sample.length(dim=0), 10)
        self.assertFalse(recv_sample[0:4].truncated.any())
        self.assertFalse(recv_sample[5:].truncated.any())
        self.assertTrue(recv_sample[4].truncated.all())

        self.assertTrue(recv_sample[0].on_reset.all())
        self.assertFalse(recv_sample[1:5].on_reset.any())
        self.assertFalse(recv_sample[6:].on_reset.any())
        self.assertTrue(recv_sample[5].on_reset.all())

    def test_env_truncation(self):
        ag = make_test_agent(self.inference_client,
                             self.sample_producer,
                             sample_steps=10,
                             bootstrap_steps=0,
                             send_full_trajectory=False,
                             send_concise_info=False,
                             pad_trajectory=False)
        for i in range(4):
            ag.observe(make_step_result())
            np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        ag.observe(make_step_result(truncate=True))
        self.assertEqual(ag.state, AgentState.WAITING_FOR_ACTION)
        self._inference()
        self.assertTrue(ag.ready_to_reset())
        ag.consume_inference_result()
        np.testing.assert_array_equal(ag.get_action().x, np.array([39], dtype=np.int32))
        self._uninference()
        self.assertEqual(ag.state, AgentState.READY_FOR_OBSERVATION)

        for i in range(6):
            ag.observe(make_step_result())
            np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))
        self.sample_producer.post.assert_called_once()
        recv_sample = self.sample_producer.post.call_args_list[0][0][0]
        self.assertEqual(recv_sample.length(dim=0), 10)
        self.assertFalse(recv_sample[0:4].truncated.any())
        self.assertFalse(recv_sample[5:].truncated.any())
        self.assertTrue(recv_sample[4].truncated.all())

        self.assertTrue(recv_sample[0].on_reset.all())
        self.assertFalse(recv_sample[1:5].on_reset.any())
        self.assertFalse(recv_sample[6:].on_reset.any())
        self.assertTrue(recv_sample[5].on_reset.all())

    def test_env_done(self):
        ag = make_test_agent(self.inference_client,
                             self.sample_producer,
                             sample_steps=10,
                             bootstrap_steps=0,
                             send_full_trajectory=False,
                             send_concise_info=False,
                             pad_trajectory=False)
        for i in range(4):
            ag.observe(make_step_result())
            np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))

        ag.observe(make_step_result(done=True))
        self.assertEqual(ag.state, AgentState.READY_FOR_OBSERVATION)

        for i in range(6):
            ag.observe(make_step_result())
            np.testing.assert_array_equal(self._get_agent_action(ag).x, np.array([39], dtype=np.int32))

        self.sample_producer.post.assert_called_once()
        recv_sample = self.sample_producer.post.call_args_list[0][0][0]
        self.assertEqual(recv_sample.length(dim=0), 10)
        self.assertFalse(recv_sample[0:4].done.any())
        self.assertFalse(recv_sample[5:].done.any())
        self.assertTrue(recv_sample[4].done.all())

        self.assertTrue(recv_sample[0].on_reset.all())
        self.assertFalse(recv_sample[1:5].on_reset.any())
        self.assertFalse(recv_sample[6:].on_reset.any())
        self.assertTrue(recv_sample[5].on_reset.all())
        self.assertTrue(recv_sample[4].done.all())


class EnvTargetTest(unittest.TestCase):

    def setUp(self) -> None:
        self.env = mock.MagicMock()
        self.env.reset = mock.MagicMock(return_value=[59, 23, 64])
        self.env.step = mock.MagicMock(return_value=[33, 44, 88])
        self.agents = [mock.MagicMock() for _ in range(3)]
        self.test_kwargs = dict(host="testhost",
                                experiment="test_exp",
                                trial="test_trial",
                                worker="worker",
                                worker_id=0)

    def make_test_target(self, env, **kwargs):
        target = _EnvTarget(env, **kwargs)
        target.init_monitor(DummyMonitor(None))
        return target

    def test_trivial(self):
        target = self.make_test_target(self.env, max_num_steps=10, agents=self.agents, curriculum=None)
        target.init_monitor(DummyMonitor(None))
        self.assertTrue(target.ready_to_reset())
        target.reset()
        for ag in self.agents:
            ag.observe.assert_called_once()
        self.agents[0].observe.assert_called_with(env_result=59, truncate=False)
        self.agents[1].observe.assert_called_with(env_result=23, truncate=False)
        self.agents[2].observe.assert_called_with(env_result=64, truncate=False)
        for ag in self.agents:
            ag.reset_mock()

        target.step()
        for ag in self.agents:
            ag.observe.assert_called_once()
        self.agents[0].observe.assert_called_with(env_result=33, truncate=False)
        self.agents[1].observe.assert_called_with(env_result=44, truncate=False)
        self.agents[2].observe.assert_called_with(env_result=88, truncate=False)
        for ag in self.agents:
            ag.reset_mock()

        target.reset()
        for ag in self.agents:
            ag.observe.assert_called_once()
        self.agents[0].observe.assert_called_with(env_result=59, truncate=False)
        self.agents[1].observe.assert_called_with(env_result=23, truncate=False)
        self.agents[2].observe.assert_called_with(env_result=64, truncate=False)

    def test_terminate(self):
        target = self.make_test_target(self.env, max_num_steps=10, agents=self.agents, curriculum=None)
        target.reset()
        for i in range(9):
            target.step()
            for ag in self.agents:
                arg = ag.observe.call_args_list[-1][1]
                self.assertTrue("truncate" not in arg or arg["truncate"] is False)
        target.step()
        for ag in self.agents:
            arg = ag.observe.call_args_list[-1][1]
            self.assertTrue("truncate" in arg and arg["truncate"] is True)


class ActorWorkerTest(unittest.TestCase):

    def inf_config(self, name):
        self.inf_servers.append(
            NameResolvingInferenceServer(self.experiment_name, self.trial_name, name, 'pickle'))

        return config.InferenceStream(
            type_=config.InferenceStream.Type.NAME,
            stream_name=name,
        )

    def sample_config(self, name):
        self.sample_consumers.append(NameResolvingSampleConsumer(self.experiment_name, self.trial_name, name))
        return config.SampleStream(
            type_=config.SampleStream.Type.NAME,
            stream_name=name,
        )

    def setUp(self):
        import os
        os.environ["WANDB_MODE"] = "disabled"
        distributed.base.name_resolve.reconfigure("memory")
        self.experiment_name = "test_exp"
        self.trial_name = "test_run"
        self.policy_name = "test_policy"
        socket.gethostbyname = mock.MagicMock(return_value="127.0.0.1")
        NameResolvingInferenceClient.get_constant = mock.MagicMock(return_value=np.zeros(10))

        self.inf_servers = []
        self.sample_consumers = []
        self.inference_streams = [self.inf_config(f"{i}_{self.policy_name}") for i in range(3)]
        self.sample_producers = [self.sample_config(f"{i}_{self.policy_name}") for i in range(3)]
        import api.testing

    def make_config(self, agent_count, ring_size=1, env_name="aerochess", max_steps=0, splits=1):
        return config.ActorWorker(env=config.Environment(type_=env_name,
                                                         args={
                                                             "n": agent_count,
                                                             "max_steps": max_steps,
                                                         }),
                                  inference_streams=self.inference_streams,
                                  sample_streams=self.sample_producers,
                                  agent_specs=[
                                      config.AgentSpec(
                                          index_regex="[0-1]",
                                          inference_stream_idx=0,
                                          sample_stream_idx=0,
                                      ),
                                      config.AgentSpec(
                                          index_regex="[2-3]",
                                          inference_stream_idx=1,
                                          sample_stream_idx=1,
                                      ),
                                      config.AgentSpec(
                                          index_regex="[4-5]",
                                          inference_stream_idx=2,
                                          sample_stream_idx=2,
                                      )
                                  ],
                                  ring_size=ring_size,
                                  inference_splits=splits,
                                  worker_info=config.WorkerInformation(experiment_name=self.experiment_name,
                                                                       trial_name=self.trial_name,
                                                                       worker_type="actor",
                                                                       worker_index=0))

    def test_trivial_config(self):
        cfg = self.make_config(agent_count=6)
        aw = ActorWorker()
        aw.configure(cfg)

    @patch("distributed.system.actor_worker.Agent")
    def test_agent_matching(self, agent_mock):
        cfg = self.make_config(agent_count=6)
        aw = ActorWorker()
        aw._make_stream_clients = mock.MagicMock(side_effect=[((1, 2, 3), (4, 5, 6))])
        aw.configure(cfg)
        self.assertEqual(len(agent_mock.call_args_list), 6)
        self.assertEqual(agent_mock.call_args_list[0][1]["inference_client"], 1)
        self.assertEqual(agent_mock.call_args_list[0][1]["sample_producer"], 4)
        self.assertEqual(agent_mock.call_args_list[1][1]["inference_client"], 1)
        self.assertEqual(agent_mock.call_args_list[1][1]["sample_producer"], 4)
        self.assertEqual(agent_mock.call_args_list[2][1]["inference_client"], 2)
        self.assertEqual(agent_mock.call_args_list[2][1]["sample_producer"], 5)
        self.assertEqual(agent_mock.call_args_list[3][1]["inference_client"], 2)
        self.assertEqual(agent_mock.call_args_list[3][1]["sample_producer"], 5)
        self.assertEqual(agent_mock.call_args_list[4][1]["inference_client"], 3)
        self.assertEqual(agent_mock.call_args_list[4][1]["sample_producer"], 6)
        self.assertEqual(agent_mock.call_args_list[5][1]["inference_client"], 3)
        self.assertEqual(agent_mock.call_args_list[5][1]["sample_producer"], 6)


class SMACIndividualEnvTest(unittest.TestCase):
    """SMAC individual agents. Some agent may die while others are alive."""

    def make_test_target(self, env, **kwargs):
        target = _EnvTarget(env, **kwargs)
        target.init_monitor(DummyMonitor(None))
        return target

    def setUp(self):
        self.cnt = self.ep_cnt = 0
        self.env = mock.MagicMock()
        self.env.reset = mock.MagicMock(return_value=[make_step_result() for _ in range(3)])
        self.env.step = mock.MagicMock(return_value=[
            make_step_result(done=(self.cnt > 10)),
            make_step_result(done=(self.cnt > 20)),
            make_step_result(truncate=(self.cnt > 30))
        ],)
        distributed.base.name_resolve.reconfigure("memory")
        self.inference_clients = []
        self.sample_producers = []
        for i in range(3):
            inference_client = mock.MagicMock()
            inference_client.default_policy_state = np.random.randn(10)
            inference_client.consume_result = mock.MagicMock(return_value=[
                RolloutResult(action=AerochessAction(np.array([i], dtype=np.int32)),
                              policy_state=np.zeros(10))
            ])
            self.inference_clients.append(inference_client)
            self.sample_producers.append(mock.MagicMock())
        self._uninference()
        self.agents = [
            make_test_agent(
                ic,
                sp,
                sample_steps=30,
                bootstrap_steps=10,
            ) for (ic, sp) in zip(self.inference_clients, self.sample_producers)
        ]
        test_kwargs = dict(host="testhost",
                           experiment="test_exp",
                           trial="test_trial",
                           worker="worker",
                           worker_id=0)
        self.env_target = self.make_test_target(
            self.env,
            max_num_steps=10000,
            curriculum=None,
            agents=self.agents,
        )

    def _inference(self):
        for inference_client in self.inference_clients:
            inference_client.is_ready = mock.MagicMock(return_value=True)

    def _uninference(self):
        for inference_client in self.inference_clients:
            inference_client.is_ready = mock.MagicMock(return_value=False)

    def test_main(self):
        self.assertTrue(self.env_target.ready_to_reset())
        self.env_target.reset()
        self.assertFalse(self.env_target.ready_to_step())
        for _ in range(100):
            self.env.step = mock.MagicMock(return_value=[
                make_step_result(done=(self.cnt > 10)),
                make_step_result(done=(self.cnt > 20)),
                make_step_result(truncate=(self.cnt > 30))
            ],)
            self._inference()
            if self.env_target.ready_to_reset():
                self.env_target.reset()
                self.ep_cnt += 1
            elif self.env_target.ready_to_step():
                self.env_target.step()
                self.cnt = (self.cnt + 1) % 32
            self._uninference()
            self.assertFalse(self.env_target.ready_to_step())
        self.assertEqual(self.ep_cnt, 3)

        # reset + step count = 101
        # agent 0 calls 11 observe() before done
        self.assertEqual(self.inference_clients[0].post_request.call_count, 11 * self.ep_cnt + 101 - 3 * 32)
        # agent 1 calls 21 observe() before done
        self.assertEqual(self.inference_clients[1].post_request.call_count, 21 * self.ep_cnt + 101 - 3 * 32)
        # agent 2 always truncate, thus always issue requests
        self.assertEqual(self.inference_clients[2].post_request.call_count, 101)
        # consume_result calls is reduced by 1 because we have not consume the final inference result
        self.assertEqual(self.inference_clients[0].consume_result.call_count,
                         11 * self.ep_cnt + 101 - 3 * 32 - 1)
        self.assertEqual(self.inference_clients[1].consume_result.call_count,
                         21 * self.ep_cnt + 101 - 3 * 32 - 1)
        self.assertEqual(self.inference_clients[2].consume_result.call_count, 101 - 1)

        self.assertEqual(self.sample_producers[0].post.call_count, 1)
        sample = self.sample_producers[0].post.call_args_list[0][0][0]
        self.assertFalse(sample.truncated.any())
        self.assertTrue(sample.on_reset[0])
        self.assertTrue(sample.done[12])
        self.assertTrue(sample.on_reset[13])
        self.assertTrue(sample.done[25])
        self.assertTrue(sample.on_reset[26])
        self.assertTrue(sample.done[38])
        self.assertTrue(sample.on_reset[39])
        self.assertEqual(self.sample_producers[1].post.call_count, 2)
        sample = self.sample_producers[1].post.call_args_list[0][0][0]
        self.assertFalse(sample.truncated.any())
        self.assertTrue(sample.on_reset[0])
        self.assertTrue(sample.done[22])
        self.assertTrue(sample.on_reset[23])
        sample = self.sample_producers[1].post.call_args_list[1][0][0]
        self.assertFalse(sample.truncated.any())
        self.assertTrue(sample.done[45 - 30])
        self.assertTrue(sample.on_reset[46 - 30])
        self.assertEqual(self.sample_producers[2].post.call_count, 3)
        sample = self.sample_producers[2].post.call_args_list[0][0][0]
        self.assertFalse(sample.done.any())
        self.assertTrue(sample.on_reset[0])
        self.assertTrue(sample.truncated[32])
        self.assertTrue(sample.on_reset[33])
        np.testing.assert_array_equal(sample.action.x[32], np.array([2], dtype=np.int32))
        sample = self.sample_producers[2].post.call_args_list[1][0][0]
        self.assertFalse(sample.done.any())
        self.assertTrue(sample.truncated[65 - 30])
        self.assertTrue(sample.on_reset[66 - 30])
        np.testing.assert_array_equal(sample.action.x[65 - 30], np.array([2], dtype=np.int32))
        sample = self.sample_producers[2].post.call_args_list[2][0][0]
        self.assertFalse(sample.done.any())
        self.assertTrue(sample.truncated[98 - 30 - 30])
        self.assertTrue(sample.on_reset[99 - 30 - 30])
        np.testing.assert_array_equal(sample.action.x[98 - 30 - 30], np.array([2], dtype=np.int32))


class AsynchronousMultiAgentEnvTest(unittest.TestCase):
    """Real-time asynchronous environments."""

    def make_test_target(self, env, **kwargs):
        target = _EnvTarget(env, **kwargs)
        target.init_monitor(DummyMonitor(None))
        return target

    def setUp(self):
        self.env = mock.MagicMock()
        distributed.base.name_resolve.reconfigure("memory")
        self.inference_clients = []
        self.sample_producers = []
        for i in range(3):
            inference_client = mock.MagicMock()
            inference_client.default_policy_state = np.random.randn(10)
            inference_client.consume_result = mock.MagicMock(return_value=[
                RolloutResult(action=AerochessAction(np.array([i], dtype=np.int32)),
                              policy_state=np.zeros(10))
            ])
            self.inference_clients.append(inference_client)
            self.sample_producers.append(mock.MagicMock())
        self._uninference()
        self.sample_steps = 5
        self.bootstrap_steps = 3
        self.agents = [
            make_test_agent(ic,
                            sp,
                            sample_steps=self.sample_steps,
                            bootstrap_steps=self.bootstrap_steps,
                            index=i)
            for i, (ic, sp) in enumerate(zip(self.inference_clients, self.sample_producers))
        ]
        test_kwargs = dict(host="testhost",
                           experiment="test_exp",
                           trial="test_trial",
                           worker="worker",
                           worker_id=0)
        self.max_num_steps = 10
        self.env_target = self.make_test_target(self.env,
                                                max_num_steps=self.max_num_steps,
                                                curriculum=None,
                                                agents=self.agents)

    def _inference(self):
        for inference_client in self.inference_clients:
            inference_client.is_ready = mock.MagicMock(return_value=True)

    def _uninference(self):
        for inference_client in self.inference_clients:
            inference_client.is_ready = mock.MagicMock(return_value=False)

    def _mock_env_randomly(self):
        reset_mask = np.random.randint(0, 2, (3,))
        step_results = [make_step_result(done=False) if reset_mask[i] else None for i in range(3)]
        self.env.reset = mock.MagicMock(return_value=step_results)
        step_mask = np.random.randint(0, 2, (3,))
        random_done = np.random.randint(0, 2, (3,)).astype(np.uint8)
        step_results = [make_step_result(done=random_done[i]) if step_mask[i] else None for i in range(3)]
        self.env.step = mock.MagicMock(return_value=step_results)
        return reset_mask, step_mask, random_done

    def test_main(self):
        valid_cnt = np.zeros(3, dtype=np.int32)
        is_consecutive_done = np.zeros(3, dtype=np.int32)
        done_steps = [[] for _ in range(3)]
        truncate_steps = [[] for _ in range(3)]

        for _ in range(100):
            reset_mask, step_mask, random_done = self._mock_env_randomly()
            self._inference()

            ################## for debug ##################
            # doing_reset = self.env_target.ready_to_reset()
            # if doing_reset:
            #     s = f"reset? {doing_reset}, {reset_mask}"
            # else:
            #     s = f"reset? {doing_reset}, {step_mask} {random_done}"
            # print(s)
            ################## for debug ##################

            if self.env_target.ready_to_reset():
                self.env_target.reset()
                step_cnt = 0
                valid_cnt += reset_mask
                prev_done = [1 - bool(reset_mask[i]) for i in range(3)]
                is_consecutive_done *= 1 - reset_mask
            elif self.env_target.ready_to_step():
                step_cnt += 1
                truncate = step_cnt >= self.max_num_steps
                whether_store_this_step = step_mask
                for i in range(3):
                    if step_mask[i]:
                        if prev_done[i] and (random_done[i] or truncate):
                            whether_store_this_step[i] = 0
                            is_consecutive_done[i] = True
                        else:
                            is_consecutive_done[i] = False
                self.env_target.step()
                valid_cnt += whether_store_this_step
                # record done and truncate
                for i in range(3):
                    if whether_store_this_step[i] and random_done[i]:
                        done_steps[i].append(valid_cnt[i] - 1)
                    if not (random_done[i] * step_mask[i]) and truncate:
                        if whether_store_this_step[i]:
                            truncate_steps[i].append(valid_cnt[i] - 1)
                        else:
                            if not step_mask[i] and not prev_done[
                                    i]:  # this step is None, then the previous step is truncate
                                truncate_steps[i].append(valid_cnt[i] - 1)
                            else:  # consecutive finishes
                                pass
                # update prev done
                for i in range(3):
                    if step_mask[i]:
                        prev_done[i] = (random_done[i] or truncate)
            self._uninference()

        # print(valid_cnt)
        for i in range(3):
            cnt = valid_cnt[i]
            self.assertEqual(self.inference_clients[i].post_request.call_count, cnt - len(done_steps[i]))
            # print(i, self.sample_producers[i].post.call_count, cnt // self.sample_steps,
            #       is_consecutive_done[i])
            if not is_consecutive_done[
                    i]:  # if consecutive done, the final cache step will be appended to memory
                cnt -= 1  # otherwise there is a cahce step that has not been put into memory
            if (cnt % self.sample_steps >= self.bootstrap_steps):
                self.assertEqual(self.sample_producers[i].post.call_count, max(cnt // self.sample_steps, 0))
            else:
                self.assertEqual(self.sample_producers[i].post.call_count,
                                 max(cnt // self.sample_steps - 1, 0))

            if self.sample_producers[i].post.call_count == 0:
                continue

            # check memory data
            memory = []
            for j in range(self.sample_producers[i].post.call_count):
                sample = self.sample_producers[i].post.call_args_list[j][0][0]
                memory.append(sample[:self.sample_steps])
            memory = recursive_aggregate(memory, lambda x: np.concatenate(x, axis=0))

            # check done
            done_indices = [0] + [s for s in done_steps[i] if s < memory.done.shape[0]
                                  ] + [memory.done.shape[0]]
            for k in range(len(done_indices[:-1])):
                idx1 = done_indices[k]
                idx2 = done_indices[k + 1]
                if idx1 != 0:
                    self.assertTrue(memory.done[idx1])
                    np.testing.assert_array_equal(memory.action.x[idx1], np.zeros(1, dtype=np.int32))
                self.assertTrue((memory.action.x[idx1 + 1:idx2] == i).all())
                self.assertFalse(memory.done[idx1 + 1:idx2].any())

            # check truncate
            truncate_indices = [0] + [s for s in truncate_steps[i] if s < memory.truncated.shape[0]
                                      ] + [memory.truncated.shape[0]]
            # print(">>>>>>>>>", truncate_steps[i], memory.truncated.squeeze(-1))
            for k in range(len(truncate_indices[:-1])):
                idx1 = truncate_indices[k]
                idx2 = truncate_indices[k + 1]
                if idx1 != 0:
                    self.assertTrue(memory.truncated[idx1])
                    np.testing.assert_array_equal(memory.action.x[idx1], np.array([i], dtype=np.int32))
                self.assertFalse(memory.truncated[idx1 + 1:idx2].any())


class TurnBasedCardGameTest(unittest.TestCase):
    """Turn-based card games, e.g. Hanabi."""

    def make_test_target(self, env, **kwargs):
        target = _EnvTarget(env, **kwargs)
        target.init_monitor(DummyMonitor(None))
        return target

    def setUp(self):
        self.head = 0

        self.sample_steps = 5
        self.bootstrap_steps = 3
        self.max_num_steps = 10
        self.num_agents = 3

        self.env = mock.MagicMock()
        distributed.base.name_resolve.reconfigure("memory")
        self.inference_clients = []
        self.sample_producers = []
        for i in range(self.num_agents):
            inference_client = mock.MagicMock()
            inference_client.default_policy_state = np.random.randn(10)
            inference_client.consume_result = mock.MagicMock(return_value=[
                RolloutResult(action=AerochessAction(np.array([i], dtype=np.int32)),
                              policy_state=np.zeros(10))
            ])
            self.inference_clients.append(inference_client)
            self.sample_producers.append(mock.MagicMock())
        self._uninference()

        self.agents = [
            make_test_agent(ic,
                            sp,
                            sample_steps=self.sample_steps,
                            bootstrap_steps=self.bootstrap_steps,
                            index=i)
            for i, (ic, sp) in enumerate(zip(self.inference_clients, self.sample_producers))
        ]
        self.test_kwargs = dict(host="testhost",
                                experiment="test_exp",
                                trial="test_trial",
                                worker="worker",
                                worker_id=0)

    def _inference(self):
        for inference_client in self.inference_clients:
            inference_client.is_ready = mock.MagicMock(return_value=True)

    def _uninference(self):
        for inference_client in self.inference_clients:
            inference_client.is_ready = mock.MagicMock(return_value=False)

    def _mock_env(self, always_not_done=True, step=None):
        done = False
        self.env.reset = mock.MagicMock(return_value=[
            make_step_result(done=False) if i == self.head else None for i in range(self.num_agents)
        ])
        if always_not_done:
            self.env.step = mock.MagicMock(return_value=[
                make_step_result(done=False) if i == self.head else None for i in range(self.num_agents)
            ])
        else:
            if step >= self.max_num_steps and np.random.random() < 0.25:
                done = True
                self.env.step = mock.MagicMock(
                    return_value=[make_step_result(done=True) for i in range(self.num_agents)])
            else:
                self.env.step = mock.MagicMock(return_value=[
                    make_step_result(done=False) if i == self.head else None for i in range(self.num_agents)
                ])
        return done

    def test_system_truncate(self):
        self.env_target = self.make_test_target(self.env,
                                                max_num_steps=self.max_num_steps,
                                                curriculum=None,
                                                agents=self.agents)
        self.head = 0
        truncate_steps = [[] for _ in range(self.num_agents)]
        for t in range(self.num_agents * (self.max_num_steps + 1)):
            agent_step_idx = t // self.num_agents
            self._mock_env()
            self._inference()
            if self.env_target.ready_to_reset():
                self.env_target.reset()
                step_count = 0
            elif self.env_target.ready_to_step():
                self.env_target.step()
                step_count += 1
                if self.max_num_steps - step_count < self.num_agents:
                    truncate_steps[self.head].append(agent_step_idx)
                truncate = (step_count >= self.max_num_steps)
            self._uninference()
            self.head = (self.head + 1) % self.num_agents

        for i in range(self.num_agents):
            self.assertEqual(self.inference_clients[i].post_request.call_count, self.max_num_steps + 1)
            if i == self.num_agents - 1:
                self.assertEqual(self.inference_clients[i].consume_result.call_count, self.max_num_steps)
            else:
                self.assertEqual(self.inference_clients[i].consume_result.call_count, self.max_num_steps + 1)

            memory_len = self.max_num_steps + 1 if i != self.num_agents - 1 else self.max_num_steps

            if (memory_len % self.sample_steps >= self.bootstrap_steps):
                self.assertEqual(self.sample_producers[i].post.call_count, memory_len // self.sample_steps)
            else:
                self.assertEqual(self.sample_producers[i].post.call_count,
                                 memory_len // self.sample_steps - 1)

            # check memory data
            memory = []
            for j in range(self.sample_producers[i].post.call_count):
                sample = self.sample_producers[i].post.call_args_list[j][0][0]
                memory.append(sample[:self.sample_steps])
            memory = recursive_aggregate(memory, lambda x: np.concatenate(x, axis=0))

            # check done
            self.assertFalse(memory.done.any())

            # check truncate
            truncate_indices = [0] + [s for s in truncate_steps[i] if s < memory.truncated.shape[0]
                                      ] + [memory.truncated.shape[0]]
            # print(">>>>>>>>>", truncate_steps[i], memory.truncated.squeeze(-1))
            for k in range(len(truncate_indices[:-1])):
                idx1 = truncate_indices[k]
                idx2 = truncate_indices[k + 1]
                if idx1 != 0:
                    self.assertTrue(memory.truncated[idx1])
                    np.testing.assert_array_equal(memory.action.x[idx1], np.array([i], dtype=np.int32))
                self.assertFalse(memory.truncated[idx1 + 1:idx2].any())

    def test_done(self):
        self.env_target = self.make_test_target(self.env,
                                                max_num_steps=1000000000,
                                                curriculum=None,
                                                agents=self.agents)
        self.head = 0
        done_steps = [[] for _ in range(self.num_agents)]
        step_count = 0
        done_times = 0
        memory_len = np.zeros(self.num_agents)
        for t in range(self.num_agents * (self.max_num_steps + 1)):
            self._inference()
            if self.env_target.ready_to_reset():
                step_count = 0
                self._mock_env()
                self.env_target.reset()
                memory_len[self.head] += 1
                self.head = (self.head + 1) % self.num_agents
            elif self.env_target.ready_to_step():
                is_done = self._mock_env(always_not_done=False, step=step_count)
                self.env_target.step()
                step_count += 1
                if not is_done:
                    memory_len[self.head] += 1
                    self.head = (self.head + 1) % self.num_agents
                else:
                    memory_len += 1
                    done_times += 1
                    for j in range(self.num_agents):
                        agent_step_idx = (t + j + (done_times - 1) * self.num_agents) // self.num_agents
                        done_steps[(self.head + j) % self.num_agents].append(agent_step_idx)
            self._uninference()

        for i in range(self.num_agents):
            self.assertEqual(self.inference_clients[i].post_request.call_count, memory_len[i] - done_times)
            if i == (self.head + self.num_agents - 1) % self.num_agents:
                self.assertEqual(self.inference_clients[i].consume_result.call_count,
                                 memory_len[i] - done_times - 1)
            else:
                self.assertEqual(self.inference_clients[i].consume_result.call_count,
                                 memory_len[i] - done_times)

            mem_len = memory_len[i] - 1 if i == (self.head + self.num_agents -
                                                 1) % self.num_agents else memory_len[i]
            if (mem_len % self.sample_steps >= self.bootstrap_steps):
                self.assertEqual(self.sample_producers[i].post.call_count, mem_len // self.sample_steps)
            else:
                self.assertEqual(self.sample_producers[i].post.call_count, mem_len // self.sample_steps - 1)

            # check memory data
            memory = []
            for j in range(self.sample_producers[i].post.call_count):
                sample = self.sample_producers[i].post.call_args_list[j][0][0]
                memory.append(sample[:self.sample_steps])
            memory = recursive_aggregate(memory, lambda x: np.concatenate(x, axis=0))

            # check truncate
            self.assertFalse(memory.truncated.any())

            # check done
            done_indices = [0] + [s for s in done_steps[i] if s < memory.done.shape[0]
                                  ] + [memory.done.shape[0]]
            for k in range(len(done_indices[:-1])):
                idx1 = done_indices[k]
                idx2 = done_indices[k + 1]
                if idx1 != 0:
                    self.assertTrue(memory.done[idx1])
                    np.testing.assert_array_equal(memory.action.x[idx1], np.zeros(1, dtype=np.int32))
                self.assertTrue((memory.action.x[idx1 + 1:idx2] == i).all())
                self.assertFalse(memory.done[idx1 + 1:idx2].any())


if __name__ == '__main__':
    unittest.main()
