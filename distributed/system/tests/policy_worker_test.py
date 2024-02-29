import threading
import unittest
import tempfile
import os
import mock
import numpy as np

from api.policy import RolloutRequest
from base.testing import wait_network
from api.environment import make as make_env
from distributed.system.inference_stream import make_client, IpInferenceClient
from distributed.system.policy_worker import PolicyWorker
from distributed.system.worker_base import MappingThread
from distributed.system.parameter_db import PytorchFilesystemParameterDB as FilesystemParameterDB
import api.testing.aerochess_env, api.testing.random_policy
import api.config as config
import distributed.base.name_resolve as name_resolve
import distributed.system.parameter_db


class PolicyWorkerTest(unittest.TestCase):

    def setUp(self):
        IpInferenceClient._shake_hand = mock.Mock()

        self.__tmp = tempfile.TemporaryDirectory()
        FilesystemParameterDB.ROOT = os.path.join(self.__tmp.name, "checkpoints")

        os.environ["WANDB_MODE"] = "disabled"
        name_resolve.reconfigure("memory")
        name_resolve.clear_subtree("inference_stream")
        self.policy_name = "test_policy"
        self.worker_info = config.WorkerInformation(experiment_name="test_exp",
                                                    trial_name="test_run",
                                                    worker_index=0)
        self.inf_stream_spec = config.InferenceStream(type_=config.InferenceStream.Type.NAME,
                                                      stream_name=self.policy_name)
        self.param_db_spec = config.ParameterDB(type_=config.ParameterDB.Type.FILESYSTEM,
                                                policy_name=self.policy_name)
        MappingThread.start = mock.MagicMock()
        MappingThread.stop = mock.MagicMock()
        threading.Thread.join = mock.MagicMock()
        threading.Thread.is_alive = mock.MagicMock(return_value=True)

    def tearDown(self):
        name_resolve.clear_subtree("inference_stream")

    def make_worker(self):
        return PolicyWorker()

    def make_worker_spec(self, batch_size, max_inference_delay=0.2, init_dir=None, foreign_policy=None):
        return config.PolicyWorker(policy_name=self.policy_name,
                                   inference_stream=self.inf_stream_spec,
                                   parameter_db=self.param_db_spec,
                                   policy=config.Policy(type_="random_policy",
                                                        args={"action_space": 5},
                                                        init_ckpt_dir=init_dir),
                                   worker_info=self.worker_info,
                                   batch_size=batch_size,
                                   max_inference_delay=max_inference_delay,
                                   pull_frequency_seconds=10,
                                   pull_max_failures=1,
                                   foreign_policy=foreign_policy)

    def test_atari_policy_worker_inference(self):
        atari_pw = self.make_worker()
        atari_pw.configure(self.make_worker_spec(batch_size=1))
        r = atari_pw._poll()
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.batch_count, 0)
        client = make_client(self.inf_stream_spec, self.worker_info)
        envs = [make_env("aerochess") for _ in range(2)]

        # reset case: request > batch_size
        req_ids = []
        for env in envs:
            req_ids.append(
                client.post_request(
                    RolloutRequest(obs=env.reset()[0].obs,
                                   policy_state=np.random.random(10),
                                   is_evaluation=np.array([False]))))
        client.flush()
        wait_network()
        atari_pw._pull_parameter_step()
        r = atari_pw._poll()  # Received requests. Will put requests for inference immediately as batch_size=1
        self.assertEqual(r.sample_count, 1)
        self.assertEqual(r.batch_count, 1)  # Put one in pipeline
        atari_pw._inference_thread._run_step()
        r = atari_pw._poll()  # Received requests and send response.
        wait_network()
        client.poll_responses()
        self.assertTrue(client.is_ready([req_ids[0]]))
        self.assertEqual(len(client.consume_result([req_ids[0]])), 1)
        self.assertEqual(r.sample_count, 1)
        self.assertEqual(r.batch_count, 1)  # Put one in pipeline
        atari_pw._inference_thread._run_step()
        atari_pw._poll()
        wait_network()
        client.poll_responses()
        self.assertTrue(client.is_ready([req_ids[1]]))
        self.assertEqual(len(client.consume_result([req_ids[1]])), 1)
        atari_pw.exit()

    def test_atari_policy_worker_batching(self):
        # reset case: request < batch_size
        atari_pw = self.make_worker()
        atari_pw.configure(self.make_worker_spec(batch_size=20))
        atari_pw._pull_parameter_step()
        r = atari_pw._poll()
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.batch_count, 0)
        client = make_client(self.inf_stream_spec, self.worker_info)
        envs = [make_env("aerochess") for _ in range(20)]

        req_ids = []
        for env in envs:
            req_ids.append(
                client.post_request(
                    RolloutRequest(obs=env.reset()[0].obs,
                                   policy_state=np.random.random(10),
                                   is_evaluation=np.array([False]))))
        client.flush()
        wait_network()
        r = atari_pw._poll()  # will inference all
        self.assertEqual(r.sample_count, 20)
        self.assertEqual(r.batch_count, 1)
        atari_pw._inference_thread._run_step()
        atari_pw._poll()  # send to client
        wait_network()
        client.poll_responses()
        self.assertTrue(client.is_ready(req_ids))
        self.assertEqual(len(client.consume_result(req_ids)), 20)

        # getting and setting state_dict
        atari_pw._poll()
        atari_pw.load_checkpoint(123)
        atari_pw.exit()

    def test_atari_policy_worker_batch_size(self):
        envs = [make_env("aerochess") for _ in range(20)]
        atari_pw = self.make_worker()
        atari_pw.configure(self.make_worker_spec(10, max_inference_delay=100))

        client = make_client(self.inf_stream_spec, self.worker_info)
        # Case1: request < batch_size
        req_ids = []
        for env in envs:
            req_ids.append(
                client.post_request(
                    RolloutRequest(obs=env.reset()[0].obs,
                                   policy_state=np.random.random(10),
                                   is_evaluation=np.array([False]))))
        client.flush()
        wait_network()
        atari_pw._pull_parameter_step()
        r = atari_pw._poll()  # Puts 10 samples for inference as queue is empty
        self.assertEqual(r.sample_count, 10)
        self.assertEqual(r.batch_count, 1)
        self.assertRaises(KeyError, client.consume_result, req_ids[:10])
        atari_pw._inference_thread._run_step()
        atari_pw._poll()
        wait_network(0.1)
        client.poll_responses()
        self.assertTrue(client.is_ready(req_ids[:10]), msg=str(client._response_cache) + " " + str(req_ids))
        self.assertEqual(len(client.consume_result(req_ids[:10])), 10)
        atari_pw._pull_parameter_step()
        r = atari_pw._poll()  # Puts 2 samples for inference as queue is empty
        atari_pw._inference_thread._run_step()
        atari_pw._poll()
        wait_network()
        client.poll_responses()
        self.assertTrue(client.is_ready(req_ids[10:]))
        self.assertEqual(len(client.consume_result(req_ids[10:])), 10)
        # Thread is not running.
        atari_pw.exit()

    def test_gpu_exception_catching(self):
        atari_pw = self.make_worker()
        atari_pw.configure(self.make_worker_spec(10))

        threading.Thread.is_alive = mock.MagicMock(return_value=False)
        self.assertRaises(RuntimeError, atari_pw._poll)
        threading.Thread.is_alive = mock.MagicMock(return_value=True)

    def test_initial_param_pull(self):
        FilesystemParameterDB.get = mock.MagicMock()

        atari_pw = self.make_worker()
        atari_pw.configure(self.make_worker_spec(10))
        atari_pw._poll()
        FilesystemParameterDB.get.assert_called_once()

    def test_absolute_path(self):
        FilesystemParameterDB.get = mock.MagicMock()
        FilesystemParameterDB.get_file = mock.MagicMock()
        import torch  # Remove when we do not depend on torch.
        torch.load = mock.MagicMock()
        atari_pw = self.make_worker()
        atari_pw.configure(self.make_worker_spec(10,
                                                 foreign_policy=config.ForeignPolicy(absolute_path="foo")))
        atari_pw._poll()
        FilesystemParameterDB.get.assert_not_called()
        FilesystemParameterDB.get_file.assert_called_once_with("foo")

    def test_make_foreign_db(self):
        distributed.system.parameter_db.make_db = mock.MagicMock()
        atari_pw = self.make_worker()
        atari_pw.configure(
            self.make_worker_spec(10,
                                  foreign_policy=config.ForeignPolicy(
                                      foreign_experiment_name="foo",
                                      foreign_trial_name="bar",
                                      param_db=config.ParameterDB(type_=config.ParameterDB.Type.FILESYSTEM))))
        wi = distributed.system.parameter_db.make_db.call_args_list[0][1]["worker_info"]
        self.assertEqual(wi.experiment_name, "foo")
        self.assertEqual(wi.trial_name, "bar")

    def test_foreign_policy(self):
        FilesystemParameterDB.get = mock.MagicMock()
        import torch  # Remove when we do not depend on torch.
        torch.load = mock.MagicMock()
        atari_pw = self.make_worker()
        atari_pw.configure(
            self.make_worker_spec(10,
                                  foreign_policy=config.ForeignPolicy(
                                      foreign_experiment_name="foo",
                                      foreign_trial_name="bar",
                                      foreign_policy_name="q",
                                      foreign_policy_identifier="s",
                                      param_db=config.ParameterDB(type_=config.ParameterDB.Type.FILESYSTEM))))
        atari_pw._poll()
        args = FilesystemParameterDB.get.call_args_list[0][1]
        self.assertEqual(args["name"], "q")
        self.assertEqual(args["identifier"], "s")


if __name__ == '__main__':
    unittest.main()
