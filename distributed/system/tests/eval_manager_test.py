import os
import socket
import tempfile
import mock
import numpy as np
import unittest

from base import name_resolve
from base.namedarray import NamedArray
from base.testing import wait_network, get_test_param
from distributed.base.monitoring import DummyMonitor
from distributed.system import parameter_db
from distributed.base import name_resolve as distributed_name_resolve
from distributed.system.inference_stream import IpInferenceClient
import api.config as config
import api.trainer
import distributed.system.eval_manager
import distributed.system.sample_stream


class TestEpisodeInfo(NamedArray):

    def __init__(
            self,
            hp: np.ndarray = np.array([0], dtype=np.float32),
            mana: np.ndarray = np.array([0], dtype=np.float32),
    ):
        super(TestEpisodeInfo, self).__init__(hp=hp, mana=mana)


def make_config(policy_name="test_policy",
                eval_stream_name="eval_test_policy",
                worker_index=0,
                worker_count=1):
    return config.EvaluationManager(
        eval_sample_stream=config.SampleStream(config.SampleStream.Type.NAME, stream_name=eval_stream_name),
        parameter_db=config.ParameterDB(config.ParameterDB.Type.METADATA, policy_name=policy_name),
        policy_name=policy_name,
        eval_tag="evaluation",
        eval_games_per_version=5,
        worker_info=config.WorkerInformation("test_exp", "test_run", "trainer", worker_index, worker_count),
    )


def random_sample_batch(version=0, hp=0, mana=0, policy_name="test_policy"):
    return api.trainer.SampleBatch(obs=np.random.random(size=(10, 10)),
                                   reward=np.random.random(size=(10, 1)),
                                   policy_version_steps=np.full(shape=(10, 1), fill_value=version),
                                   info=TestEpisodeInfo(hp=np.full(shape=(10, 1), fill_value=hp),
                                                        mana=np.full(shape=(10, 1), fill_value=mana)),
                                   info_mask=np.concatenate([np.zeros(
                                       (9, 1)), np.ones((1, 1))], axis=0),
                                   policy_name=np.full(shape=(10, 1), fill_value=policy_name))


def make_test_producer(policy_name="test_policy", rank=0):
    producer = distributed.system.sample_stream.make_producer(
        config.SampleStream(config.SampleStream.Type.NAME, stream_name=policy_name),
        worker_info=config.WorkerInformation("test_exp", "test_run", "policy", rank, 100),
    )
    producer.init_monitor(DummyMonitor(None))
    return producer


class TestEvalManager(unittest.TestCase):

    def setUp(self) -> None:
        IpInferenceClient._shake_hand = mock.Mock()
        self.__tmp = tempfile.TemporaryDirectory()
        parameter_db.PytorchFilesystemParameterDB.ROOT = os.path.join(self.__tmp.name, "checkpoints")

        os.environ["WANDB_MODE"] = "disabled"
        socket.gethostbyname = mock.MagicMock(return_value="127.0.0.1")
        name_resolve.reconfigure("memory", log_events=True)
        distributed_name_resolve.reconfigure("memory", log_events=True)

    def tearDown(self) -> None:
        db = parameter_db.make_db(config.ParameterDB(type_=config.ParameterDB.Type.METADATA,
                                                     policy_name="test_policy"),
                                  worker_info=config.WorkerInformation(
                                      experiment_name="test_exp",
                                      trial_name="test_run",
                                  ))
        try:
            db.clear("test_policy")
        except FileNotFoundError:
            pass

    def test_loginfo(self):
        test_parameter_db = parameter_db.make_db(config.ParameterDB(type_=config.ParameterDB.Type.FILESYSTEM,
                                                                    policy_name="test_policy"),
                                                 worker_info=config.WorkerInformation(
                                                     experiment_name="test_exp",
                                                     trial_name="test_run",
                                                 ))
        try:
            test_parameter_db.clear("test_policy")
        except FileNotFoundError:
            pass
        eval_manager = distributed.system.eval_manager.EvalManager()
        eval_manager.configure(make_config("test_policy", "eval", "metadata"))
        eval_manager.eval_stream_init_monitor()
        producer = make_test_producer(policy_name="eval")
        wait_network()
        r = eval_manager._poll()
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.batch_count, 0)

        for _ in range(5):
            producer.post(random_sample_batch(version=0))
        producer.flush()
        wait_network()
        # Eval manager does not accept sample until the first version is pushed.
        for _ in range(5):
            r = eval_manager._poll()
            self.assertEqual(r.sample_count, 1)
            self.assertEqual(r.batch_count, 0)
        r = eval_manager._poll()
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.batch_count, 0)

        test_parameter_db.push("test_policy", get_test_param(version=1), version="1")
        for _ in range(5):
            producer.post(random_sample_batch(version=0))
        producer.flush()
        wait_network()
        for _ in range(5):
            r = eval_manager._poll()
            self.assertEqual(r.sample_count, 1)
            self.assertEqual(r.batch_count, 0)
        r = eval_manager._poll()
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.batch_count, 0)

        test_parameter_db.push("test_policy", get_test_param(20), version="20")
        for _ in range(5):
            producer.post(random_sample_batch(version=1))
        producer.flush()
        wait_network()
        for _ in range(4):
            r = eval_manager._poll()
            self.assertEqual(r.sample_count, 1)
            self.assertEqual(r.batch_count, 0)
        r = eval_manager._poll()
        self.assertEqual(r.sample_count, 1)
        self.assertEqual(r.batch_count, 1)
        r = eval_manager._poll()
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.batch_count, 0)

        # Evaluation manager loads to version 20. 10 episodes will be logged.
        for _ in range(10):
            producer.post(random_sample_batch(version=20))
        producer.flush()
        wait_network()
        for __ in range(2):
            for _ in range(4):
                r = eval_manager._poll()
                self.assertEqual(r.sample_count, 1)
                self.assertEqual(r.batch_count, 0)
            r = eval_manager._poll()
            self.assertEqual(r.sample_count, 1)
            self.assertEqual(r.batch_count, 1)

        test_parameter_db.push("test_policy", get_test_param(50), version="50")
        r = eval_manager._poll()
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.batch_count, 0)

    def test_update_metadata(self):
        test_parameter_db = parameter_db.make_db(config.ParameterDB(type_=config.ParameterDB.Type.METADATA,
                                                                    policy_name="test_policy"),
                                                 worker_info=config.WorkerInformation(
                                                     experiment_name="test_exp",
                                                     trial_name="test_run",
                                                 ))
        try:
            test_parameter_db.clear("test_policy")
        except FileNotFoundError:
            pass
        eval_manager = distributed.system.eval_manager.EvalManager()
        eval_manager.configure(make_config("test_policy", "eval", "metadata"))
        eval_manager.eval_stream_init_monitor()
        producer = make_test_producer(policy_name="eval")
        wait_network()
        r = eval_manager._poll()
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.batch_count, 0)

        # If parameter does not exists in file system, Nothing will be updated.
        producer.post(random_sample_batch(version=0))
        producer.flush()
        wait_network()
        r = eval_manager._poll()
        self.assertEqual(r.sample_count, 1)
        self.assertEqual(r.batch_count, 0)

        # Update first version.
        test_parameter_db.push("test_policy", get_test_param(version=0), version="0", tags="temp_tag")
        for _ in range(5):
            producer.post(random_sample_batch(version=0, hp=100, mana=20))
        producer.flush()
        wait_network()
        # Eval manager does not accept sample until the first version is pushed.
        for _ in range(5):
            r = eval_manager._poll()
            self.assertEqual(r.sample_count, 1)
            self.assertEqual(r.batch_count, 1)
        self.assertDictEqual(test_parameter_db._get_metadata("test_policy", str(0)), {
            "hp": 100.,
            "mana": 20.
        })

        # Update second version, where the first remain the same.
        test_parameter_db.push("test_policy", get_test_param(version=5), version="5", tags="temp_tag")
        for _ in range(3):
            producer.post(random_sample_batch(version=5, hp=200, mana=40))
        producer.flush()
        wait_network()
        for _ in range(3):
            r = eval_manager._poll()
            self.assertEqual(r.sample_count, 1)
            self.assertEqual(r.batch_count, 1)
        self.assertDictEqual(test_parameter_db._get_metadata("test_policy", str(0)), {
            "hp": 100.,
            "mana": 20.
        })
        self.assertDictEqual(test_parameter_db._get_metadata("test_policy", str(5)), {
            "hp": 200.,
            "mana": 40.
        })

        # Update the first version again.

        for _ in range(1):
            producer.post(random_sample_batch(version=0, hp=40, mana=80))
        producer.flush()
        wait_network()
        r = eval_manager._poll()
        self.assertEqual(r.sample_count, 1)
        self.assertEqual(r.batch_count, 1)
        self.assertDictEqual(test_parameter_db._get_metadata("test_policy", str(0)), {"hp": 90., "mana": 30.})

        # Update Both at the same time.
        producer.post(random_sample_batch(version=0, hp=20, mana=100))
        producer.post(random_sample_batch(version=5, hp=0, mana=0))
        producer.flush()
        wait_network()
        for _ in range(2):
            r = eval_manager._poll()
            self.assertEqual(r.sample_count, 1)
            self.assertEqual(r.batch_count, 1)
        self.assertDictEqual(test_parameter_db._get_metadata("test_policy", str(0)), {"hp": 80., "mana": 40.})
        self.assertDictEqual(test_parameter_db._get_metadata("test_policy", str(5)), {
            "hp": 150.,
            "mana": 30.
        })


if __name__ == '__main__':
    unittest.main()
