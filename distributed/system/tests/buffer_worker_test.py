import tempfile
import os
import mock
import unittest
import torch

from base.buffer import ReplayEntry
from base.testing import wait_network
from api.testing import random_policy
import api.config as config_pkg
import distributed.system.parameter_db as db
import distributed.system.buffer_worker


def make_config(policy, data_augmenter) -> config_pkg.BufferWorker:
    return config_pkg.BufferWorker(
        from_sample_stream="123",
        to_sample_stream="456",
        worker_info=config_pkg.WorkerInformation("test_exp", "test_run", "policy", 100),
        policy=policy,
        policy_name="testing",
        reanalyze_target="some_algo",
        policy_identifier="latest",
        parameter_db=config_pkg.ParameterDB(type_=config_pkg.ParameterDB.Type.FILESYSTEM,
                                            policy_name="testing"),
        unpack_batch_before_post=False,
        data_augmenter=data_augmenter)


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.__tmp = tempfile.TemporaryDirectory()
        db.PytorchFilesystemParameterDB.ROOT = os.path.join(self.__tmp.name, "checkpoints")
        self.__version = 0
        param_db = db.make_db(
            spec=config_pkg.ParameterDB(type_=config_pkg.ParameterDB.Type.FILESYSTEM),
            worker_info=config_pkg.WorkerInformation("test_exp", "test_run", "policy", 100),
        )
        param_db.push("testing", torch.randn(10, 10), version="100")

    @mock.patch("distributed.system.sample_stream")
    def test_trivial(self, sap_stm):
        bw = distributed.system.buffer_worker.BufferWorker()
        policy = config_pkg.Policy(type_="random_policy", args=dict(action_space=10))
        augmenter = config_pkg.DataAugmenter(type_="NULL", args=dict())
        cfg = make_config(policy, None)
        bw.configure(cfg)
        sap_stm.make_consumer.assert_called_once()
        self.assertTrue(sap_stm.make_consumer.call_args_list[0][0][0], "123")
        sap_stm.make_producer.assert_called_once()
        self.assertTrue(sap_stm.make_consumer.call_args_list[0][0][0], "456")

        bw1 = distributed.system.buffer_worker.BufferWorker()
        bw1.configure(make_config(policy, augmenter))

        bw2 = distributed.system.buffer_worker.BufferWorker()
        bw2.configure(make_config(None, augmenter))

    @mock.patch("api.policy")
    @mock.patch("base.buffer.Buffer")
    @mock.patch("api.environment.NullAugmenter")
    @mock.patch("distributed.system.sample_stream.SampleConsumer")
    @mock.patch("distributed.system.sample_stream.SampleProducer")
    def test_poll(self, pro, con, aug, buffer, pol):
        import api.policy
        pol.version = 0
        pol.device = "cpu"
        api.policy.make = mock.MagicMock(return_value=pol)
        pol.reanalyze = mock.MagicMock(return_value="32021")

        import distributed.system.sample_stream
        distributed.system.sample_stream.make_consumer = mock.MagicMock(return_value=con)
        distributed.system.sample_stream.make_producer = mock.MagicMock(return_value=pro)

        import base.buffer
        base.buffer.make_buffer = mock.MagicMock(return_value=buffer)

        augmenter = config_pkg.DataAugmenter(type_="NULL", args=dict())
        # environment.env_base.NullAugmenter = mock.MagicMock()
        bw = distributed.system.buffer_worker.BufferWorker()
        bw.configure(make_config("abc", augmenter))
        bw._poll()
        con.consume_to.assert_called_once()
        base.buffer.Buffer.empty.assert_called_once()
        base.buffer.Buffer.empty = mock.MagicMock(return_value=False)
        base.buffer.Buffer.get = mock.MagicMock(
            return_value=ReplayEntry(receive_time=0, reuses_left=1, sample="12023"))
        bw._poll()
        wait_network()
        base.buffer.Buffer.get.assert_called_once()
        pol.reanalyze.assert_called_once()
        self.assertEqual(pol.reanalyze.call_args_list[0][0][0], "12023")
        r = bw._poll()
        pro.post.assert_called_once_with("32021")
        self.assertEqual(r.sample_count, 5)
        self.assertEqual(r.batch_count, 1)


if __name__ == '__main__':
    unittest.main()
