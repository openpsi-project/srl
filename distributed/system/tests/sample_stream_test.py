import sys
import mock
import unittest
import queue
import time

from base.testing import get_testing_port, wait_network
from distributed.base.monitoring import DummyMonitor
from distributed.system.sample_stream import *
import api.config as config


def make_test_consumer(type_="ip",
                       port=-1,
                       experiment_name="test_exp",
                       trial_name="test_run",
                       policy_name="test_policy",
                       rank=0):
    if type_ == "ip":
        consumer = make_consumer(config.SampleStream(type_=config.SampleStream.Type.IP, address=f"*:{port}"))
    elif type_ == "name":
        consumer = make_consumer(
            config.SampleStream(type_=config.SampleStream.Type.NAME, stream_name=policy_name),
            config.WorkerInformation(experiment_name=experiment_name,
                                     trial_name=trial_name,
                                     worker_index=rank))
    elif type_ == "shared_memory":
        consumer = make_consumer(
            config.SampleStream(type_=config.SampleStream.Type.SHARED_MEMORY, stream_name=policy_name),
            config.WorkerInformation(experiment_name=experiment_name, trial_name=trial_name))
    if type_ != "ip":
        consumer.init_monitor(DummyMonitor(None))
    return consumer


def make_test_producer(type_="ip",
                       port=-1,
                       experiment_name="test_exp",
                       trial_name="test_run",
                       policy_name="test_policy",
                       rank=0):
    if type_ == "ip":
        producer = make_producer(
            config.SampleStream(type_=config.SampleStream.Type.IP, address=f"localhost:{port}"))
    elif type_ == "name":
        producer = make_producer(
            config.SampleStream(type_=config.SampleStream.Type.NAME, stream_name=policy_name),
            config.WorkerInformation(experiment_name=experiment_name,
                                     trial_name=trial_name,
                                     worker_index=rank))
    elif type_ == "round_robin":
        producer = make_producer(
            config.SampleStream(type_=config.SampleStream.Type.NAME_ROUND_ROBIN, stream_name=policy_name),
            config.WorkerInformation(experiment_name=experiment_name, trial_name=trial_name))
    elif type_ == "shared_memory":
        producer = make_producer(
            config.SampleStream(type_=config.SampleStream.Type.SHARED_MEMORY, stream_name=policy_name),
            config.WorkerInformation(experiment_name=experiment_name, trial_name=trial_name))
    if type_ != "ip":
        producer.init_monitor(DummyMonitor(None))
    return producer


def run_shared_memory_producer():
    pass


def run_shared_memory_consumer():
    pass


class SampleStreamTest(unittest.TestCase):

    def setUp(self):
        name_resolve.reconfigure("memory", log_events=True)
        name_resolve.clear_subtree("/")
        sys.modules["gfootball"] = mock.Mock()
        sys.modules["gfootball.env"] = mock.Mock()

    def tearDown(self):
        name_resolve.clear_subtree("/")

    def sample_batch(self, sample_steps, version=0):
        return recursive_aggregate([
            SampleBatch(
                obs=np.random.random((10,)).astype(np.float32),
                policy_state=np.random.random((2, 2)).astype(np.float32),
                on_reset=np.array([False], dtype=np.uint8),
                action=np.array([np.random.randint(19)]).astype(np.int32),
                log_probs=np.random.random(1,).astype(np.int32),
                reward=np.array([0], dtype=np.float32).astype(np.int32),
                info=np.random.randint(0, 2, (1,)),
                policy_version_steps=np.array([version], dtype=np.int64),
            ) for _ in range(sample_steps)
        ], np.stack)

    def test_simple(self):
        port = get_testing_port()
        consumer = make_test_consumer(port=port)
        producer = make_test_producer(port=port)
        buffer = queue.Queue()

        self.assertEqual(consumer.consume_to(buffer), 0)
        self.assertTrue(buffer.empty())

        producer.post(self.sample_batch(5))
        producer.flush()
        wait_network()
        self.assertEqual(consumer.consume_to(buffer), 1)
        self.assertFalse(buffer.empty())
        self.assertEqual(buffer.get().length(dim=0), 5)
        self.assertTrue(buffer.empty())

        producer.post(base.namedarray.from_dict(dict(a=np.array([5, 6, 7]))))
        producer.post(base.namedarray.from_dict(dict(a=np.array([3, 3, 3]))))
        producer.flush()
        wait_network()
        self.assertEqual(consumer.consume_to(buffer), 2)
        self.assertFalse(buffer.empty())
        np.testing.assert_equal(buffer.get().a, [5, 6, 7])
        np.testing.assert_equal(buffer.get().a, [3, 3, 3])
        self.assertTrue(buffer.empty())

    def test_multiple_producers(self):
        port = get_testing_port()
        consumer = make_test_consumer(port=port)
        producer1 = make_test_producer(port=port)
        producer2 = make_test_producer(port=port)
        buffer = queue.Queue()

        self.assertEqual(consumer.consume_to(buffer), 0)
        self.assertTrue(buffer.empty())

        producer1.post(self.sample_batch(5))
        producer2.post(self.sample_batch(6))
        producer1.flush()
        producer2.flush()
        wait_network()
        self.assertEqual(consumer.consume_to(buffer), 2)
        self.assertFalse(buffer.empty())
        self.assertSetEqual(
            {buffer.get().length(dim=0), buffer.get().length(dim=0)}, {5, 6})  # Order is not guaranteed.
        self.assertTrue(buffer.empty())

        producer1.post(self.sample_batch(5))
        producer1.post(self.sample_batch(5))
        producer2.post(self.sample_batch(5))
        producer1.post(self.sample_batch(5))
        producer1.flush()
        producer2.flush()
        time.sleep(0.01)
        self.assertEqual(consumer.consume_to(buffer), 4)
        self.assertFalse(buffer.empty())
        [buffer.get() for _ in range(4)]
        self.assertTrue(buffer.empty())

    def test_name_resolving_pair(self):
        consumer = make_test_consumer(type_="name")
        producer = make_test_producer(type_="name", rank=0)
        buffer = queue.Queue()

        self.assertEqual(consumer.consume_to(buffer), 0)
        self.assertTrue(buffer.empty())

        producer.post(self.sample_batch(5))
        producer.flush()
        wait_network()
        self.assertEqual(consumer.consume_to(buffer), 1)
        self.assertFalse(buffer.empty())
        self.assertEqual(buffer.get().length(0), 5)
        self.assertTrue(buffer.empty())

        producer.post(self.sample_batch(5))
        producer.post(self.sample_batch(5))
        producer.flush()
        wait_network()
        self.assertEqual(consumer.consume_to(buffer), 2)
        self.assertFalse(buffer.empty())
        buffer.get()
        buffer.get()
        self.assertTrue(buffer.empty())

    def test_name_resolving_multiple(self):
        consumer1 = make_test_consumer(type_="name")
        consumer2 = make_test_consumer(type_="name")
        if consumer1.address > consumer2.address:
            consumer1, consumer2 = consumer2, consumer1
        producer1 = make_test_producer(type_="name", rank=0)
        producer2 = make_test_producer(type_="name", rank=1)
        buffer = queue.Queue()

        self.assertEqual(consumer1.consume_to(buffer), 0)
        self.assertEqual(consumer2.consume_to(buffer), 0)
        self.assertTrue(buffer.empty())

        producer1.post(self.sample_batch(5))
        producer2.post(self.sample_batch(6))
        producer1.flush()
        producer2.flush()
        wait_network()
        self.assertEqual(consumer1.consume_to(buffer), 1)
        self.assertFalse(buffer.empty())
        self.assertEqual(buffer.get().length(dim=0), 5)
        self.assertEqual(consumer2.consume_to(buffer), 1)
        self.assertFalse(buffer.empty())
        self.assertEqual(buffer.get().length(dim=0), 6)

        self.assertTrue(buffer.empty())

        producer1.post(self.sample_batch(5))
        producer1.post(self.sample_batch(5))
        producer2.post(self.sample_batch(5))
        producer1.post(self.sample_batch(5))
        producer2.post(self.sample_batch(5))
        producer1.flush()
        producer2.flush()

        wait_network()
        self.assertEqual(consumer1.consume_to(buffer), 3)
        self.assertEqual(consumer2.consume_to(buffer), 2)
        self.assertFalse(buffer.empty())
        [buffer.get() for _ in range(5)]
        self.assertTrue(buffer.empty())

    def test_zip(self):
        consumer1 = make_test_consumer(type_="name", policy_name="alice")
        consumer2 = make_test_consumer(type_="name", policy_name="bob")
        producer1 = make_test_producer(type_="name", policy_name="alice")
        producer1.post = mock.MagicMock()
        producer2 = make_test_producer(type_="name", policy_name="bob")
        producer2.post = mock.MagicMock()
        zipped_producer = zip_producers([producer1, producer2])
        zipped_producer.post(self.sample_batch(5, version=2))
        zipped_producer.flush()
        producer1.post.assert_called_once()
        producer2.post.assert_called_once()

    def test_round_robin(self):
        buffer = queue.Queue()
        consumer1 = make_test_consumer(type_="name", policy_name="alice")
        consumer2 = make_test_consumer(type_="name", policy_name="alice")
        producer = make_test_producer(type_="round_robin", policy_name="alice")
        producer.post(self.sample_batch(5))
        producer.flush()
        wait_network()
        c11 = consumer1.consume_to(buffer)
        c21 = consumer2.consume_to(buffer)
        producer.post(self.sample_batch(5))
        producer.flush()
        wait_network()
        c12 = consumer1.consume_to(buffer)
        c22 = consumer2.consume_to(buffer)
        self.assertEqual(c11 + c21, 1)
        self.assertEqual(c11 + c12, 1)
        self.assertEqual(c21 + c22, 1)
        self.assertEqual(c22 + c12, 1)

    def test_consumer_consume(self):
        consumer1 = make_test_consumer(type_="name", policy_name="alice")
        producer = make_test_producer(type_="name", policy_name="alice")
        s = self.sample_batch(5)
        producer.post(s)
        producer.flush()
        wait_network()
        s1 = consumer1.consume()
        for key in s.keys():
            if s[key] is None:
                self.assertIsNone(s1[key])
            else:
                np.testing.assert_equal(s[key], s1[key])
        self.assertRaises(NothingToConsume, consumer1.consume)


if __name__ == '__main__':
    unittest.main()
