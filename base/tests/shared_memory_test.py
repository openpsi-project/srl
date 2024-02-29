import dataclasses
import datetime
import time
import unittest
import numpy as np
import itertools

from base.testing import *
from base.shared_memory import *
from base.namedarray import NamedArray, recursive_aggregate, recursive_apply, from_dict


class SampleBatch(NamedArray):

    def __init__(self,
                 obs: NamedArray,
                 on_reset: np.ndarray = None,
                 done: np.ndarray = None,
                 truncated: np.ndarray = None,
                 action: NamedArray = None,
                 reward: np.ndarray = None,
                 info: NamedArray = None,
                 info_mask: np.ndarray = None,
                 policy_state: NamedArray = None,
                 policy_name: np.ndarray = None,
                 policy_version_steps: np.ndarray = None,
                 **kwargs):
        super(SampleBatch, self).__init__(
            obs=obs,
            on_reset=on_reset,
            done=done,
            truncated=truncated,
            action=action,
            reward=reward,
            info=info,
            info_mask=info_mask,
            policy_state=policy_state,
            policy_name=policy_name,
            policy_version_steps=policy_version_steps,
        )


def make_sample_batch(ep_len):
    policy_name_collection = ['default', 'opponent', 'test']
    sample = dict(
        obs=dict(x=np.random.randn(ep_len, 10), y=np.random.randn(ep_len, 5)),
        on_reset=np.random.randint(0, 2, (ep_len, 1)),
        done=np.random.randint(0, 2, (ep_len, 1)),
        action=dict(x=np.random.randint(0, 6, (ep_len, 1))),
        reward=np.random.randn(ep_len, 1),
        policy_state=dict(hx=np.random.randn(ep_len, 1, 64)),
        # policy_name=np.stack([np.array([np.random.choice(policy_name_collection)]) for _ in range(ep_len)]),
        policy_version_steps=np.zeros((ep_len, 1), dtype=np.int32),
    )
    return from_dict(sample)


experiment_name = "shared_memory_test"
trial_name = str(datetime.datetime.now()).replace(' ', '-')
stream_name = "default"
qsize = 10
total_sample_cnt = 100


class ServerTest(unittest.TestCase):

    def setUp(self):
        self.server = SharedMemoryDockServer(experiment_name, trial_name, stream_name, qsize)

    def test_write(self):
        assert qsize >= 4
        server = self.server
        with self.assertRaises(ValueError):
            server.release_write([-1])
        with self.assertRaises(RuntimeError):
            server.release_write([1])
        for i in range(qsize):
            self.assertEqual(server.acquire_write()[0], i)
        self.assertIsNone(server.acquire_write())
        server.release_write([0, 1])
        self.assertEqual(server.acquire_write(2), [0, 1])
        server.release_write(list(range(qsize)))
        with self.assertRaises(RuntimeError):
            server.release_write(1)

    def test_read_write(self):
        assert qsize == 10
        server = self.server
        self.assertIsNone(server.acquire_read())
        for i in range(qsize):
            self.assertEqual(server.acquire_write()[0], i)
        self.assertIsNone(server.acquire_read())
        server.release_write([0, qsize // 2])
        self.assertEqual(sorted(server.acquire_read(2)), [0, qsize // 2])

        server.release_write([i for i in range(qsize) if i != 0 and i != qsize // 2])
        self.assertIsNone(server.acquire_write(allow_overwrite=False))
        self.assertIsNone(server.acquire_read(qsize - 1))
        server.acquire_read(qsize - 2)
        self.assertIsNone(server.acquire_read())

        server.release_read([0])
        self.assertEqual(server.acquire_write()[0], 0)
        server.release_write([0])
        server.release_read(list(range(1, qsize)))

        np.testing.assert_array_equal(server._is_writable, np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        np.testing.assert_array_equal(server._is_readable, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    def test_reuse(self):
        assert qsize == 10
        server = SharedMemoryDockServer(experiment_name, trial_name, stream_name, qsize, reuses=2)
        server.acquire_write(qsize)
        for i in range(qsize // 2):
            server.release_write([i])

        self.assertEqual(server.acquire_read(1, preference='old')[0], 0)
        self.assertEqual(server.acquire_read(1, preference='fresh')[0], qsize // 2 - 1)

        server.release_read([0, qsize // 2 - 1])
        self.assertEqual(sorted(server.acquire_read(qsize // 2 - 2, preference='more_reuses_left')),
                         [i for i in range(1, qsize // 2 - 1)])

        for i in range(qsize // 2, qsize):
            server.release_write([i])

        self.assertEqual(sorted(server.acquire_read(qsize // 2, preference="fresh")),
                         [i for i in range(qsize // 2, qsize)])
        np.testing.assert_array_equal(server._is_writable, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        np.testing.assert_array_equal(server._is_readable, np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
        np.testing.assert_array_equal(server._is_being_read, np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 1]))


if __name__ == '__main__':
    unittest.main()
