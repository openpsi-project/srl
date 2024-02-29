import dataclasses
import numpy as np
import time
import unittest

from base.buffer import (PrioritizedReplayBuffer, PriorityQueueBuffer, ReplayEntry, SimpleQueueBuffer)
from base.namedarray import NamedArray
import base.timeutil


@dataclasses.dataclass
class PriorityBufferConfig:
    max_size = 5
    reuses = 2


class SampleBatch(NamedArray):

    def __init__(self, obs: NamedArray, sampling_weight: np.ndarray = None, **kwargs):
        super(SampleBatch, self).__init__(obs=obs, **kwargs)
        self.register_metadata(sampling_weight=sampling_weight,)


class AnalyzedResult(NamedArray):

    def __init__(self, value: np.ndarray, ret: np.ndarray):
        super(AnalyzedResult, self).__init__(value=value, ret=ret)


class BufferTest(unittest.TestCase):

    def test_replay_compare(self):
        # Prefer newer ones.
        a = ReplayEntry(reuses_left=2, receive_time=time.time(), sample=[])
        time.sleep(0.01)
        b = ReplayEntry(reuses_left=2, receive_time=time.time(), sample=[])
        self.assertTrue(a < b)

        # Prefer more recent samples ones.
        a = ReplayEntry(reuses_left=1, receive_time=time.time(), sample=[])
        b = ReplayEntry(reuses_left=2, receive_time=time.time(), sample=[])
        self.assertTrue(b > a)

    def test_prioritized_queue_buffer(self):
        b = PriorityQueueBuffer(max_size=5, reuses=2, batch_size=1)
        self.assertEqual(b.empty(), True)

        # Test reuse.
        b.put([1, 2, 3])
        self.assertEqual(b.qsize(), 1)
        self.assertListEqual(b.get().sample.squeeze(-1).tolist(), [1, 2, 3])

        # Data should be put back into buffer.
        self.assertEqual(b.qsize(), 1)
        self.assertListEqual(b.get().sample.squeeze(-1).tolist(), [1, 2, 3])

        # b is reused twice, the buffer is now empty.
        self.assertEqual(b.qsize(), 0)
        self.assertEqual(b.empty(), True)

        # Test dropping.
        for i in range(10):
            b.put([i])

        self.assertEqual(b.qsize(), 5)

        for _ in range(2):
            for i in range(9, 4, -1):
                # Lifo.
                self.assertListEqual(b.get().sample.squeeze(-1).tolist(), [i])

        with self.assertRaises(AssertionError):
            b.get()

    def test_no_batching(self):
        b = PriorityQueueBuffer(max_size=5, reuses=1, batch_size=0)
        d = "Some things cannot be batched"
        b.put(d)
        self.assertEqual(b.get().sample, d)
        b = SimpleQueueBuffer(batch_size=0)
        d = "Some things cannot be batched"
        b.put(d)
        self.assertEqual(b.get().sample, d)

    def test_per(self):
        buffer_kwargs = dict(
            max_size=10,
            sample_length=10,
            batch_size=2,
            warmup_transitions=51,
            seed=1,
            priority_interpolation_eta=0.9,
        )
        with self.assertRaises(ValueError):
            b = PrioritizedReplayBuffer(
                **buffer_kwargs,
                alpha=0.6,
                beta=0.3,
                beta_scheduler=base.timeutil.LinearScheduler(init_value=0.4, total_iters=100, end_value=1),
            )
        b = PrioritizedReplayBuffer(
            **buffer_kwargs,
            alpha=1.0,
            beta=1.0,
            max_priority=1.0,
        )

        obs_dim = 5

        def _make_sample_batch(idx, with_precomputed_ret=False):
            if with_precomputed_ret:
                return SampleBatch(
                    obs=np.ones((10, obs_dim), dtype=np.float32) * idx,
                    analyzed_result=AnalyzedResult(
                        value=np.random.randn(10, 1),
                        ret=np.random.randn(10, 1),
                    ),
                )
            else:
                return SampleBatch(
                    obs=np.ones((10, obs_dim), dtype=np.float32) * idx,
                    analyzed_result=AnalyzedResult(
                        value=np.ones((10, 1), dtype=np.float32),
                        ret=np.zeros((10, 1), dtype=np.float32),
                    ),
                )

        b.put(_make_sample_batch(0, True))
        self.assertEqual(len(np.unique(b._min_tree._value)), 10 - 10 + 1 + 1)
        for i in range(1, 3):
            b.put(_make_sample_batch(i))
        self.assertTrue(b.empty())
        for i in range(3, 10):
            b.put(_make_sample_batch(i))
        self.assertTrue(b.full())
        b.put(_make_sample_batch(11))
        self.assertTrue(b.full())

        already_sampled_indices = []

        for j in range(4):  # less than max_size // batch_size
            r: ReplayEntry = b.get()
            self.assertIsInstance(r, ReplayEntry)
            self.assertEqual(r.sample.length(0), 10)
            self.assertEqual(r.sample.length(1), 2)
            self.assertEqual(r.sample.length(2), obs_dim)
            for idx in r.sampling_indices:
                self.assertNotIn(idx, already_sampled_indices)
            if j == 0:
                np.testing.assert_array_equal(r.sample.metadata['sampling_weight'],
                                              np.ones(2, dtype=np.float32))
            # set priority to zero, these data will not be sampled any more
            b.update_priorities(r.sampling_indices, np.zeros(2, dtype=np.float32) + 1e-10)

            already_sampled_indices += list(r.sampling_indices)


if __name__ == '__main__':
    unittest.main()
