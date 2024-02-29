from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, List, Dict, Union
import bisect
import math
import logging
import numpy as np
import queue
import random
import time

import numpy as np

from base.namedarray import recursive_aggregate, recursive_apply, array_like, NamedArray
from base.segment_tree import MinSegmentTree, SumSegmentTree
import base.namedarray
import base.numpy_utils
import base.timeutil

import logging

logger = logging.getLogger("Buffer")


@dataclass(order=True)
class ReplayEntry:
    reuses_left: int
    receive_time: float
    sample: Any = field(compare=False)
    reuses: int = field(default=0, compare=False)
    sampling_indices: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.sample)


class Buffer:

    def put(self, x):
        raise NotImplementedError()

    def get(self):
        raise NotImplementedError()

    def empty(self):
        raise NotImplementedError()

    def qsize(self) -> int:
        raise NotImplementedError()


class SimpleQueueBuffer(Buffer):

    def __init__(self, batch_size=None, max_size=1e9):
        self.__queue = queue.SimpleQueue()
        self.__max_size = max_size
        self.__tmp_storage = []
        self.batch_size = batch_size

    def qsize(self):
        return self.__queue.qsize()

    def put(self, x):
        if self.batch_size:
            self.__tmp_storage.append(x)
            if len(self.__tmp_storage) >= self.batch_size:
                # TODO : as mentioned by qiwei, we may want to rename this to get_batch
                data = recursive_aggregate(self.__tmp_storage[:self.batch_size],
                                           lambda x: np.stack(x, axis=1))
                self.__tmp_storage = self.__tmp_storage[self.batch_size:]
                self.__queue.put_nowait(data)
        else:
            self.__queue.put_nowait(x)

        if self.__queue.qsize() > self.__max_size:
            self.__queue.get_nowait()

    def get(self) -> ReplayEntry:
        return ReplayEntry(reuses_left=0,
                           reuses=0,
                           receive_time=time.time(),
                           sample=self.__queue.get_nowait())

    def empty(self):
        return self.__queue.empty()


class PriorityQueueBuffer:

    def __init__(self, max_size=16, reuses=1, batch_size=1):
        self.__buffer = []
        self.__tmp_storage = []
        self.__max_size = max_size
        self.reuses = reuses
        self.batch_size = batch_size

    @property
    def overflow(self):
        return len(self.__buffer) > self.__max_size

    def full(self):
        return len(self.__buffer) == self.__max_size

    def empty(self):
        return len(self.__buffer) == 0

    def qsize(self):
        return len(self.__buffer)

    def put(self, x):
        # FIXME: this is a hack to make sure that the info is not used in trainer.
        # x.info_mask[:] = 0
        # x.info = None
        if self.batch_size:
            x.trainer_worker_recv_timestamp = np.full(shape=x.on_reset.shape,
                                                      fill_value=int(time.time()),
                                                      dtype=np.int64)
            self.__tmp_storage.append(x)
            if len(self.__tmp_storage) >= self.batch_size:
                # st = time.monotonic()
                data = recursive_aggregate(self.__tmp_storage[:self.batch_size],
                                           lambda x: np.stack(x, axis=1))
                # logger.info(
                #     f"batch aggregation took {time.monotonic() - st:.3f} seconds, batch size {self.batch_size}."
                # )
                self.__tmp_storage = self.__tmp_storage[self.batch_size:]
                self.__put(ReplayEntry(reuses_left=self.reuses, sample=data, receive_time=time.time()))
                return True
        else:
            self.__put(ReplayEntry(reuses_left=self.reuses, sample=x, receive_time=time.time()))
        return False

    def __put(self, r):
        # sample_policy_version = r.sample.average_of("policy_version_steps",
        #                                             ignore_negative=True)
        # sample_policy_version_min = r.sample.min_of("policy_version_steps",
        #                                             ignore_negative=True)
        # logger.info(f"Put Sample, sample policy version: avg: {sample_policy_version}, min: {sample_policy_version_min}, reuse left {r.reuses_left}")
        bisect.insort(self.__buffer, r)
        while self.overflow:
            self.__drop()

    def get(self) -> ReplayEntry:
        assert not self.empty(), "attempting to get from empty buffer."
        r = self.__buffer.pop(-1)
        r.reuses_left -= 1
        r.reuses += 1
        # sample_policy_version = r.sample.average_of("policy_version_steps",
        #                                             ignore_negative=True)
        # sample_policy_version_min = r.sample.min_of("policy_version_steps",
        #                                             ignore_negative=True)
        # logger.info(f"Get Sample, sample policy version: avg: {sample_policy_version}, min: {sample_policy_version_min}, reuse left {r.reuses_left}")
        # aw_post = r.sample.average_of("actor_worker_post_timestamp")
        # aw_flush = r.sample.average_of("actor_worker_flush_timestamp")
        # tw_recv = r.sample.average_of("trainer_worker_recv_timestamp")
        # tw_batch = r.receive_time
        # now = time.time()
        # logger.info(f"timestamps: aw_post {aw_post}, aw_flush {aw_flush}, tw_recv {tw_recv}, tw_batch {tw_batch}, now {now}, \n"
        #             f"diff {(aw_flush-aw_post, tw_recv-aw_flush, tw_batch-tw_recv, now-tw_recv)}")
        if not self.full() and r.reuses_left > 0:
            self.__put(r)

        return r

    def __drop(self):
        self.__buffer.pop(0)


# Select a random chunk with self._batch_length. Just a faster implementation.
def _slicing(x, t, b, batch_length, batch_size):
    indices = b[None, :] + (t[None, :] + np.arange(batch_length)[:, None]) * x.shape[1]
    indices = indices.ravel()
    return x.reshape(-1, *x.shape[2:])[indices].reshape(batch_length, batch_size, *x.shape[2:])


class SimpleReplayBuffer(Buffer):

    def __init__(self,
                 max_size: int,
                 batch_size: int,
                 warmup_transitions: int,
                 seed: int = 0,
                 sample_length: Optional[int] = None,
                 batch_length: Optional[int] = None):
        """Uniform replay buffer.

        If batch_length < sample_length, sampled chunks may overlap.

        Args:
            max_size (int): The maximum size of the internal `SampleBatch` (i.e., self._buffer.length(1)).
            batch_size (int): The number of chunks required for each training step.
            warmup_transitions (int): The number of *transitions* required before training begins.
            seed (int, optional): Random seed for sampling. Defaults to 0.
            sample_length (int, optional): The length of received `SampleBatch`.
                If not given, it will be set to the length of the first received `SampleBatch`.
            batch_length (int, optional): The chunk length required for each training step.
                If not given, defaults to sample_length.
        """
        self._max_size = max_size
        self._sample_length = sample_length
        self._batch_length = batch_length
        self._batch_size = batch_size
        self._warmup_transitions = warmup_transitions
        self._buffer = None  # will be initialized after receiving the first sample

        self._total_transitions = 0
        self._total_chunks = 0
        self._ptr = 0

        self._replay_times = np.zeros(self._max_size * self.n_chunks_per_sample, dtype=np.int64)

        np.random.seed(seed)

    def qsize(self):
        return self._total_transitions

    @property
    def n_chunks_per_sample(self):
        return self._sample_length - self._batch_length + 1

    def empty(self):
        return self._total_transitions < self._warmup_transitions

    def full(self):
        return self._total_transitions == self._max_size * self._sample_length

    def put(self, x):
        if hasattr(x, "policy_name"):
            # FIXME: temporary hack
            x.policy_name = None
        if self._sample_length is None:
            self._sample_length = x.length(0)
        if self._batch_length is None:
            self._batch_length = self._sample_length
        if x.length(0) != self._sample_length:
            raise RuntimeError("The sample received in SimpleReplayBuffer"
                               " has a different sample length from configuration! "
                               f"({x.length(0)} vs {self._sample_length})")
        if self._buffer is None:
            self._buffer = base.namedarray.from_flattened([(
                k,
                np.zeros(
                    (v.shape[0], self._max_size, *v.shape[1:]),
                    dtype=v.dtype,
                ) if v is not None else None,
            ) for k, v in base.namedarray.flatten(x)],)
            # TODO: trying not to import torch in base
            import torch
            self._torch_buffer = recursive_apply(self._buffer, lambda x: torch.from_numpy(x).pin_memory())
            self._buffer = recursive_apply(self._torch_buffer, lambda x: x.numpy())

        self._buffer[:, self._ptr] = x

        offset = self._ptr * self.n_chunks_per_sample
        self._replay_times[offset:offset + self.n_chunks_per_sample] = 0

        self._total_transitions = min(self._total_transitions + self._sample_length,
                                      self._max_size * self._sample_length)
        self._total_chunks = min(self._max_size * self.n_chunks_per_sample,
                                 self._total_chunks + self.n_chunks_per_sample)
        self._ptr = (self._ptr + 1) % self._max_size

    def get(self):
        global_indices = np.random.choice(self._total_chunks, self._batch_size, replace=False)
        time_indices = global_indices % self.n_chunks_per_sample
        batch_indices = global_indices // self.n_chunks_per_sample

        avg_reuses = self._replay_times[global_indices].mean()
        self._replay_times[global_indices] += 1

        return ReplayEntry(
            reuses_left=-1,
            sample=recursive_apply(
                self._buffer,
                lambda x: _slicing(x, time_indices, batch_indices, self._batch_length, self._batch_size)),
            receive_time=-1,
            reuses=avg_reuses,
        )


class PrioritizedReplayBuffer(SimpleReplayBuffer):

    def __init__(self,
                 max_size: int,
                 sample_length: int,
                 batch_size: int,
                 warmup_transitions: int,
                 seed: int = 0,
                 burn_in_steps: int = 0,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_scheduler: Optional[Union[base.timeutil.Scheduler,
                                                base.timeutil.ChainedScheduler]] = None,
                 max_priority: float = 1.0,
                 priority_interpolation_eta: Optional[float] = None):
        """Initialize Piroritized experience replay (https://arxiv.org/pdf/1511.05952.pdf).

        Sampled chunks will not overlap.

        Args:
            max_size (int): The maximum length of the internal `SampleBatch` list.
            sample_length (int): The length of received `SampleBatch` (must be provided such that we know the capacity of transitions).
            batch_size (int): The number of chunks required for each training step.
            warmup_transitions (int): The number of *transitions* required before training begins.
            seed (int, optional): Random seed for sampling. Defaults to 0.
            alpha (float, optional): Priority exponent. Defaults to 0.6.
            beta (float, optional): Sample weight exponent. Defaults to 0.4.
            beta_scheduler (str, optional): The scheduler of `beta`. Defaults to None.
            max_priority (float): The maximum priority used for initialization. Defaults to 1.0.
            prioriti_interpolation_eta (float, optional): The interpolation factor of max and mean TD error.
        """
        if alpha < 0:
            raise ValueError("`alpha` must be a non-negative number.")
        if beta <= 0:
            raise ValueError("`beta` must be a positive number.")
        if beta_scheduler is None:
            beta_scheduler = base.timeutil.ConstantScheduler(init_value=beta, total_iters=math.inf)
        if beta_scheduler.init_value != beta:
            raise ValueError("`init_value` of `beta_scheduler` should be the same as the `beta` argument.")

        super(PrioritizedReplayBuffer, self).__init__(
            max_size=max_size,
            batch_size=batch_size,
            warmup_transitions=warmup_transitions,
            seed=seed,
            sample_length=sample_length,
            batch_length=sample_length,
        )
        self._burn_in_steps = burn_in_steps
        assert self.n_chunks_per_sample == 1
        self._max_priority = max_priority
        self._alpha, self._beta = alpha, beta
        self._beta_scheduler = beta_scheduler

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self._max_size * self.n_chunks_per_sample:
            tree_capacity *= 2

        self._sum_tree = SumSegmentTree(tree_capacity)
        self._min_tree = MinSegmentTree(tree_capacity)

        # used for logging
        self._replay_times = np.zeros(self._max_size * self.n_chunks_per_sample, dtype=np.int64)
        self._priorities = np.zeros(self._max_size * self.n_chunks_per_sample, dtype=np.float32)

        self._training_steps = 0

        self._eta = priority_interpolation_eta

    def _compute_init_priorities(self, sample):
        if not hasattr(sample, "analyzed_result") or sample.analyzed_result.ret is None:
            return self._max_priority * np.ones(self.n_chunks_per_sample, dtype=np.float32)

        td_err = np.abs(sample.analyzed_result.ret -
                        sample.analyzed_result.value).squeeze(-1)[self._burn_in_steps:]
        mean_td_err = td_err.mean(keepdims=True)
        max_td_err = td_err.max(keepdims=True)

        priorities = self._eta * max_td_err + (1 - self._eta) * mean_td_err
        assert priorities.shape == (self.n_chunks_per_sample,)
        return priorities

    def put(self, x):
        # logger.info(f"PER buffer put with policy version steps {x.policy_version_steps.squeeze()} and on_reset {x.on_reset.squeeze()} and done {x.done.squeeze()}.")
        if hasattr(x, "policy_name"):
            # FIXME: temporary hack
            x.policy_name = None
        if x.length(0) != self._sample_length:
            raise RuntimeError("The sample received in PrioritizedReplayBuffer"
                               " has a different sample length from configuration! "
                               f"({x.length(0)} vs {self._sample_length})")
        if self._buffer is None:
            self._buffer = base.namedarray.from_flattened([(
                k,
                np.zeros(
                    (v.shape[0], self._max_size, *v.shape[1:]),
                    dtype=v.dtype,
                ) if v is not None else None,
            ) for k, v in base.namedarray.flatten(x)],)
            # TODO: trying not to import torch in base
            import torch
            self._torch_buffer = recursive_apply(self._buffer, lambda x: torch.from_numpy(x).pin_memory())
            self._buffer = recursive_apply(self._torch_buffer, lambda x: x.numpy())

        self._buffer[:, self._ptr] = x

        # Compute initial priority.
        init_priorities = self._compute_init_priorities(x)

        offset = self._ptr * self.n_chunks_per_sample
        for idx in range(offset, offset + self.n_chunks_per_sample):
            self._sum_tree[idx] = init_priorities[idx - offset]**self._alpha
            self._min_tree[idx] = init_priorities[idx - offset]**self._alpha

        # logger.info(
        #     f"Buffer put flushed sample replay times: {self._replay_times[offset:offset + self.n_chunks_per_sample]}, "
        #     f"flushed priority {self._priorities[offset:offset + self.n_chunks_per_sample]}, "
        #     f"mean/max priority {self._priorities.mean()}/{self._priorities.max()}, "
        #     f"initial priority {init_priorities**self._alpha}, ptr {self._ptr}.")
        self._replay_times[offset:offset + self.n_chunks_per_sample] = 0
        self._priorities[offset:offset + self.n_chunks_per_sample] = init_priorities**self._alpha

        self._total_transitions = min(self._total_transitions + self._sample_length,
                                      self._max_size * self._sample_length)
        self._total_chunks = min(self._max_size * self.n_chunks_per_sample,
                                 self._total_chunks + self.n_chunks_per_sample)
        self._ptr = (self._ptr + 1) % self._max_size

    def _sample_proportional(self):
        res = np.zeros(self._batch_size, dtype=np.int64)
        p_total = self._sum_tree.sum(0, self._total_chunks - 1)
        every_range_len = p_total / self._batch_size
        for i in range(self._batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._sum_tree.find_prefixsum_idx(mass)
            res[i] = idx
        return np.sort(res)

    def get(self):
        idxes = self._sample_proportional()
        # TODO: batch first
        time_indices = idxes % self.n_chunks_per_sample
        batch_indices = idxes // self.n_chunks_per_sample

        # Select a random chunk with self._batch_length. Just a faster implementation.
        data = recursive_apply(
            self._buffer,
            lambda x: _slicing(x, time_indices, batch_indices, self._batch_length, self._batch_size))

        weights = []
        p_min = self._min_tree.min() / self._sum_tree.sum()
        max_weight = (p_min * self._total_chunks)**(-self._beta)

        for idx in idxes:
            p_sample = self._sum_tree[idx] / self._sum_tree.sum()
            weight = (p_sample * self._total_chunks)**(-self._beta)
            weights.append(weight / max_weight)

        data.register_metadata(sampling_weight=np.array(weights, dtype=np.float32))
        # logger.info(f"PER buffer GET replay times {self._replay_times[idxes]}, indices {idxes}, priorities {self._priorities[idxes]}, "
        #             f"on_reset? {data.on_reset.any(0).squeeze()}, done? {data.done.any(0).squeeze()}, non-zero reward? {(data.reward > 0).any(0).squeeze()}, "
        #             f"policy version {data.policy_version_steps.min(0).squeeze()}.")
        return ReplayEntry(reuses_left=-1,
                           sample=data,
                           receive_time=-1,
                           sampling_indices=idxes,
                           reuses=self._replay_times[idxes].mean())

    def update_priorities(self, idxes, priorities):
        assert idxes.shape == priorities.shape == (self._batch_size,), (idxes.shape, priorities.shape)
        self._replay_times[idxes] += 1
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self._total_chunks
            self._sum_tree[idx] = priority**self._alpha
            self._min_tree[idx] = priority**self._alpha

            self._max_priority = max(self._max_priority, priority)

        self._priorities[idxes] = priorities**self._alpha
        self._training_steps += 1
        self._beta = self._beta_scheduler.get(self._training_steps)
        # logger.info(f"PER buffer update new priorities {idxes, self._priorities[idxes]}.")
        # if self._training_steps % 100 == 0:
        #     logger.info(f"PER buffer average/max replay times {self._replay_times.mean()}/{self._replay_times.max()}")


# class DALIPrioritizedReplayBuffer(PrioritizedReplayBuffer):

#     def __init__(self,
#                  max_size: int,
#                  sample_length: int,
#                  batch_length: int,
#                  batch_size: int,
#                  warmup_transitions: int,
#                  seed: int = 0,
#                  alpha: float = 0.6,
#                  beta: float = 0.4,
#                  beta_scheduler: Optional[Union[base.timeutil.Scheduler,
#                                                 base.timeutil.ChainedScheduler]] = None,
#                  max_priority: float = 1.0,
#                  priority_interpolation_eta: Optional[float] = None,
#                  num_threads: int = 4,
#                  **pipline_kwargs):
#         super().__init__(max_size, sample_length, batch_length, batch_size, warmup_transitions, seed, alpha,
#                          beta, beta_scheduler, max_priority, priority_interpolation_eta)
#         self.num_threads = num_threads
#         self.pipeline_kwargs = pipline_kwargs
#         self.pipe = None
#         self.pipe_initialized = False

#         self._sample_batch_size_ptr = 0

#     def _sample_one_propotional(self):
#         res = np.zeros(self._batch_size, dtype=np.int64)
#         p_total = self._sum_tree.sum(0, self._total_transitions - 1)
#         every_range_len = p_total / self._batch_size
#         mass = random.random() * every_range_len + self._sample_batch_size_ptr * every_range_len
#         self._sample_batch_size_ptr = (self._sample_batch_size_ptr + 1) % self._batch_size
#         return self._sum_tree.find_prefixsum_idx(mass)

#     def _get_flattened_sample(self):
#         idx = self._sample_one_propotional()
#         time_idx = idx % self.n_transitions_per_sample
#         batch_idx = idx // self.n_transitions_per_sample
#         return [v[time_idx:time_idx + self._batch_length, batch_idx] for v in self._sample_values] + [idx]

#     def put(self, x):
#         super().put(x)
#         if not self.pipe_initialized:
#             self._buffer_keys, self._buffer_values = zip(*base.namedarray.flatten(self._buffer))
#             self._sample_keys = [k for k, v in zip(self._buffer_keys, self._buffer_values) if v is not None]
#             self._sample_values = [v for v in self._buffer_values if v is not None]

#             from nvidia.dali.pipeline import Pipeline
#             import nvidia.dali.fn as fn
#             self.pipe = Pipeline(batch_size=self._batch_size,
#                                  num_threads=self.num_threads,
#                                  **self.pipeline_kwargs)
#             with self.pipe:
#                 a = fn.external_source(
#                     source=self._get_flattened_sample,
#                     num_outputs=len(self._sample_keys) + 1,
#                 )
#                 self.pip.set_outputs(*a)
#             self.pipe.build()
#             self.pipe_initialized = True

#     def get(self, x):
#         values = self.pipe.run()


def make_buffer(name, **buffer_args):
    if name == "simple_queue":
        return SimpleQueueBuffer(**buffer_args)
    elif name == "priority_queue":
        return PriorityQueueBuffer(**buffer_args)
    elif name == "simple_replay_buffer":
        return SimpleReplayBuffer(**buffer_args)
    elif name == "prioritized_replay_buffer":
        return PrioritizedReplayBuffer(**buffer_args)
