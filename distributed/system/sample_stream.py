"""This module defines the data flow between the actor workers and the trainers. It is a simple
producer-consumer model.

A side note that our design chooses to let actor workers see all the data, and posts trajectory
samples to the trainer, instead of letting the policy workers doing so.
"""
from typing import Optional, List, Union, Any
import datetime
import threading
import logging
import json
import numpy as np
import pickle
import socket
import warnings
import zmq
import os
import time
from statistics import mean

from api.trainer import SampleBatch
from base.namedarray import recursive_aggregate, size_bytes
from base.shared_memory import SharedMemoryReader, SharedMemoryWriter, NothingToRead
import api.config
import base.buffer
import distributed.base.monitoring
import distributed.base.name_resolve as name_resolve
import base.names as names
import base.namedarray

logger = logging.getLogger("SampleStream")
SND_RCV_HWM = 10
ZMQ_IO_THREADS = 8


class NothingToConsume(Exception):
    pass


class SampleProducer:
    """Used by the actor workers to post samples to the trainers.
    """

    def post(self, sample):
        """Post a sample. Implementation should be thread safe.
        Args:
            sample: data to be sent.
        """
        raise NotImplementedError()

    def flush(self):
        """Flush all posted samples.
        Thread-safety:
            The implementation of `flush` is considered thread-unsafe. Therefore, on each producer end, only one
            thread should call flush. At the same time, it is safe to call `post` on other threads.
        """
        raise NotImplementedError()

    def init_monitor(self, monitor):
        """ Initialize monitor for sample producer.
        """
        pass

    def close(self):
        """ Explicitly close sample stream. """
        pass


class SampleConsumer:
    """Used by the trainers to acquire samples.

    TODO: we can either implement the replay buffer here or in the trainer. Leave that to-decide.
    """

    def consume_to(self, buffer: base.buffer.Buffer, max_iter) -> int:
        """Consumes all available samples to a target buffer.

        Returns:
            The count of samples added to the buffer.
        """
        raise NotImplementedError()

    def consume(self) -> Any:
        """Consume one from stream. Blocking consume is not supported as it may cause workers to stuck.
        Returns:
            Whatever is sent by the producer.

        Raises:
            NoSampleException: if nothing can be consumed from sample stream.
        """
        raise NotImplementedError()

    def init_monitor(self, monitor):
        """ Initialize monitor for sample producer.
        """
        pass

    def close(self):
        """ Explicitly close sample stream. """
        pass


class NullSampleProducer(SampleProducer):
    """NullSampleProducer discards all samples.
    """

    def flush(self):
        pass

    def post(self, sample):
        pass


class IpSampleProducer(SampleProducer):
    """A simple implementation: sends all samples to a specific consumer (trainer worker).
    """

    def __init__(self, target_address, serialization_method):
        self.__context = zmq.Context(io_threads=ZMQ_IO_THREADS)
        self.__socket = self.__context.socket(zmq.PUSH)
        self.__socket.connect(f"tcp://{target_address}")
        self.__socket.setsockopt(zmq.LINGER, 0)
        self.__socket.setsockopt(zmq.SNDHWM, SND_RCV_HWM)
        self.__sample_buffer = []
        self.__sample_bytes_indicators = []
        self.__post_lock = threading.Lock()
        self.__serialization_method = serialization_method

    def __del__(self):
        self.__socket.close()

    def post(self, sample, is_bytes=False):
        # sample.actor_worker_post_timestamp = np.full(shape=sample.on_reset.shape, fill_value=int(time.time()), dtype=np.int64)
        # data = base.namedarray.dumps(sample, method=self.__serialization_method)
        with self.__post_lock:
            self.__sample_buffer.append(sample)
            self.__sample_bytes_indicators.append(is_bytes)

    def flush(self):
        with self.__post_lock:
            ds = self.__sample_buffer
            is_bytes = self.__sample_bytes_indicators
            self.__sample_buffer = []
            self.__sample_bytes_indicators = []
        for d, ib in zip(ds, is_bytes):
            try:
                # d.actor_worker_flush_timestamp = np.full(shape=d.on_reset.shape, fill_value=int(time.time()), dtype=np.int64)
                if not ib:
                    d = base.namedarray.dumps(d, method=self.__serialization_method)
                self.__socket.send_multipart(d)  # flags=zmq.DONTWAIT
            except zmq.error.Again:
                logger.info(f"Sample producer drops messages!")
                continue


class IpSampleConsumer(SampleConsumer):

    def __init__(self, address):
        self.__context = zmq.Context(io_threads=ZMQ_IO_THREADS)
        self.__socket = self.__context.socket(zmq.PULL)
        self.__socket.RCVTIMEO = 200
        self.__socket.setsockopt(zmq.LINGER, 0)
        self.__socket.setsockopt(zmq.RCVHWM, SND_RCV_HWM)
        if address == "":
            host_ip = socket.gethostbyname(socket.gethostname())
            port = self.__socket.bind_to_random_port(f"tcp://{host_ip}")
            address = f"{host_ip}:{port}"
        else:
            self.__socket.bind(f"tcp://{address}")
        self.address = address
        self.first_time = True

    def __del__(self):
        self.__socket.close()

    def consume_to(self, buffer, max_iter=16):
        count = 0
        for _ in range(max_iter):
            try:
                data = self.__socket.recv_multipart(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            try:
                sample = base.namedarray.loads(data)
            except pickle.UnpicklingError:
                warnings.warn("Unpickling failed. One sample is discarded.")
                continue
            ###############################################################################
            sample.buffer_recv_timestamp = np.full(shape=sample.on_reset.shape,
                                                   fill_value=datetime.datetime.now().timestamp(),
                                                   dtype=np.float64)
            ###############################################################################
            buffer.put(sample)
            if self.first_time:
                logger.info(f"one sample size {size_bytes(sample)}")
                self.first_time = False
            count += 1
        return count

    # def consume_to(self, buffer, max_iter=16):
    #     count = 0
    #     for _ in range(max_iter):
    #         try:
    #             data = self.__socket.recv_multipart(zmq.NOBLOCK)
    #         except zmq.ZMQError:
    #             break

    #         while len(data) > 0:
    #             d = data[:3]
    #             data = data[3:]
    #             try:
    #                 sample = base.namedarray.loads(d)
    #             except pickle.UnpicklingError:
    #                 warnings.warn("Unpickling failed. One sample is discarded.")
    #                 continue
    #             ###############################################################################
    #             sample.buffer_recv_timestamp = np.full(shape=sample.on_reset.shape,
    #                                                    fill_value=datetime.datetime.now().timestamp(),
    #                                                    dtype=np.float64)
    #             ###############################################################################
    #             buffer.put(sample)
    #             count += 1

    #     return count

    def consume(self) -> Any:
        """Note that this method blocks for 0.2 seconds if no sample can be consumed. Therefore, it is safe to make
        a no-sleeping loop on this method. For example:
        while not interrupted:
            try:
                data = consumer.consume()
            except NothingToConsume:
                continue
            process(data)
        """
        # This method should not be called.
        assert False
        try:
            data = self.__socket.recv_multipart()
        except zmq.ZMQError:
            raise NothingToConsume()
        try:
            sample = base.namedarray.loads(data)
        except pickle.UnpicklingError:
            warnings.warn("Unpickling failed. One sample is discarded.")
            raise NothingToConsume()
        return sample


class NameResolvingSampleConsumer(IpSampleConsumer):

    def __init__(self, experiment_name, trial_name, stream_name, address=""):
        super().__init__(address)
        self.__name_entry = name_resolve.add_subentry(names.sample_stream(experiment_name=experiment_name,
                                                                          trial_name=trial_name,
                                                                          stream_name=stream_name),
                                                      json.dumps({"address": self.address}),
                                                      keepalive_ttl=30)
        self.monitor = None

    def consume_to(self, buffer, max_iter=16):
        count = super(NameResolvingSampleConsumer, self).consume_to(buffer, max_iter)
        # self.monitor.metric("marl_sample_stream_received").inc(count)
        return count

    def consume(self, block=True) -> Any:
        sample = super(NameResolvingSampleConsumer, self).consume()
        # self.monitor.metric("marl_sample_stream_received").inc(1)
        return sample

    def init_monitor(self, monitor: distributed.base.monitoring.Monitor):
        """ Get monitor from trainer workers """
        self.monitor = monitor
        metric = dict(marl_sample_stream_received="Counter")
        self.monitor.update_metrics(metric)


class NameResolvingSampleProducer(IpSampleProducer):

    def __init__(self, experiment_name, trial_name, stream_name, rank, serialization_method):
        name = names.sample_stream(experiment_name=experiment_name,
                                   trial_name=trial_name,
                                   stream_name=stream_name)
        server_json_configs = list(sorted(name_resolve.get_subtree(name)))

        assert len(server_json_configs) > 0, f"No trainer configuration found. Initialize trainers first" \
                                             f"experiment: {experiment_name} stream_name: {stream_name}"
        target_tw_rank = rank % len(server_json_configs)
        server_config = json.loads(server_json_configs[target_tw_rank])

        super().__init__(target_address=server_config['address'].replace("*", "localhost"),
                         serialization_method=serialization_method)
        self.monitor = None

    def post(self, sample: SampleBatch, is_bytes=False):
        # self.monitor.metric("marl_sample_stream_sent").inc(1)
        # self.monitor.metric("marl_sample_producer_parameter_version").set(
        #     sample.average_of("policy_version_steps") or 0)
        super(NameResolvingSampleProducer, self).post(sample, is_bytes=is_bytes)

    def init_monitor(self, monitor: distributed.base.monitoring.Monitor):
        """ Get monitor from actor workers """
        self.monitor = monitor
        metrics = dict(marl_sample_stream_sent="Counter", marl_sample_producer_parameter_version="Gauge")
        self.monitor.update_metrics(metrics)


class NameResolvingMultiAgentSampleProducer(NameResolvingSampleProducer):

    def __init__(self, experiment_name, trial_name, stream_name, rank, serialization_method):
        super(NameResolvingMultiAgentSampleProducer, self).__init__(experiment_name, trial_name, stream_name,
                                                                    rank, serialization_method)
        self.__cache = []
        self.logger = logging.getLogger("MA Producer")

    def post(self, sample: SampleBatch):
        self.__cache.append(sample)

    def flush(self):
        if self.__cache:
            if None in [sample.unique_policy_version for sample in self.__cache]:
                self.__cache = []
                return

            super(NameResolvingMultiAgentSampleProducer,
                  self).post(recursive_aggregate(self.__cache, lambda x: np.stack(x, axis=1)))
            self.logger.debug(f"posted samples with "
                              f"version {[sample.unique_policy_version for sample in self.__cache]} "
                              f"and name {[sample.unique_policy_name for sample in self.__cache]}.")
            self.__cache = []


class ZippedSampleProducer(SampleProducer):

    def __init__(self, sample_producers: List[SampleProducer]):
        self.__producers = sample_producers

    def post(self, sample):
        # TODO: With the current implementation, we are pickling samples for multiple times.
        for p in self.__producers:
            p.post(sample)

    def flush(self):
        for p in self.__producers:
            p.flush()


class RoundRobinNameResolvingSampleProducer(SampleProducer):

    def __init__(self, experiment_name, trial_name, stream_name, rank, serialization_method):
        name = names.sample_stream(experiment_name=experiment_name,
                                   trial_name=trial_name,
                                   stream_name=stream_name)
        server_json_configs = list(sorted(name_resolve.get_subtree(name)))

        assert len(server_json_configs) > 0, f"No trainer configuration found. Initialize trainers first" \
                                             f"experiment: {experiment_name} stream_name: {stream_name}"
        self.__streams = [
            NameResolvingSampleProducer(experiment_name,
                                        trial_name,
                                        stream_name,
                                        rank=r,
                                        serialization_method=serialization_method)
            for r in range(len(server_json_configs))
        ]
        self.__current_idx = rank % len(server_json_configs)

    def post(self, sample):
        self.__streams[self.__current_idx].post(sample)
        self.__current_idx = (self.__current_idx + 1) % len(self.__streams)

    def flush(self):
        for p in self.__streams:
            p.flush()

    def init_monitor(self, monitor):
        for s in self.__streams:
            s.init_monitor(monitor)


class BroadcastNameResolvingSampleProducer(SampleProducer):

    def __init__(self, experiment_name, trial_name, stream_name, serialization_method):
        name = names.sample_stream(experiment_name=experiment_name,
                                   trial_name=trial_name,
                                   stream_name=stream_name)
        server_json_configs = list(sorted(name_resolve.get_subtree(name)))

        assert len(server_json_configs) > 0, f"No trainer configuration found. Initialize trainers first" \
                                             f"experiment: {experiment_name} stream_name: {stream_name}"
        self.__streams = [
            NameResolvingSampleProducer(experiment_name,
                                        trial_name,
                                        stream_name,
                                        rank=r,
                                        serialization_method=serialization_method)
            for r in range(len(server_json_configs))
        ]

    def post(self, sample):
        sample = base.namedarray.dumps(sample)
        for stream in self.__streams:
            stream.post(sample, is_bytes=True)

    def flush(self):
        for stream in self.__streams:
            stream.flush()


class InlineSampleProducer(SampleProducer):
    """Testing Only! Will not push parameters.
    """

    def __init__(self, trainer, policy):
        from api.trainer import make
        from distributed.system.parameter_db import make_db
        from api.config import ParameterDB

        self.trainer = make(trainer, policy)
        self.buffer = []
        self.logger = logging.getLogger("Inline Training")
        self.param_db = make_db(ParameterDB(type_=ParameterDB.Type.LOCAL_TESTING))
        self.param_db.push(name="", checkpoint=self.trainer.get_checkpoint(), version=0)

    def post(self, sample):
        self.buffer.append(sample)
        self.logger.debug("Receive sample.")

    def flush(self):
        if len(self.buffer) >= 5:
            batch_sample = recursive_aggregate(self.buffer, aggregate_fn=lambda x: np.stack(x, axis=1))
            batch_sample.policy_name = None
            self.trainer.step(batch_sample)
            self.param_db.push(name="", checkpoint=self.trainer.get_checkpoint(), version=0)
            self.logger.info("Trainer step is successful!")
            self.logger.debug(f"Trainer steps. now on version {self.trainer.policy.version}.")
            self.buffer = []


class SharedMemorySampleProducer(SampleProducer):

    def __init__(self, experiment_name, trial_name, stream_name, qsize):
        # Default: 1 GB shared memory for sample buffer
        self.__shared_memory_writer = SharedMemoryWriter(qsize, experiment_name, trial_name, stream_name)
        self.__post_lock = threading.Lock()
        self.__sample_buffer = []

    def post(self, sample):
        with self.__post_lock:
            self.__sample_buffer.append(sample)

    def flush(self):
        with self.__post_lock:
            tmp = self.__sample_buffer
            self.__sample_buffer = []
        for x in tmp:
            self.__shared_memory_writer.write(x)

    def close(self):
        self.__shared_memory_writer.close()


class SharedMemorySampleConsumer(SampleConsumer):

    def __init__(self, experiment_name, trial_name, stream_name, qsize, batch_size):
        self.__shared_memory_reader = SharedMemoryReader(qsize, experiment_name, trial_name, stream_name,
                                                         batch_size)
        self.first_time = True

    def consume_to(self, buffer, max_iter):
        count = 0
        for _ in range(max_iter):
            try:
                st = time.monotonic()
                sample = self.__shared_memory_reader.read()
            except NothingToRead:
                break
            if_batch = buffer.put(sample)
            if self.first_time:
                logger.info(f"one sample size {size_bytes(sample)}")
                self.first_time = False
            count += 1
        return count

    def consume(self):
        try:
            return self.__shared_memory_reader.read()
        except NothingToRead:
            raise NothingToConsume()

    def close(self):
        self.__shared_memory_reader.close()


def make_producer(spec: Union[str, api.config.SampleStream],
                  worker_info: Optional[api.config.WorkerInformation] = None):
    """Initializes a sample producer (client).

    Args:
        spec: Configuration of the sample stream.
        worker_info: Worker information.
    """
    if isinstance(spec, str):
        spec = api.config.SampleStream(type_=api.config.SampleStream.Type.NAME, stream_name=spec)
    if spec.type_ == api.config.SampleStream.Type.IP:
        return IpSampleProducer(target_address=spec.address, serialization_method=spec.serialization_method)
    elif spec.type_ == api.config.SampleStream.Type.NAME:
        return NameResolvingSampleProducer(experiment_name=worker_info.experiment_name,
                                           trial_name=worker_info.trial_name,
                                           stream_name=spec.stream_name,
                                           rank=worker_info.worker_index,
                                           serialization_method=spec.serialization_method)
    elif spec.type_ == api.config.SampleStream.Type.NAME_MULTI_AGENT:
        return NameResolvingMultiAgentSampleProducer(experiment_name=worker_info.experiment_name,
                                                     trial_name=worker_info.trial_name,
                                                     stream_name=spec.stream_name,
                                                     rank=worker_info.worker_index,
                                                     serialization_method=spec.serialization_method)
    elif spec.type_ == api.config.SampleStream.Type.NAME_ROUND_ROBIN:
        return RoundRobinNameResolvingSampleProducer(experiment_name=worker_info.experiment_name,
                                                     trial_name=worker_info.trial_name,
                                                     stream_name=spec.stream_name,
                                                     rank=worker_info.worker_index,
                                                     serialization_method=spec.serialization_method)
    elif spec.type_ == api.config.SampleStream.Type.NAME_BROADCAST:
        return BroadcastNameResolvingSampleProducer(experiment_name=worker_info.experiment_name,
                                                    trial_name=worker_info.trial_name,
                                                    stream_name=spec.stream_name,
                                                    serialization_method=spec.serialization_method)
    elif spec.type_ == api.config.SampleStream.Type.NULL:
        return NullSampleProducer()
    elif spec.type_ == api.config.SampleStream.Type.INLINE_TESTING:
        return InlineSampleProducer(trainer=spec.trainer, policy=spec.policy)
    elif spec.type_ == api.config.SampleStream.Type.SHARED_MEMORY:
        return SharedMemorySampleProducer(experiment_name=worker_info.experiment_name,
                                          trial_name=worker_info.trial_name,
                                          stream_name=spec.stream_name,
                                          qsize=spec.qsize)
    else:
        raise NotImplementedError()


def make_consumer(spec: Union[str, api.config.SampleStream],
                  worker_info: Optional[api.config.WorkerInformation] = None):
    """Initializes a sample consumer (server).

    Args:
        spec: Configuration of the sample stream.
        worker_info: Worker information.
    """
    if isinstance(spec, str):
        spec = api.config.SampleStream(type_=api.config.SampleStream.Type.NAME, stream_name=spec)
    if spec.type_ == api.config.SampleStream.Type.IP:
        return IpSampleConsumer(address=spec.address)
    elif spec.type_ == api.config.SampleStream.Type.NAME:
        return NameResolvingSampleConsumer(experiment_name=worker_info.experiment_name,
                                           trial_name=worker_info.trial_name,
                                           stream_name=spec.stream_name)
    elif spec.type_ == api.config.SampleStream.Type.SHARED_MEMORY:
        return SharedMemorySampleConsumer(experiment_name=worker_info.experiment_name,
                                          trial_name=worker_info.trial_name,
                                          stream_name=spec.stream_name,
                                          qsize=spec.qsize,
                                          batch_size=spec.batch_size)
    else:
        raise NotImplementedError()


def zip_producers(sample_producers: List[SampleProducer]):
    return ZippedSampleProducer(sample_producers)
