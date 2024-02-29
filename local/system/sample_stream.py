"""This module defines the data flow between the actor workers and the trainers. It is a simple
producer-consumer model.

A side note that our design chooses to let actor workers see all the data, and posts trajectory
samples to the trainer, instead of letting the policy workers doing so.
"""
from typing import Optional, List, Union, Any

import api.config
import base.buffer


class NothingToConsume(Exception):
    pass


class SampleProducer:
    """Used by the actor workers to post samples to the trainers.
    """

    @property
    def type(self):
        raise NotImplementedError()

    def post(self, sample):
        raise NotImplementedError()


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


class ZippedSampleProducer(SampleProducer):

    def __init__(self, sample_producers: List[SampleProducer]):
        self.__producers = sample_producers

    def post(self, sample):
        # TODO: With the current implementation, we are pickling samples for multiple times.
        for p in self.__producers:
            p.post(sample)


ALL_SAMPLE_PRODUCER_CLS = {}
ALL_SAMPLE_CONSUMER_CLS = {}


def register_producer(type_: api.config.SampleStream.Type, cls):
    ALL_SAMPLE_PRODUCER_CLS[type_] = cls


def register_consumer(type_: api.config.SampleStream.Type, cls):
    ALL_SAMPLE_CONSUMER_CLS[type_] = cls


def make_producer(spec: Union[str, api.config.SampleStream, SampleProducer],
                  worker_info: Optional[api.config.WorkerInformation] = None):
    """Initializes a sample producer (client).

    Args:
        spec: Configuration of the sample stream.
        worker_info: Worker information.
    """
    if isinstance(spec, SampleProducer):
        return spec
    if isinstance(spec, str):
        spec = api.config.SampleStream(type_=api.config.SampleStream.Type.NAME, stream_name=spec)
    if spec.worker_info is None:
        spec.worker_info = worker_info
    return ALL_SAMPLE_PRODUCER_CLS[spec.type_](spec)


def make_consumer(spec: Union[str, api.config.SampleStream, SampleConsumer],
                  worker_info: Optional[api.config.WorkerInformation] = None):
    """Initializes a sample consumer (server).

    Args:
        spec: Configuration of the sample stream.
        worker_info: Worker information.
    """
    if isinstance(spec, SampleConsumer):
        return spec
    if isinstance(spec, str):
        spec = api.config.SampleStream(type_=api.config.SampleStream.Type.NAME, stream_name=spec)
    if spec.worker_info is None:
        spec.worker_info = worker_info
    return ALL_SAMPLE_CONSUMER_CLS[spec.type_](spec)


def zip_producers(sample_producers: List[SampleProducer]):
    return ZippedSampleProducer(sample_producers)
