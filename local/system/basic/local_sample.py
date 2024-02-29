import queue
from typing import Any, Tuple
import logging

from base.namedarray import recursive_aggregate
from local.system.sample_stream import SampleProducer, SampleConsumer, NothingToConsume, register_producer
import base.namedarray
import api.config


class LocalSampleProducer(SampleProducer):

    @property
    def type(self):
        return None

    def __init__(self, q):
        self.q = q

    def post(self, sample):
        self.q.put(base.namedarray.dumps(sample))


class LocalSampleConsumer(SampleConsumer):

    def __init__(self, q):
        self.q = q
        self.consume_timeout = 2

    def consume_to(self, buffer, max_iter=64):
        count = 0
        for _ in range(max_iter):
            try:
                sample = base.namedarray.loads(self.q.get_nowait())
            except queue.Empty:
                break
            buffer.put(sample)
            count += 1
        return count

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
        try:
            return base.namedarray.loads(self.q.get(timeout=self.consume_timeout))
        except queue.Empty:
            raise NothingToConsume()


class NullSampleProducer(SampleProducer):
    """NullSampleProducer discards all samples.
    """

    @property
    def type(self):
        return api.config.SampleStream.Type.NULL

    def __init__(self, spec):
        pass

    def post(self, sample):
        pass


class InlineSampleProducer(SampleProducer):
    """Testing Only! Will not push parameters.
    """

    @property
    def type(self):
        return api.config.SampleStream.Type.INLINE_TESTING

    def __init__(self, spec: api.config.SampleStream):
        from api.trainer import make
        from local.system.parameter_db import make_db
        from api.config import ParameterDB

        self.trainer = make(spec.trainer, spec.policy)
        self.buffer = []
        self.logger = logging.getLogger("Inline Training")
        self.param_db = make_db(ParameterDB(type_=ParameterDB.Type.LOCAL_TESTING))
        self.param_db.push(name="", checkpoint=self.trainer.get_checkpoint(), version=0)

    def post(self, sample):
        self.buffer.append(sample)
        self.logger.debug("Receive sample.")
        if len(self.buffer) >= 5:
            batch_sample = recursive_aggregate(self.buffer, aggregate_fn=lambda x: np.stack(x, axis=1))
            batch_sample.policy_name = None
            self.trainer.step(batch_sample)
            self.param_db.push(name="", checkpoint=self.trainer.get_checkpoint(), version=0)
            self.logger.info("Trainer step is successful!")
            self.logger.debug(f"Trainer steps. now on version {self.trainer.policy.version}.")
            self.buffer = []


def make_local_pair(q) -> Tuple[SampleProducer, SampleConsumer]:
    return LocalSampleProducer(q), LocalSampleConsumer(q)


register_producer(api.config.SampleStream.Type.INLINE_TESTING, InlineSampleProducer)
register_producer(api.config.SampleStream.Type.NULL, NullSampleProducer)
