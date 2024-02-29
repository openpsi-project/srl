import argparse
import logging
import numpy as np
import queue
import threading
import time

from api.trainer import SampleBatch
from base.namedarray import recursive_aggregate
from base.testing import get_testing_port
from distributed.system.sample_stream import make_producer, make_consumer
import api.config as config

logger = logging.getLogger("sample stream benchmark")


class IpSampleStreamBenchmark:

    def __init__(
            self,
            producer_num=1,
            run_time=1,
            obs_shape=(10,),
    ):
        self.producer_num = producer_num
        self.run_time = run_time
        self.obs_shape = obs_shape
        self.port = get_testing_port()

    def sample_batch(self, sample_steps, version=0):
        return recursive_aggregate([
            SampleBatch(
                obs=np.random.random(self.obs_shape).astype(np.float32),
                policy_state=np.random.random((2, 2)).astype(np.float32),
                on_reset=np.array([False], dtype=np.uint8),
                action=np.array([np.random.randint(19)]).astype(np.int32),
                log_probs=np.random.random(1,).astype(np.int32),
                reward=np.array([0], dtype=np.float32).astype(np.int32),
                info=np.random.randint(0, 2, (1,)),
                policy_version_steps=np.array([version], dtype=np.int64),
            ) for _ in range(sample_steps)
        ], np.stack)

    def producer_func(self, start_event):
        producer = make_producer(
            config.SampleStream(type_=config.SampleStream.Type.IP, address=f"localhost:{self.port}"))

        start_event.wait()
        logger.debug(f'producer thread {threading.current_thread().ident} start posting...')

        count = 0
        start_time = time.time()
        while True:
            producer.post(self.sample_batch(5))
            count += 1
            elapsed = time.time() - start_time
            if elapsed > self.run_time:
                break
        end_time = time.time()
        logger.debug(
            f'thread {threading.current_thread().ident} posted {count} samples in {end_time - start_time:.3f} secs.'
        )

    def consumer_func(self, start_event):
        consumer = make_consumer(
            config.SampleStream(type_=config.SampleStream.Type.IP, address=f"*:{self.port}"))
        buffer = queue.Queue()

        start_event.wait()
        logger.debug(f'consumer thread {threading.current_thread().ident} start consuming...')

        sample_count = 0
        start_time = time.time()
        while True:
            sample_count += consumer.consume_to(buffer)
            elapsed = time.time() - start_time
            if elapsed > self.run_time:
                break
        end_time = time.time()
        logger.info(f'consumer cosumes {sample_count} samples in {end_time - start_time:.3f} secs totally.')
        logger.info(f'consumer throughput：{sample_count / (end_time - start_time):.3f} samples/sec')

    def multi_thread_benchmark(self):
        logger.info('ip sample stream benchmark (multi_thread)')
        logger.info(
            f'{self.producer_num} producers, run time: {self.run_time} seconds , obs shape: {self.obs_shape}')

        start_event = threading.Event()
        consumer_thread = threading.Thread(target=self.consumer_func, args=(start_event,))
        consumer_thread.start()
        producers_thread = []
        for _ in range(self.producer_num):
            t = threading.Thread(target=self.producer_func, args=(start_event,))
            producers_thread.append(t)
            t.start()
        start_event.set()

        consumer_thread.join()
        for t in producers_thread:
            t.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=[
        'ip',
    ], help='type of benchmark(ip,)')
    parser.add_argument('--mode', default='local', choices=[
        'local',
    ], help='mode of benchmark(local,)')
    parser.add_argument('-pn',
                        '--producer_num',
                        default=1,
                        type=int,
                        help='the number of producers, 1 by default')
    parser.add_argument('-rt',
                        '--run_time',
                        default=1,
                        type=float,
                        help='run time(sec) of producers, 1 sec by default')
    parser.add_argument('-os',
                        '--obs_shape',
                        default=[
                            10,
                        ],
                        type=int,
                        nargs='+',
                        help='shape of SampleBatch.obs, (10,) by default')
    parser.add_argument('-l',
                        '--log_level',
                        choices=['critical', 'error', 'warning', 'info', 'debug'],
                        default='info',
                        help='level of log(critical, error, warning, info, debug)')

    args = parser.parse_args()
    log_levels = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }
    logging.basicConfig(level=log_levels[args.log_level])

    benchmark = IpSampleStreamBenchmark(producer_num=args.producer_num,
                                        run_time=args.run_time,
                                        obs_shape=tuple(args.obs_shape))
    if args.mode == 'local':
        benchmark.multi_thread_benchmark()
