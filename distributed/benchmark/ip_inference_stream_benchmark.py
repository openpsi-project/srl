import argparse
import logging
import numpy as np
import threading
import time

from base.namedarray import recursive_aggregate
from base.testing import get_testing_port
from distributed.system.inference_stream import make_client, make_server
import api.config as config
import api.environment as environment
import api.policy as policy

logger = logging.getLogger("inference stream benchmark")


class IpInferenceStreamBenchmark:

    def __init__(self, client_num=1, run_time=1, obs_shape=(10,), batch_size=1):
        self.client_num = client_num
        self.run_time = run_time
        self.batch_size = batch_size
        self.obs_shape = obs_shape
        self.port1 = get_testing_port()
        self.port2 = get_testing_port()
        self._all_latency = []

    def rollout_request(self):
        return policy.RolloutRequest(obs=np.random.random(self.obs_shape).astype(np.float32))

    def rollout_result(self, client_id, request_id, received_time):
        return policy.RolloutResult(action=environment.Action(),
                                    client_id=client_id,
                                    request_id=request_id,
                                    received_time=received_time)

    def client_func(self, start_event):
        client = make_client(spec=config.InferenceStream(type_=config.InferenceStream.Type.IP,
                                                         stream_name="benchmark",
                                                         address=f"localhost:{self.port1}",
                                                         address1=f"localhost:{self.port2}"))
        start_event.wait()
        logger.debug(f'client thread {threading.current_thread().ident} start requesting...')

        count = 0
        latency = []
        start_time = time.time()
        while True:
            requests = []
            for _ in range(self.batch_size):
                req_id = client.post_request(request=self.rollout_request(), flush=False)
                requests.append(req_id)
            send_time = time.monotonic_ns()
            client.flush()
            count += self.batch_size
            while True:
                if client.is_ready(requests):
                    latency_time = (time.monotonic_ns() - send_time) / 1e6
                    latency.append(latency_time)
                    self._all_latency.append(latency_time)
                    client.consume_result(inference_ids=requests)
                    break
            elapsed = time.time() - start_time
            if elapsed > self.run_time:
                break
        end_time = time.time()
        logger.debug(f'client {threading.current_thread().ident} posted {count} requests in '
                     f'{end_time - start_time:.3f} second(s).')
        logger.debug(
            f'client {threading.current_thread().ident} average latency: {sum(latency) / len(latency):.3f} ms'
        )

    def server_func(self, start_event, stop_event):
        server = make_server(spec=config.InferenceStream(type_=config.InferenceStream.Type.IP,
                                                         stream_name="benchmark",
                                                         address=f"*:{self.port1}",
                                                         address1=f"*:{self.port2}"))
        start_event.wait()
        logger.debug(f'server thread {threading.current_thread().ident} start responsing...')

        respond_count = 0
        start_time = time.time()
        while not stop_event.isSet():
            requests = server.poll_requests()
            if len(requests) > 0:
                responses = []
                for req in requests:
                    for i in range(req.length(dim=0)):
                        res = self.rollout_result(client_id=req.client_id[i],
                                                  request_id=req.request_id[i],
                                                  received_time=req.received_time[i])
                        responses.append(res)
                respond_count += len(responses)
                server.respond(recursive_aggregate(responses, np.stack))
        end_time = time.time()
        logger.info(f'server {threading.current_thread().ident} posted {respond_count} responses'
                    f' in {end_time - start_time:.3f} secs')
        logger.info(f'server throughput: {respond_count / (end_time - start_time):.3f} responses/sec')

    def multi_thread_benchmark(self):
        logger.info('ip inference stream benchmark (multi_thread).')
        logger.info(f'{self.client_num} client(s), 1 server, run time: {self.run_time} seconds.')

        start_event = threading.Event()
        stop_event = threading.Event()
        server_thread = threading.Thread(target=self.server_func,
                                         args=(
                                             start_event,
                                             stop_event,
                                         ),
                                         daemon=True)
        server_thread.start()
        client_threads = []
        for _ in range(self.client_num):
            t = threading.Thread(target=self.client_func, args=(start_event,))
            client_threads.append(t)
            t.start()
        start_event.set()

        for t in client_threads:
            t.join()
        logger.info(f'client average latency: {sum(self._all_latency) / len(self._all_latency):.3f} ms')
        stop_event.set()
        server_thread.join()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=[
        'ip',
    ], help='type of benchmark(ip,)')
    parser.add_argument('--mode', default='local', choices=[
        'local',
    ], help='mode of benchmark(local,)')
    parser.add_argument('-cn',
                        '--client_num',
                        default=1,
                        type=int,
                        help='the number of clients, 1 by default')
    parser.add_argument('-rt',
                        '--run_time',
                        default=1,
                        type=float,
                        help='run time(sec) of clients, 1 sec by default')
    parser.add_argument('-os',
                        '--obs_shape',
                        default=[
                            10,
                        ],
                        type=int,
                        nargs='+',
                        help='shape of RolloutRequest.obs, (10,) by default')
    parser.add_argument('-l',
                        '--log_level',
                        choices=['critical', 'error', 'warning', 'info', 'debug'],
                        default='info',
                        help='level of log(critical, error, warning, info, debug)')
    parser.add_argument('-bs', '--batch_size', default=1, type=int, help='client batch size')
    return parser.parse_args()


def set_log_level(log_level):
    log_levels = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }
    logging.basicConfig(level=log_levels[log_level])


def start_benchmark(args):
    benchmark = IpInferenceStreamBenchmark(client_num=args.client_num,
                                           run_time=args.run_time,
                                           obs_shape=tuple(args.obs_shape),
                                           batch_size=args.batch_size)
    if args.mode == 'local':
        benchmark.multi_thread_benchmark()


if __name__ == '__main__':
    arguments = parse_args()
    set_log_level(arguments.log_level)
    start_benchmark(arguments)
