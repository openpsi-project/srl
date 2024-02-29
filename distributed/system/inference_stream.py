"""This module defines the data flow between policy workers and actor workers.

In our design, actor workers are in charge of executing env.step() (typically simulation), while
policy workers running policy.rollout_step() (typically neural network inference). The inference
stream is the abstraction of the data flow between them: the actor workers send environment
observations as requests, and the policy workers return actions as responses, both plus other
additional information.
"""
from typing import Tuple, List, Optional, Any, Union
import binascii
import json
import logging
import numpy as np
import os
import pickle
import prometheus_client
import socket
import time
import threading
import zmq

from base.namedarray import recursive_aggregate, size_bytes
import api.config
import api.policy
import base.numpy_utils
import base.namedarray
import base.timeutil
import distributed.base.name_resolve as name_resolve
import base.network
import base.names as names
import base.shared_memory as shared_memory
import base.numpy_utils
import distributed.system.parameter_db as db

# TODO: Refactor inference streams
_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), '_experiments')
_ROLLOUT_REQUEST_RETRY_INTERVAL_SECONDS = 100.  # 100.
_INLINE_PASSIVE_PULL_FREQUENCY_SECONDS = 2
_INLINE_PULL_PARAMETER_ON_START = True

METRIC_INFERENCE_LATENCY_SECONDS_REMOTE = prometheus_client.Summary("marl_inference_latency_seconds", "",
                                                                    ["host", "experiment", "trial", "stream"])
METRIC_INFERENCE_LATENCY_SECONDS_LOCAL = prometheus_client.Summary("marl_inference_latency_seconds_local", "",
                                                                   ["host", "experiment", "trial", "stream"])

logger = logging.getLogger("InferenceStream")


class InferenceClient:
    """Interface used by the actor workers to obtain actions given current observation."""

    def post_request(self, request: api.policy.RolloutRequest, index=0) -> int:
        """Set the client_id and request_id of the request and cache the request.

        Args:
            request: RolloutRequest of length 1.
            index: index of the request in the shared memory buffer
        """
        raise NotImplementedError()

    def poll_responses(self):
        """Poll all responses from inference server. This method is considered thread unsafe and only called by the
        main process.
        """
        raise NotImplementedError()

    def is_ready(self, inference_ids: List[int], index) -> bool:
        """Check whether a specific request is ready to be consumed.

        Args:
            inference_ids: a list of requests to check

        Outputs:
            is_ready: whether the inference_ids are all ready.
        """
        raise NotImplementedError()

    def register_agent(self):
        return 0

    def consume_result(self, inference_ids: List[int], index):
        """Consume a result with specific request_id, returns un-pickled message.
        Raises KeyError if inference id is not ready. Make sure you call is_ready before consuming.

        Args:
            inference_ids: a list of requests to consume.

        Outputs:
            results: list of rollout_request.
        """
        raise NotImplementedError()

    def flush(self):
        """Send all cached inference requests to inference server.
        Implementations are considered thread-unsafe.
        """
        raise NotImplementedError()

    def get_constant(self, name: str) -> Any:
        """Retrieve the constant value saved by inference server.

        Args:
            name: name of the constant to get.

        Returns:
            value: the value set by inference server.
        """
        raise NotImplementedError()


class InferenceServer:
    """Interface used by the policy workers to serve inference requests."""

    def poll_requests(self) -> List[api.policy.RolloutRequest]:
        """Consumes all incoming requests.

        Returns:
            RequestPool: A list of requests, already batched by client.
        """
        raise NotImplementedError()

    def respond(self, response: api.policy.RolloutResult):
        """Send rollout results to inference clients.

        Args:
            response: rollout result to send.
        """
        raise NotImplementedError()

    def set_constant(self, name: str, value: Any):
        """Retrieve the constant value saved by inference server.

        Args:
            name: name of the constant to get.
            value: the value to be set, can be any object that can be pickled..
        """
        raise NotImplementedError()


class IpInferenceClient(InferenceClient):
    """Inference Client based on IP, Currently implemented in ZMQ.
    By calling client.post_request(msg, flush=False), the inference client will
    cache the the posted request. When client.flush() is called, the inference client will batch
    all the inference request and send to inference server.

    NOTE: Calling client.post_request(msg, flush=True) is discouraged. Sending scattered request may
    overwhelm the inference side.
    """

    def __init__(self, server_addresses: Union[List[str], str], serialization_method: str):
        """Init method of ip inference client.

        Args:
            address of on the server.
        """
        self.__addresses = server_addresses if isinstance(server_addresses, List) else [server_addresses]
        logger.debug(f"Client: connecting to servers {self.__addresses}")

        # Client_id is a randomly generated np.int32.
        self.client_id = np.random.randint(0, 2147483647)
        self.__context, self.__socket = self._make_sockets()
        self._request_count = 0
        self.__request_buffer = []

        self._response_cache = {}
        self._request_send_time = {}
        self._pending_requests = {}
        self._retried_requests = set()
        self._metric_inference_latency_seconds = None

        self.__req_id_generation_lock = threading.Lock()  # Lock self._request_count for safe id generation.
        # Lock self.__request_buffer for concurrent writing/ flushing.
        self.__request_buffer_lock = threading.Lock()
        # Locks self._pending_request and self._request_send_time
        self.__request_metadata_lock = threading.Lock()

        self.__retry_frequency_control = base.timeutil.FrequencyControl(frequency_seconds=5)
        self.__debug_frequency_control = base.timeutil.FrequencyControl(frequency_seconds=5)

        self.__serialization_method = serialization_method

        # debug
        self.first_time_req = True
        self.first_time_response = True

    def __del__(self):
        self.__socket.close(linger=False)
        self.__context.destroy(linger=False)

    def _make_sockets(self) -> Tuple[zmq.Context, zmq.Socket]:
        """Setup ZMQ socket for posting observations and receiving actions.
        Sockets are shared across multiple environments in the actor.

        Outbound message:
            [request]

        Inbound message:
            [b"client_id", msg]
        """
        ctx = zmq.Context()
        socket = ctx.socket(zmq.DEALER)
        socket.identity = self.client_id.to_bytes(length=4, byteorder="little")
        for addr in self.__addresses:
            socket.connect(f"tcp://{addr}")
        socket.setsockopt(zmq.LINGER, 0)
        return ctx, socket

    def __post_request(self, request) -> int:
        """Buffer a request and get a new request id (Thread Safe).
        """
        with self.__req_id_generation_lock:
            req_id = self._request_count
            self._request_count += 1

        logger.debug(f"Generated req_id {req_id}")
        self.__post_request_with_id(request, req_id)
        return req_id

    def __post_request_with_id(self, request, req_id):
        request.request_id = np.array([req_id], dtype=np.int64)
        request.client_id = np.array([self.client_id], dtype=np.int32)
        with self.__request_buffer_lock:
            self.__request_buffer.append(request)
        self._request_send_time[req_id] = time.monotonic_ns()
        self._pending_requests[req_id] = request

    def post_request(self, request: api.policy.RolloutRequest, index=0) -> int:
        return self.__post_request(request)

    def is_ready(self, inference_ids, index) -> bool:
        check_retry = self.__retry_frequency_control.check()  # If true, we retry expired requests.
        for req_id in inference_ids:
            if req_id not in list(self._response_cache.keys()):
                if check_retry:
                    with self.__request_metadata_lock:
                        expired = req_id in self._request_send_time.keys() and \
                                  (time.monotonic_ns() - self._request_send_time[req_id]) / 1e9 > _ROLLOUT_REQUEST_RETRY_INTERVAL_SECONDS
                        r = self._pending_requests[req_id]
                    if expired:
                        self.__post_request_with_id(r, req_id)
                        logger.info(f"Request with req_id {req_id} timed out. Retrying...")
                return False
        return True

    def consume_result(self, inference_ids, index):
        if self.__debug_frequency_control.check():
            logger.debug("Cached reqs", list(self._response_cache.keys()))
            logger.debug("Pending reqs", list(self._pending_requests.keys()))
        return [self._response_cache.pop(req_id) for req_id in inference_ids]

    def flush(self):
        if len(self.__request_buffer) > 0:
            with self.__request_buffer_lock:
                one_request = self.__request_buffer[0]
                agg_request = recursive_aggregate(self.__request_buffer, np.stack)
                self.__request_buffer = []
            self.__socket.send_multipart(
                base.namedarray.dumps(agg_request, method=self.__serialization_method))

            # if self.first_time_req:
            #     logger.info(f"One agged request size {size_bytes(agg_request)}, one request size {size_bytes(one_request)}")
            #     pickle.dump(agg_request, open("aggregated_request.pkl", "wb"))
            #     pickle.dump(one_request, open("one_request.pkl", "wb"))
            #     self.first_time_req = False

    def poll_responses(self):
        """Get all action messages from inference servers. Thread unsafe"""
        try:
            _msg = self.__socket.recv_multipart(zmq.NOBLOCK)
            responses = base.namedarray.loads(_msg)
            # if self.first_time_response:
            #     logger.info(f"Responses size {size_bytes(responses)}, len {responses.length(dim=0)}")
            #     pickle.dump(responses, open("aggregated_request.pkl", "wb"))
            #     self.first_time_response = False
            for i in range(responses.length(dim=0)):
                req_id = responses.request_id[i, 0]
                # with self.__consume_lock:
                if req_id in list(self._response_cache.keys()):
                    raise ValueError(
                        "receiving multiple result with request id {}."
                        "Have you specified different inferencer client_ids for actors?".format(req_id))
                else:
                    if req_id not in list(self._request_send_time.keys()):
                        if req_id in self._retried_requests:
                            logger.warning(f"Received multiple responses for request {req_id}. "
                                           f"Request {req_id} has been retried, ignoring this case.")
                            continue
                        # This is impossible.
                        # raise RuntimeError(f"Impossible case: got response but I didn't send it? {req_id}")
                        continue
                    with self.__request_metadata_lock:
                        latency = (time.monotonic_ns() - self._request_send_time.pop(req_id)) / 1e9
                        self._pending_requests.pop(req_id)
                    if self._metric_inference_latency_seconds is not None:
                        self._metric_inference_latency_seconds.observe(latency)
                    self._response_cache[req_id] = responses[i]
                    logger.debug(f"Response cache: {list(self._response_cache.keys())}")
            return responses.length(dim=0)
        except zmq.ZMQError:
            return 0
        except Exception as e:
            raise e

    def get_constant(self, name):
        raise NotImplementedError()


class IpInferenceServer(InferenceServer):
    """InferenceServer inited by IP, Currently implemented in ZMQ pub-sub pattern.
    Be careful only to initialize server within process, as ZMQ context cannot
    be inherited from one process to another
    """

    def __init__(self, address, serialization_method):
        """Init method of Ip inference server.
        Args:
            address: Address of Server
        """
        self.__context = zmq.Context()
        self.__socket = self.__context.socket(zmq.ROUTER)
        self.__socket.setsockopt(zmq.LINGER, 0)

        if address == '':
            host_ip = base.network.gethostip()
            port = self.__socket.bind_to_random_port(f"tcp://{host_ip}")
            address = f"{host_ip}:{port}"
        else:
            self.__socket.bind(f"tcp://{address}")

        logger.debug(f"Server: address {address}")

        self._address = address
        self.__serialization_method = serialization_method

    def __del__(self):
        self.__socket.close()
        self.__context.destroy(linger=False)

    def poll_requests(self, max_iter=64):
        request_batches = []
        for _ in range(max_iter):
            try:
                client_id_, *msg = self.__socket.recv_multipart(zmq.NOBLOCK)
                # Client id is recorded in msg itself, we don't actually need client_id_.
                try:
                    requests = base.namedarray.loads(msg)
                except pickle.UnpicklingError:
                    logger.info(f"Unpickling request failed. content: {msg}")
                    continue
                requests.received_time[:] = time.monotonic_ns()
                request_batches.append(requests)
            except zmq.ZMQError:
                break
        return request_batches

    def respond(self, responses: api.policy.RolloutResult):
        logger.debug(f"respond to req_ids: {responses.request_id}")
        idx = np.concatenate([[0],
                              np.where(np.diff(responses.client_id[:, 0]))[0] + 1, [responses.length(dim=0)]])
        for i in range(len(idx) - 1):
            self.__socket.send_multipart([
                int(responses.client_id[idx[i], 0]).to_bytes(length=4, byteorder="little"),
            ] + base.namedarray.dumps(responses[idx[i]:idx[i + 1]], method=self.__serialization_method))

    def set_constant(self, name, value):
        raise NotImplementedError()


class NameResolvingInferenceServer(IpInferenceServer):
    """Inference Server By name
    """

    def __init__(self, experiment_name, trial_name, stream_name, serialization_method):
        assert "/" not in experiment_name, "illegal character \"/\" in experiment name"
        assert "/" not in stream_name, "illegal character \"/\" in stream name"

        # Calling super class init will call __make_sockets, which in turn sets values to addresses.
        super().__init__("", serialization_method)
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name
        self.__stream_name = stream_name
        name_resolve.add_subentry(
            names.inference_stream(experiment_name=experiment_name,
                                   trial_name=trial_name,
                                   stream_name=stream_name),
            json.dumps({
                "address": self._address,
            }),
            keepalive_ttl=300,
        )

    def set_constant(self, name, value):
        """NOTE: Currently set/get constant are implemented for policy state. In other cases, use with caution,
        as values in name_resolve are not guaranteed to be unique or consistent.
        """
        name_resolve.add(
            names.inference_stream_constant(experiment_name=self.__experiment_name,
                                            trial_name=self.__trial_name,
                                            stream_name=self.__stream_name,
                                            constant_name=name),
            binascii.b2a_base64(pickle.dumps(value)).decode(),
            keepalive_ttl=30,
            replace=True,
        )


class NameResolvingInferenceClient(IpInferenceClient):
    """Inference Client by name. Client will try to find policy worker configs in the target directory.
    With that said, controller should always setup policy worker and trainer worker before setting up actor
    worker.
    """

    def __init__(self, experiment_name, trial_name, stream_name, rank, serialization_method):
        name = names.inference_stream(experiment_name=experiment_name,
                                      trial_name=trial_name,
                                      stream_name=stream_name)
        server_json_configs = list(sorted(name_resolve.get_subtree(name)))
        target_pw_rank = rank % len(server_json_configs)
        server_config = json.loads(server_json_configs[target_pw_rank])

        super().__init__(server_addresses=server_config['address'].replace("*", "localhost"),
                         serialization_method=serialization_method)
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name
        self.__stream_name = stream_name

        self._metric_inference_latency_seconds = METRIC_INFERENCE_LATENCY_SECONDS_REMOTE.labels(
            host=base.network.gethostname(), experiment=experiment_name, trial=trial_name, stream=stream_name)

    def get_constant(self, name):
        r = name_resolve.get(
            names.inference_stream_constant(experiment_name=self.__experiment_name,
                                            trial_name=self.__trial_name,
                                            stream_name=self.__stream_name,
                                            constant_name=name)).encode()
        return pickle.loads(binascii.a2b_base64(r))


class InlineInferenceClient(InferenceClient):

    def poll_responses(self):
        pass

    def __init__(self,
                 policy,
                 policy_name,
                 param_db,
                 worker_info,
                 pull_interval,
                 policy_identifier,
                 parameter_service_client=None,
                 foreign_policy=None,
                 accept_update_call=True,
                 population=None,
                 policy_sample_probs=None):
        self.policy_name = policy_name
        self.__policy_identifier = policy_identifier
        import os
        os.environ["MARL_CUDA_DEVICES"] = "cpu"
        self.policy = api.policy.make(policy)
        self.policy.eval_mode()
        self.__logger = logging.getLogger("Inline Inference")
        self._request_count = 0
        self.__request_buffer = []
        self._response_cache = {}
        self.__pull_freq_control = base.timeutil.FrequencyControl(
            frequency_seconds=pull_interval, initial_value=_INLINE_PULL_PARAMETER_ON_START)
        self.__passive_pull_freq_control = base.timeutil.FrequencyControl(
            frequency_seconds=_INLINE_PASSIVE_PULL_FREQUENCY_SECONDS,
            initial_value=_INLINE_PULL_PARAMETER_ON_START,
        )  # TODO: make it configurable?
        self.__load_absolute_path = None
        self.__accept_update_call = accept_update_call
        self.__pull_fail_count = 0
        self.__max_pull_fails = 1000
        self.__parameter_service_client = None

        # Parameter DB / Policy name related.
        if foreign_policy is not None:
            p = foreign_policy
            i = worker_info
            pseudo_worker_info = api.config.WorkerInformation(experiment_name=p.foreign_experiment_name
                                                              or i.experiment_name,
                                                              trial_name=p.foreign_trial_name or i.trial_name)
            self.__param_db = db.make_db(p.param_db, worker_info=pseudo_worker_info)
            self.__load_absolute_path = p.absolute_path
            self.__load_policy_name = p.foreign_policy_name or policy_name
            self.__policy_identifier = p.foreign_policy_identifier or policy_identifier
        else:
            self.__param_db = db.make_db(param_db, worker_info=worker_info)
            self.__load_policy_name = policy_name
            self.__policy_identifier = policy_identifier

        if parameter_service_client is not None and self.__load_absolute_path is None:
            self.__parameter_service_client = db.make_client(parameter_service_client, worker_info)
            self.__parameter_service_client.subscribe(experiment_name=self.__param_db.experiment_name,
                                                      trial_name=self.__param_db.trial_name,
                                                      policy_name=self.__load_policy_name,
                                                      tag=self.__policy_identifier,
                                                      callback_fn=self.policy.load_checkpoint,
                                                      use_current_thread=True)

        self.configure_population(population, policy_sample_probs)

        self.__log_frequency_control = base.timeutil.FrequencyControl(frequency_seconds=10)
        self._metric_inference_latency_seconds = METRIC_INFERENCE_LATENCY_SECONDS_LOCAL.labels(
            host=base.network.gethostname(),
            experiment=worker_info.experiment_name,
            trial=worker_info.trial_name,
            stream=policy_name)

    def configure_population(self, population, policy_sample_probs):
        if population is not None:
            assert policy_sample_probs is None or len(policy_sample_probs) == len(population), (
                f"Size of policy_sample_probs {len(policy_sample_probs)} and population {len(population)} must be the same."
            )
            self.__population = population
            if policy_sample_probs is None:
                policy_sample_probs = np.ones(len(population)) / len(population)
            self.__policy_sample_probs = policy_sample_probs
        elif self.policy_name is None:
            policy_names = self.__param_db.list_names()
            if len(policy_names) == 0:
                raise ValueError(
                    "You set policy_name and population to be None, but no existing policies were found.")
            logger.info(f"Auto-detected population {policy_names}")
            self.__population = policy_names
            self.__policy_sample_probs = np.ones(len(policy_names)) / len(policy_names)
        else:
            self.__population = None
            self.__policy_sample_probs = None

    def post_request(self, request: api.policy.RolloutRequest, index) -> int:
        request.request_id = np.array([self._request_count], dtype=np.int64)
        req_id = self._request_count
        self.__request_buffer.append(request)
        self._request_count += 1
        self.flush()
        return req_id

    def is_ready(self, inference_ids: List[int], index) -> bool:
        for req_id in inference_ids:
            if req_id not in list(self._response_cache.keys()):
                return False
        return True

    def consume_result(self, inference_ids: List[int], index):
        return [self._response_cache.pop(req_id) for req_id in inference_ids]

    def load_parameter(self):
        """Method exposed to Actor worker so we can reload parameter when env is done.
        """
        if self.__passive_pull_freq_control.check() and self.__accept_update_call:
            # This reduces the unnecessary workload of mongodb.
            self.__load_parameter()

    def __get_checkpoint_from_db(self, block=False):
        if self.__load_absolute_path is not None:
            return self.__param_db.get_file(self.__load_absolute_path)
        else:
            return self.__param_db.get(name=self.__load_policy_name,
                                       identifier=self.__policy_identifier,
                                       block=block)

    def __load_parameter(self):
        if self.__population is None:
            policy_name = self.policy_name
        else:
            policy_name = np.random.choice(self.__population, p=self.__policy_sample_probs)
        checkpoint = self.__get_checkpoint_from_db(block=self.policy.version < 0)
        self.policy.load_checkpoint(checkpoint)
        self.policy_name = policy_name
        self.__logger.debug(f"Loaded {self.policy_name}'s parameter of version {self.policy.version}")

    def flush(self):
        start = time.monotonic_ns()
        if self.__pull_freq_control.check():
            self.__load_parameter()

        if self.__parameter_service_client is not None:
            self.__parameter_service_client.poll()

        if self.__log_frequency_control.check():
            self.__logger.debug(f"Policy Version: {self.policy.version}")

        if len(self.__request_buffer) > 0:
            # self.__logger.debug("Inferencing")
            agg_req = recursive_aggregate(self.__request_buffer, np.stack)
            rollout_results = self.policy.rollout(agg_req)
            rollout_results.request_id = agg_req.request_id
            rollout_results.policy_version_steps = np.full(shape=agg_req.client_id.shape,
                                                           fill_value=self.policy.version)
            rollout_results.policy_name = np.full(shape=agg_req.client_id.shape, fill_value=self.policy_name)
            self.__request_buffer = []
            for i in range(rollout_results.length(dim=0)):
                self._response_cache[rollout_results.request_id[i, 0]] = rollout_results[i]
        self._metric_inference_latency_seconds.observe((time.monotonic_ns() - start) / 1e9)

    def get_constant(self, name: str) -> Any:
        if name == "default_policy_state":
            return self.policy.default_policy_state
        else:
            raise NotImplementedError(name)


class ZippedInferenceClients(InferenceClient):
    # TODO : The use case of zipped inference is not clear. Keeping it for consistency with sample stream.

    def __init__(self, inference_clients):
        self.__clients = inference_clients

    def post_request(self, request: api.policy.RolloutRequest, index=0) -> int:
        raise NotImplementedError()

    def is_ready(self, inference_ids: List[int], index) -> bool:
        raise NotImplementedError()

    def consume_result(self, inference_ids: List[int], index):
        raise NotImplementedError()

    def flush(self):
        raise NotImplementedError("Flushing should be called on standalone inference clients (not zipped).")

    def get_constant(self, name: str) -> Any:
        raise NotImplementedError()


class PinnedSharedMemoryInferenceClient(InferenceClient):
    """ Inference client that uses shared memory to communicate with the server.
    """

    def __init__(self, experiment_name, trial_name, stream_name, lock):
        """Init method of ip inference client.

        Args:
            address of on the server.
        """
        self.request_lock = lock[0]
        self.response_lock = lock[1]

        self.__experiment_name = experiment_name
        self.__trial_name = trial_name
        self.__stream_name = stream_name
        self.__qsize = 0

        # connect to inference server (shared memory dock server)
        self.__rpc_client = shared_memory.SharedMemoryRpcClient(experiment_name, trial_name, "inference",
                                                                stream_name)

        # Client_id is a randomly generated non-zero np.int32
        self.client_id = np.random.randint(1, 2147483647)
        logger.info(f"Generated client id {self.client_id} for shared memory inference client.")
        self._request_count = 0
        self.__request_buffer = []
        self.__index_buffer = []
        self.__all_indices = []
        self.__all_indices_sorted = False

        self.__req_id_generation_lock = threading.Lock()  # Lock self._request_count for safe id generation.
        # Lock self.__request_buffer for concurrent writing/ flushing.
        self.__request_buffer_lock = threading.Lock()

        self._request_dock = None
        self._request_dock_name = names.shared_memory(experiment_name, trial_name, stream_name + "_request")
        self._response_dock = None
        self._response_dock_name = names.shared_memory(experiment_name, trial_name, stream_name + "_response")

        self._shm_name = self._request_dock_name.strip("/").replace("/", "_")

    def __post_request_with_id(self, request, req_id, index):
        request.request_id = np.array([req_id], dtype=np.int32)
        request.client_id = np.array([self.client_id], dtype=np.int32)
        request.buffer_index = np.array([index], dtype=np.int32)
        request.ready = np.array([True], dtype=np.bool8)
        with self.__request_buffer_lock:
            self.__request_buffer.append(request)
            self.__index_buffer.append(index)

    def post_request(self, request: api.policy.RolloutRequest, index=0) -> int:
        """Buffer a request and get a new request id (Thread Safe).
        Index is used to identify which agent is posting the request.
        """
        with self.__req_id_generation_lock:
            req_id = self._request_count
            self._request_count += 1

        logger.debug(f"Generated req_id {req_id}")
        self.__post_request_with_id(request, req_id, index)
        return req_id

    def is_ready(self, inf_id, inf_index) -> bool:
        """Get all action messages from inference servers. Thread unsafe"""
        if self._response_dock is None:
            self.__qsize = self.__rpc_client.call("get_qsize", dict())
            self._response_dock = shared_memory.reader_make_shared_memory_dock(self.__qsize,
                                                                               self.__experiment_name,
                                                                               self.__trial_name,
                                                                               self.__stream_name +
                                                                               "_response",
                                                                               second_dim_index=False,
                                                                               timeout=1)
        if self._response_dock is None:
            return False

        with self.response_lock.client_locked():
            return self._response_dock.get_key("ready", inf_index[0])[0]

    def consume_result(self, inference_ids, inf_index):
        return [self._response_dock.get(inf_index[0])]

    def register_agent(self):
        # Called in worker configuration
        index = self.__rpc_client.call("register_agent", dict())
        self.__all_indices.append(index)
        return index

    def flush(self):
        if len(self.__request_buffer) > 0:
            if self._request_dock is None:
                self.__qsize = self.__rpc_client.call("get_qsize", dict())
                self._request_dock = shared_memory.writer_make_shared_memory_dock(self.__request_buffer[0],
                                                                                  self.__qsize,
                                                                                  self.__experiment_name,
                                                                                  self.__trial_name,
                                                                                  self.__stream_name +
                                                                                  "_request",
                                                                                  second_dim_index=False)

            with self.__request_buffer_lock:
                indices = self.__index_buffer
                agg_request = recursive_aggregate(self.__request_buffer, np.stack)
                self.__request_buffer = []
                self.__index_buffer = []

            assert (np.array(indices) == agg_request.buffer_index[:, 0]
                    ).all(), f"Put indices do not match {indices}, {agg_request.buffer_index[:, 0]}."
            #### This should locked from `poll_requests` of inference server.
            with self.request_lock.client_locked():
                # print("flush acquire client")
                # self.request_lock.acquire_client()
                self._request_dock.put(indices, agg_request, sort=False)
                # print("flush release client")
                # self.request_lock.release_client()
            ####

    def poll_responses(self):
        pass
        # """Get all action messages from inference servers. Thread unsafe"""
        # if self._response_dock is None:
        #     self.__qsize = self.__rpc_client.call("get_qsize", dict())
        #     self._response_dock = shared_memory.reader_make_shared_memory_dock(self.__qsize,
        #                                                                        self.__experiment_name,
        #                                                                        self.__trial_name,
        #                                                                        self.__stream_name+"_response",
        #                                                                        second_dim_index=False,
        #                                                                        timeout=1)
        # if self._response_dock is None:
        #     return 0

        # if not self.__all_indices_sorted:
        #     self.__all_indices = np.sort(self.__all_indices)
        #     # logger.info(f"DEBUG: My client id {self.client_id}, My indices {len(self.__all_indices)} {self.__all_indices[0]} {self.__all_indices[-1]}, {self.__all_indices}")
        #     self.__all_indices_sorted = True

        # #### this should be locked by a by a response lock that locks out the `respond` method of inference server

        # st = time.monotonic()
        # with self.response_lock.client_locked():
        #     t0 = time.monotonic()
        #     all_responses_ready = self._response_dock.get_key("ready", self.__all_indices)
        #     # Get whether ready responses indices of this inference client
        #     t1 = time.monotonic()
        #     ready_indices = self.__all_indices[np.where(all_responses_ready)[0]]
        #     t2 = time.monotonic()
        #     if len(ready_indices) == 0:
        #         # print("poll response release client")
        #         return 0 # No ready responses
        #     self._response_dock.put_key("ready", ready_indices, np.array([[False]] * len(ready_indices), dtype=np.bool8))
        #     t3 = time.monotonic()

        # responses = self._response_dock.get(ready_indices)
        # # logger.info(f"Got responses request ids {self.client_id}, indices {ready_indices}: {responses.request_id[:, 0]}")
        # ####
        # assert (responses.client_id[:,0] == self.client_id).all(), f"Received responses from other clients, {ready_indices}, {responses.client_id[:,0]}"

        # for i in range(responses.length(dim=0)):
        #     req_id = responses.request_id[i, 0]
        #     # with self.__consume_lock:
        #     if req_id in list(self._response_cache.keys()):
        #         raise ValueError(
        #             "receiving multiple result with request id {}."
        #             "Have you specified different inferencer client_ids for actors?".format(req_id))
        #     else:
        #         if req_id not in list(self._request_send_time.keys()):
        #             if req_id in self._retried_requests:
        #                 logger.warning(f"Received multiple responses for request {req_id}. "
        #                                 f"Request {req_id} has been retried, ignoring this case.")
        #                 continue
        #             # This is impossible.
        #             # raise RuntimeError(f"Impossible case: got response but I didn't send it? {req_id}")
        #             raise RuntimeError(f"Impossible case: got response but I didn't send it? {req_id}.\n"
        #                                f"My request count: {self._request_count}\n"
        #                                f"My sent request ids: {list(self._request_send_time.keys())}\n"
        #                                f"My pending request ids: {list(self._pending_requests.keys())}\n"
        #                                f"My client ids: {self.client_id}\n"
        #                                f"This request index: {responses.buffer_index[i, 0]}\n"
        #                                f"This request id: {responses.request_id[i, 0]}\n"
        #                                f"This batch request id: {responses.request_id[:, 0]}\n"
        #                                f"This batch indices from responses: {responses.buffer_index[:, 0]}\n"
        #                                f"This batch indices: {ready_indices}\n"
        #                                f"This client all indices: {self.__all_indices}\n"
        #                                f"This request client id: {responses.client_id[i, 0]}\n"
        #                                f"This batch client id: {responses.client_id[:,0]}\n"
        #                                f"Assertion: {(responses.client_id[:,0] == self.client_id)} \n"
        #                                f"Assertion.all: {(responses.client_id[:,0] == self.client_id).all()} \n"
        #                                f"Responses base: {responses.client_id.base}\n"
        #                                f"This batch ready: {responses.ready[:,0]}\n")
        #         with self.__request_metadata_lock:
        #             latency = (time.monotonic_ns() - self._request_send_time.pop(req_id)) / 1e9
        #             self._pending_requests.pop(req_id)
        #         if self._metric_inference_latency_seconds is not None:
        #             self._metric_inference_latency_seconds.observe(latency)
        #         self._response_cache[req_id] = responses[i]
        #         logger.debug(f"Response cache: {list(self._response_cache.keys())}")

        # return responses.length(dim=0)

    def get_constant(self, name):
        r = name_resolve.get(
            names.inference_stream_constant(experiment_name=self.__experiment_name,
                                            trial_name=self.__trial_name,
                                            stream_name=self.__stream_name,
                                            constant_name=name)).encode()
        return pickle.loads(binascii.a2b_base64(r))


class PinnedSharedMemoryInferenceServer(InferenceServer):
    # dock server, reader, writer
    def __init__(self, experiment_name, trial_name, stream_name, lock):
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name
        self.__stream_name = stream_name

        self.request_lock = lock[0]
        self.response_lock = lock[1]

        self.__rpc_client = shared_memory.SharedMemoryRpcClient(experiment_name, trial_name, "inference",
                                                                stream_name)

        self.__qsize = 0

        self._request_dock = None
        self._request_dock_name = names.shared_memory(experiment_name, trial_name, stream_name + "_request")

        self._response_dock = None
        self._response_dock_name = names.shared_memory(experiment_name, trial_name, stream_name + "_response")

        self._shm_name = self._response_dock_name.strip("/").replace("/", "_")

        # debug
        self.__client_id_to_indices = {}

    def poll_requests(self, batch_size=None):
        """ Called by policy worker. Read all ready requests in request shared memory dock.
        Set received time.
        """
        if self._request_dock is None:
            # Initialize request shared memory dock
            self.__qsize = self.__rpc_client.call("get_qsize", dict())
            self._request_dock = shared_memory.reader_make_shared_memory_dock(self.__qsize,
                                                                              self.__experiment_name,
                                                                              self.__trial_name,
                                                                              self.__stream_name + "_request",
                                                                              second_dim_index=False,
                                                                              timeout=1)
        if self._request_dock is None:
            return []

        #### This should lock out all clients from calling flush()
        # Check all request shared memory block
        # logger.info("poll requests acquire server")
        # self.request_lock.acquire_server()

        with self.request_lock.server_locked():
            all_ready = self._request_dock.get_key("ready")
            # Find all ready indices
            ready_indices = np.where(all_ready)[0]
            if batch_size is None:
                if len(ready_indices) == 0:
                    # print("poll requests release server")
                    return []
            elif len(ready_indices) < batch_size:
                return []

            self._request_dock.put_key("ready", ready_indices,
                                       np.array([[False]] * len(ready_indices), dtype=np.bool8))
            # logger.info("poll requests release server")
            # self.request_lock.release_server()
            ####

        #### lock write do not lock read
        batch = self._request_dock.get(ready_indices)
        # logger.info(f"Got requests {indices}, {batch.request_id[:,0]}, {batch.client_id[:,0]}, {batch.buffer_index[:,0]}")
        batch.received_time[:] = time.monotonic_ns()
        return [batch]

    def respond(self, responses: api.policy.RolloutResult):
        """ Respond with an aggregated rollout result, write them into response shared memory dock.
        """

        if self._response_dock is None:
            self.__qsize = self.__rpc_client.call("get_qsize", dict())
            logger.info("Response dock writer creating.")
            # Initialize response shared memory dock
            # Here we use the first entry of input responses as shared memory dock entry example.
            self._response_dock = shared_memory.writer_make_shared_memory_dock(responses[0],
                                                                               self.__qsize,
                                                                               self.__experiment_name,
                                                                               self.__trial_name,
                                                                               self.__stream_name +
                                                                               "_response",
                                                                               second_dim_index=False)
            logger.info("Response dock created by server respond.")

        ####
        # logger.info("respond acquire server")
        # self.response_lock.acquire_server()
        # batch_size = responses.length(dim=0)
        # responses.ready = np.array([[True]] * batch_size)
        with self.response_lock.server_locked():
            indices = responses.buffer_index[:, 0]
            # logger.info(f"Respond put {indices}, {responses.request_id[:, 0]}")

            self._response_dock.put(indices, responses, sort=False)
        # logger.info("respond release server")
        # self.response_lock.release_server()
        ####

    def set_constant(self, name, value):
        name_resolve.add(
            names.inference_stream_constant(experiment_name=self.__experiment_name,
                                            trial_name=self.__trial_name,
                                            stream_name=self.__stream_name,
                                            constant_name=name),
            binascii.b2a_base64(pickle.dumps(value)).decode(),
            keepalive_ttl=30,
            replace=True,
        )


def make_server(spec: Union[str, api.config.InferenceStream],
                worker_info: Optional[api.config.WorkerInformation] = None,
                lock=None):
    """Initializes an inference stream server.

    Args:
        spec: Inference stream specification.
        worker_info: The server worker information.
    """
    if isinstance(spec, str):
        spec = api.config.InferenceStream(type_=api.config.InferenceStream.Type.NAME, stream_name=spec)
    if spec.type_ == api.config.InferenceStream.Type.IP:
        server = IpInferenceServer(address=spec.address, serialization_method=spec.serialization_method)
    elif spec.type_ == api.config.InferenceStream.Type.NAME:
        server = NameResolvingInferenceServer(experiment_name=worker_info.experiment_name,
                                              trial_name=worker_info.trial_name,
                                              stream_name=spec.stream_name,
                                              serialization_method=spec.serialization_method)
    elif spec.type_ == api.config.InferenceStream.Type.SHARED_MEMORY:
        server = PinnedSharedMemoryInferenceServer(
            experiment_name=worker_info.experiment_name,
            trial_name=worker_info.trial_name,
            stream_name=spec.stream_name,
            lock=[lock[spec.lock_index * 2], lock[spec.lock_index * 2 + 1]])
    else:
        raise NotImplementedError(spec.type_)
    return server


def make_client(spec: Union[str, api.config.InferenceStream],
                worker_info: Optional[api.config.WorkerInformation] = None,
                lock=None):
    """Initializes an inference stream client.

    Args:
        spec: Inference stream specification.
        worker_info: The client worker information.
    """
    if isinstance(spec, str):
        spec = api.config.InferenceStream(type_=api.config.InferenceStream.Type.NAME, stream_name=spec)
    if spec.type_ == api.config.InferenceStream.Type.IP:
        client = IpInferenceClient(server_addresses=spec.address,
                                   serialization_method=spec.serialization_method)
    elif spec.type_ == api.config.InferenceStream.Type.NAME:
        client = NameResolvingInferenceClient(experiment_name=worker_info.experiment_name,
                                              trial_name=worker_info.trial_name,
                                              stream_name=spec.stream_name,
                                              rank=worker_info.worker_index,
                                              serialization_method=spec.serialization_method)
    elif spec.type_ == api.config.InferenceStream.Type.INLINE:
        client = InlineInferenceClient(policy=spec.policy,
                                       policy_name=spec.policy_name,
                                       param_db=spec.param_db,
                                       worker_info=worker_info,
                                       pull_interval=spec.pull_interval_seconds,
                                       policy_identifier=spec.policy_identifier,
                                       foreign_policy=spec.foreign_policy,
                                       accept_update_call=spec.accept_update_call,
                                       population=spec.population,
                                       parameter_service_client=spec.parameter_service_client,
                                       policy_sample_probs=spec.policy_sample_probs)
    elif spec.type_ == api.config.InferenceStream.Type.SHARED_MEMORY:
        client = PinnedSharedMemoryInferenceClient(
            experiment_name=worker_info.experiment_name,
            trial_name=worker_info.trial_name,
            stream_name=spec.stream_name,
            lock=[lock[spec.lock_index * 2], lock[spec.lock_index * 2 + 1]])
    else:
        raise NotImplementedError(spec.type_)
    return client


def zip_clients(inference_clients: List[InferenceClient]):
    return ZippedInferenceClients(inference_clients)
