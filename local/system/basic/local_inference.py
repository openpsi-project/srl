"""This module defines the data flow between policy workers and actor workers.

In our design, actor workers are in charge of executing env.step() (typically simulation), while
policy workers running policy.rollout_step() (typically neural network inference). The inference
stream is the abstraction of the data flow between them: the actor workers send environment
observations as requests, and the policy workers return actions as responses, both plus other
additional information.
"""

from typing import List, Any
import logging
import numpy as np
import queue
import time

import prometheus_client

from base.namedarray import recursive_aggregate
from local.system.inference_stream import InferenceClient, InferenceServer, register_client
import api.policy
import api.config
import base.network
import base.timeutil
import local.system.parameter_db as db

_INLINE_PASSIVE_PULL_FREQUENCY_SECONDS = 2
_INLINE_PULL_PARAMETER_ON_START = True

logger = logging.getLogger("InferenceStream")

METRIC_INFERENCE_LATENCY_SECONDS_LOCAL = prometheus_client.Summary("marl_inference_latency_seconds_local", "",
                                                                   ["host", "experiment", "trial", "stream"])


class InlineInferenceClient(InferenceClient):

    @property
    def type(self):
        return api.config.InferenceStream.Type.INLINE

    def __init__(self, spec: api.config.InferenceStream):
        self.policy_name = spec.policy_name
        self.__policy_identifier = spec.policy_identifier
        import os
        os.environ["MARL_CUDA_DEVICES"] = "cpu"
        self.policy = api.policy.make(spec.policy)
        self.policy.eval_mode()
        self.__logger = logging.getLogger("Inline Inference")
        self._request_count = 0
        self.__request_buffer = []
        self._response_cache = {}
        self.__pull_freq_control = base.timeutil.FrequencyControl(
            frequency_seconds=spec.pull_interval_seconds, initial_value=_INLINE_PULL_PARAMETER_ON_START)
        self.__passive_pull_freq_control = base.timeutil.FrequencyControl(
            frequency_seconds=_INLINE_PASSIVE_PULL_FREQUENCY_SECONDS,
            initial_value=_INLINE_PULL_PARAMETER_ON_START,
        )  # TODO: make it configurable?
        self.__load_absolute_path = None
        self.__accept_update_call = spec.accept_update_call
        self.__pull_fail_count = 0
        self.__max_pull_fails = 1000
        self.__parameter_service_client = None

        # Parameter DB / Policy name related.
        if spec.foreign_policy is not None:
            p = spec.foreign_policy
            i = spec.worker_info
            pseudo_worker_info = api.config.WorkerInformation(experiment_name=p.foreign_experiment_name
                                                              or i.experiment_name,
                                                              trial_name=p.foreign_trial_name or i.trial_name)
            self.__param_db = db.make_db(p.param_db, worker_info=pseudo_worker_info)
            self.__load_absolute_path = p.absolute_path
            self.__load_policy_name = p.foreign_policy_name or spec.policy_name
            self.__policy_identifier = p.foreign_policy_identifier or spec.policy_identifier
        else:
            self.__param_db = db.make_db(spec.param_db, worker_info=spec.worker_info)
            self.__load_policy_name = spec.policy_name
            self.__policy_identifier = spec.policy_identifier

        if spec.parameter_service_client is not None and self.__load_absolute_path is None:
            self.__parameter_service_client = db.make_client(spec.parameter_service_client, spec.worker_info)
            self.__parameter_service_client.subscribe(experiment_name=self.__param_db.experiment_name,
                                                      trial_name=self.__param_db.trial_name,
                                                      policy_name=self.__load_policy_name,
                                                      tag=self.__policy_identifier,
                                                      callback_fn=self.policy.load_checkpoint,
                                                      use_current_thread=True)

        self.configure_population(spec.population, spec.policy_sample_probs)

        self.__log_frequency_control = base.timeutil.FrequencyControl(frequency_seconds=10)
        self._metric_inference_latency_seconds = METRIC_INFERENCE_LATENCY_SECONDS_LOCAL.labels(
            host=base.network.gethostname(),
            experiment=spec.worker_info.experiment_name,
            trial=spec.worker_info.trial_name,
            stream=spec.policy_name)

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

    def post_request(self, request: api.policy.RolloutRequest, flush=True) -> int:
        request.request_id = np.array([self._request_count], dtype=np.int64)
        req_id = self._request_count
        self.__request_buffer.append(request)
        self._request_count += 1
        if flush:
            self.flush()
        return req_id

    def is_ready(self, inference_ids: List[int]) -> bool:
        for req_id in inference_ids:
            if req_id not in self._response_cache:
                return False
        return True

    def consume_result(self, inference_ids: List[int]):
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
            self.__logger.info(f"Policy Version: {self.policy.version}")

        if len(self.__request_buffer) > 0:
            self.__logger.debug("Inferencing")
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


class LocalInferenceClient(InferenceClient):

    @property
    def type(self):
        return None

    def __init__(self, req_q, resp_q):
        self.req_q = req_q
        self.resp_q = resp_q
        self.client_id = np.random.randint(0, 2147483647)
        self._request_count = 0
        self.__request_buffer = []

        self._response_cache = {}
        self._request_send_time = {}
        self._pending_requests = {}
        self._retried_requests = set()
        self._metric_inference_latency_seconds = None

    def post_request(self, request: api.policy.RolloutRequest, flush=True) -> int:
        request.client_id = np.array([self.client_id], dtype=np.int32)
        request.request_id = np.array([self._request_count], dtype=np.int64)
        req_id = self._request_count
        self.__request_buffer.append(request)
        self._request_send_time[req_id] = time.monotonic_ns()
        self._pending_requests[req_id] = request
        self._request_count += 1
        if flush:
            self.flush()
        return req_id

    def __poll_responses(self):
        """Get all action messages from inference servers."""
        try:
            responses = base.namedarray.loads(self.resp_q.get_nowait())
            for i in range(responses.length(dim=0)):
                req_id = responses.request_id[i, 0]
                if req_id in self._response_cache:
                    raise ValueError(
                        "receiving multiple result with request id {}."
                        "Have you specified different inferencer client_ids for actors?".format(req_id))
                else:
                    if req_id not in self._request_send_time:
                        if req_id in self._retried_requests:
                            logger.warning(f"Received multiple responses for request {req_id}. "
                                           f"Request {req_id} has been retried, ignoring this case.")
                            continue
                        # This is impossible.
                        raise RuntimeError(f"Impossible case: got response but I didn't send it? {req_id}")
                    latency = (time.monotonic_ns() - self._request_send_time.pop(req_id)) / 1e9
                    self._pending_requests.pop(req_id)
                    if self._metric_inference_latency_seconds is not None:
                        self._metric_inference_latency_seconds.observe(latency)
                    self._response_cache[req_id] = responses[i]
        except queue.Empty:
            pass
        except Exception as e:
            raise e

    def is_ready(self, inference_ids) -> bool:
        self.__poll_responses()
        for req_id in inference_ids:
            if req_id not in self._response_cache:
                return False
        return True

    def consume_result(self, inference_ids):
        return [self._response_cache.pop(req_id) for req_id in inference_ids]

    def flush(self):
        if len(self.__request_buffer) > 0:
            agg_request = base.namedarray.dumps(recursive_aggregate(self.__request_buffer, np.stack))
            self.req_q.put(agg_request)
            self.__request_buffer = []

    def get_constant(self, name: str) -> Any:
        name_, value = self.resp_q.get(timeout=30)
        if name_ != name:
            raise ValueError(f"Unexpected constance name: {name_} != {name}")
        return value


class LocalInferenceServer(InferenceServer):

    def __init__(self, req_qs, resp_qs):
        self.req_qs = req_qs
        self.resp_qs = resp_qs
        self.client_id_queue_map = {}

    def poll_requests(self):
        request_batches = []
        for q1, q2 in zip(self.req_qs, self.resp_qs):
            try:
                requests: api.policy.RolloutRequest = base.namedarray.loads(q1.get_nowait())
                requests.received_time[:] = time.monotonic_ns()
                client_id = requests.client_id[0, 0].item()
                self.client_id_queue_map[client_id] = q2
                request_batches.append(requests)
            except queue.Empty:
                break
        return request_batches

    def respond(self, responses: api.policy.RolloutResult):
        idx = np.concatenate([[0],
                              np.where(np.diff(responses.client_id[:, 0]))[0] + 1, [responses.length(dim=0)]])
        for i in range(len(idx) - 1):
            client_id = responses.client_id[idx[i], 0].item()
            self.client_id_queue_map[client_id].put(base.namedarray.dumps(responses[idx[i]:idx[i + 1]]))

    def set_constant(self, name, value):
        # TODO: This is a unsafe implementation
        for q in self.resp_qs:
            q.put((name, value))


register_client(api.config.InferenceStream.Type.INLINE, InlineInferenceClient)


def make_local_server(req_qs, resp_qs) -> InferenceServer:
    return LocalInferenceServer(req_qs, resp_qs)


def make_local_client(req_q, resp_q) -> InferenceClient:
    return LocalInferenceClient(req_q, resp_q)
