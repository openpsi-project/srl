"""This module defines the data flow between policy workers and actor workers.

In our design, actor workers are in charge of executing env.step() (typically simulation), while
policy workers running policy.rollout_step() (typically neural network inference). The inference
stream is the abstraction of the data flow between them: the actor workers send environment
observations as requests, and the policy workers return actions as responses, both plus other
additional information.
"""
from typing import List, Optional, Any, Union

import api.config
import api.policy


class InferenceClient:
    """Interface used by the actor workers to obtain actions given current observation."""

    @property
    def type(self):
        raise NotImplementedError()

    def post_request(self, request: api.policy.RolloutRequest, flush=True) -> int:
        """Set the client_id and request_id of the request and cache the request.

        Args:
            request: RolloutRequest of length 1.
            flush: whether to send the request immediately
        """
        raise NotImplementedError()

    def is_ready(self, inference_ids: List[int]) -> bool:
        """Check whether a specific request is ready to be consumed.

        Args:
            inference_ids: a list of requests to check

        Outputs:
            is_ready: whether the inference_ids are all ready.
        """
        raise NotImplementedError()

    def consume_result(self, inference_ids: List[int]):
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


ALL_INFERENCE_CLIENT_CLS = {}
ALL_INFERENCE_SERVER_CLS = {}


def register_server(type_: api.config.InferenceStream.Type, cls):
    ALL_INFERENCE_SERVER_CLS[type_] = cls


def register_client(type_: api.config.InferenceStream.Type, cls):
    ALL_INFERENCE_CLIENT_CLS[type_] = cls


def make_server(spec: Union[str, api.config.InferenceStream, InferenceServer],
                worker_info: Optional[api.config.WorkerInformation] = None):
    """Initializes an inference stream server.

    Args:
        spec: Inference stream specification.
        worker_info: The server worker information.
    """
    if isinstance(spec, InferenceServer):
        return spec
    if spec.worker_info is None:
        spec.worker_info = worker_info
    if isinstance(spec, str):
        spec = api.config.InferenceStream(type_=api.config.InferenceStream.Type.NAME, stream_name=spec)
    return ALL_INFERENCE_SERVER_CLS[spec.type_](spec)


def make_client(spec: Union[str, api.config.InferenceStream],
                worker_info: Optional[api.config.WorkerInformation] = None):
    """Initializes an inference stream client.

    Args:
        spec: Inference stream specification.
        worker_info: The client worker information.
    """
    if isinstance(spec, InferenceClient):
        return spec
    if isinstance(spec, str):
        spec = api.config.InferenceStream(type_=api.config.InferenceStream.Type.NAME, stream_name=spec)
    if spec.worker_info is None:
        spec.worker_info = worker_info
    return ALL_INFERENCE_CLIENT_CLS[spec.type_](spec)
