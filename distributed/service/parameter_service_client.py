"""
client which could connects to parameter service and subscribes parameters.
"""
from typing import Dict
import getpass
import hashlib
import io
import json
import logging
import queue
import socket
import threading
import time
import torch
import zmq

MULTICAST_METADATA_HEXLENGTH = 16
PARAMETER_SUBSCRIPTION_ID_HEXLENGTH = 16
CLIENT_TOUCH_INTERVAL_SECONDS = 30

logger = logging.getLogger("param-service-client")


def get_host_name():
    return socket.gethostname()


def get_host_ip():
    hostname = get_host_name()
    return socket.gethostbyname(hostname)


class ParameterServiceClientBase:

    def connect(self, address):
        """Connect to specified parameter service.
        """
        raise NotImplementedError()

    def subscribe(self, experiment_name, trial_name, policy_name, tag, callback_fn, user_name):
        """Subscribe specified parameter.
        callback_fn is called when there is new parameter.
        """
        raise NotImplementedError()

    def unsubscribe(self, sub_key):
        """Unsubscribe specified parameter.
        """
        raise NotImplementedError()


class ParameterServiceClient(ParameterServiceClientBase):

    class SubInstance:
        """Listen on specified parameter.
        Callback function is called when there is new parameter.
        """

        def __init__(self, sub_address, sub_key, call_back):
            self.__interrupt = True
            self.__thread = None
            self.__sub_address = sub_address
            self.__topic = sub_key.encode("ascii")
            self.__sub_socket = None
            self.__call_back = call_back
            self.__buffer = None
            self.__msg_sum = None
            self.__serving_idx = -1
            self.__pending_chunks = set()
            logger.debug(f"a new sub instance is created")

        def __del__(self):
            logger.debug(f"a sub instance is destroyed")

        def _set_sub_socket(self):
            ctx = zmq.Context()
            self.__sub_socket = ctx.socket(zmq.SUB)
            self.__sub_socket.setsockopt(zmq.RATE, 80000)
            self.__sub_socket.setsockopt(zmq.SUBSCRIBE, self.__topic)
            self.__sub_socket.setsockopt(zmq.RCVTIMEO, 5000)
            self.__sub_socket.connect(self.__sub_address)
            logger.info(f"connected to {self.__sub_address}")

        def _listen_on(self):
            self._set_sub_socket()

            while not self.__interrupt:
                try:
                    data = self.__sub_socket.recv()[PARAMETER_SUBSCRIPTION_ID_HEXLENGTH:]
                except zmq.ZMQError:
                    continue
                if self.__sub_address.startswith("tcp"):
                    buffer = io.BytesIO(data)
                    logger.debug("received a new parameter")
                    self.__call_back(torch.load(buffer, map_location="cpu"))
                else:
                    l = MULTICAST_METADATA_HEXLENGTH
                    if len(data) < 64 + l * 5:
                        raise RuntimeError(f"Invalid message length {len(data)}")
                    msg_sum = data[:64]
                    chunk_idx = int(data[64:64 + l], base=16)
                    chunks = int(data[64 + l:64 + l * 2], base=16)
                    msg_length = int(data[64 + l * 2:64 + l * 3], base=16)
                    start = int(data[64 + l * 3:64 + l * 4], base=16)
                    end = int(data[64 + l * 4:64 + l * 5], base=16)
                    serving_idx = int(data[64 + l * 5:64 + l * 6], base=16)
                    msg = data[64 + l * 6:]

                    if serving_idx < self.__serving_idx:
                        continue

                    if self.__buffer is None or msg_sum != self.__msg_sum:
                        logger.debug(f"Initiating new buffer, length {msg_length}, sum {msg_sum}, "
                                     f"serving idx {serving_idx} chunks {chunks}")
                        # New message
                        self.__buffer = io.BytesIO(b"0" * msg_length)
                        self.__msg_sum = msg_sum
                        self.__pending_chunks = set(range(chunks))
                        self.__serving_idx = serving_idx

                    if chunk_idx not in self.__pending_chunks or len(msg) != end - start:
                        logger.debug(f"Unexpected chunk index {chunk_idx} out of {self.__pending_chunks}")
                    else:
                        self.__buffer.getbuffer()[start:end] = msg
                        self.__pending_chunks.remove(chunk_idx)

                    if len(self.__pending_chunks) == 0:
                        try:
                            total_sum = hashlib.sha256(self.__buffer.getvalue()).hexdigest().encode("ascii")
                            logger.info(f"Received data sum: {total_sum}")
                            if total_sum != self.__msg_sum:
                                raise RuntimeError(
                                    f"Multicast checksum failed {self.__msg_sum} != {total_sum}")
                            self.__call_back(torch.load(self.__buffer, map_location="cpu"))
                        except queue.Full:
                            logger.debug(f"Callback failed: parameter queue is full.")
                        except Exception as e:
                            logger.error(f"Callback {self.__call_back} failed: {e}")
                        finally:
                            self.__buffer = None

            self.__sub_socket.close()

        def start(self):
            self.__interrupt = False
            self.__thread = threading.Thread(target=self._listen_on)
            self.__thread.start()

        def stop(self):
            self.__interrupt = True
            self.__thread.join()

    def __init__(self):
        self.__request_socket = None
        self.__client_id = ""
        self.__subscriptions: Dict[str, ParameterServiceClient.SubInstance] = {}
        self.__request_socket_lock = threading.Lock()
        self.__touch_thread = None
        self.__touching = False

        hostname = get_host_name()
        if hostname.startswith("frl") or hostname.startswith("dev"):
            self.__client_type = "cluster"
        else:
            self.__client_type = "local"

        logger.info(f"a new client is created")

    def __del__(self):
        logger.debug("a ParameterServiceClient is destroyed")
        subs = list(self.__subscriptions.keys())
        for sub in subs:
            self.unsubscribe(sub)

        self.__touching = False

        if self.__request_socket is not None:
            self.__request_socket.close()

    def _send_request_to_server(self, req):
        if self.__request_socket is not None:
            self.__request_socket.send(json.dumps(req).encode("ascii"))
            logger.debug(f"send {req['type']} request")

    def _receive_response_from_sever(self):
        if self.__request_socket is not None:
            msg = self.__request_socket.recv()
            rep = json.loads(msg)
            logger.debug(f"receive response {rep} from server")
            return rep

    def _new_sub_instance(self, sub_address, sub_key, call_back):
        sub_instance = self.SubInstance(sub_address, sub_key, call_back)
        sub_instance.start()
        return sub_instance

    def _set_request_socket(self, address):
        ctx = zmq.Context()
        self.__request_socket = ctx.socket(zmq.REQ)
        self.__request_socket.setsockopt(zmq.SNDTIMEO, 5000)
        self.__request_socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.__request_socket.connect(f"tcp://{address}")

    def _keep_client_alive(self):
        self.__touch_thread = threading.Thread(target=self._touch_server, daemon=True)
        self.__touching = True
        self.__touch_thread.start()

    def _touch_server(self):
        while self.__touching:
            time.sleep(CLIENT_TOUCH_INTERVAL_SECONDS)
            if len(self.__subscriptions) > 0:
                req = {
                    'type': 'touch',
                    'client_id': self.__client_id,
                }
                try:
                    rep = self._send_req_and_recv_rep(req)
                    if rep['status'] == 'ok':
                        continue
                    elif rep['status'] == 'error':
                        return
                except zmq.ZMQError as e:
                    logger.error(f"zmq error: {e}")
                    continue

    def _send_req_and_recv_rep(self, req):
        with self.__request_socket_lock:
            self._send_request_to_server(req)
            return self._receive_response_from_sever()

    def connect(self, address):
        logger.info(f"connecting to server on {address}")
        self._set_request_socket(address)
        request = {
            "type": "connect",
        }
        rep = self._send_req_and_recv_rep(request)
        if rep['status'] == 'ok':
            self.__client_id = rep["client_id"]
            self._keep_client_alive()
            logger.info(f"connected to server on {address}")
        else:
            logger.error(f"failed connect to {address}")

        return self.__client_id

    def subscribe(self,
                  experiment_name,
                  trial_name,
                  policy_name,
                  tag,
                  callback_fn,
                  user_name=getpass.getuser()):
        sub = {
            "user_name": user_name,
            "experiment_name": experiment_name,
            "trial_name": trial_name,
            "policy_name": policy_name,
            "tag_name": tag,
        }
        logger.debug(f"new sub:{sub}")
        request = {
            "type": "subscribe",
            "client_id": self.__client_id,
            "client_type": self.__client_type,
            **sub
        }
        rep = self._send_req_and_recv_rep(request)
        if rep['status'] == 'ok':
            sub_key = rep["sub_key"]
            if sub_key in self.__subscriptions.keys():
                logger.info(f"repeated subscription")
            else:
                pub_address = rep['pub_address']
                if pub_address.startswith('epgm'):
                    hostip = get_host_ip()
                    pub_address = pub_address.format(hostip)
                sub_instance = self._new_sub_instance(pub_address, sub_key, callback_fn)
                self.__subscriptions[sub_key] = sub_instance
            return sub_key
        else:
            logger.error(rep['comments'])
            return ""

    def unsubscribe(self, sub_key):
        if sub_key not in self.__subscriptions.keys():
            raise KeyError(f"Cannot unsubscribe {sub_key}: No such subscription.")
        else:
            request = {
                "type": "unsubscribe",
                "client_id": self.__client_id,
                "sub_key": sub_key,
            }
            rep = self._send_req_and_recv_rep(request)
            if rep["status"] == "ok":
                self.__subscriptions[sub_key].stop()
                self.__subscriptions.pop(sub_key)
            else:
                logger.error(rep['comments'])


def make_client():
    return ParameterServiceClient()
