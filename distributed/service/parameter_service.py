"""
microservice which publishes subscribed parameters.
"""
from typing import Dict, List
import argparse
import hashlib
import json
import logging
import math
import random
import socket
import threading
import time
import uuid
import zmq

from distributed.system.parameter_db import make_db
import api.config

MULTICAST_PACKAGE_SIZE = 5 * 1024 * 1024  # Bytes
MULTICAST_METADATA_HEXLENGTH = 16
MULTICAST_SEND_INTERVAL = 0.5  # seconds
PARAMETER_SUBSCRIPTION_ID_HEXLENGTH = 16
PUBLISHER_INSPECTION_INTERVAL_SECONDS = 120
SUBSCRIPTION_MAX_TTL_SECONDS = 300
SUBSCRIPTION_EXTEND_TTL_PER_TOUCH = 90
DEFAULT_PARAMETER_DB_TYPE = api.config.ParameterDB.Type.FILESYSTEM

logger = logging.getLogger("param-service")


def get_host_name():
    return socket.gethostname()


def get_host_ip():
    hostname = get_host_name()
    return socket.gethostbyname(hostname)


class ParameterServiceBase:

    def run(self):
        """run the service
        """
        raise NotImplementedError()

    def stop_all(self):
        """stop the service
        """
        raise NotImplementedError()


class ParameterService(ParameterServiceBase):

    class PubInstance:
        """publish specified parameter through tcp or epgm
        """

        def __init__(self, sub_req, host_ip=get_host_ip()):
            self.__interrupt = True
            self.__thread = None
            self.__last_sum = ""
            self.__serving_count = 0
            self.__sub_req = sub_req
            self.__topic = ParameterService.get_sub_key(sub_req).encode("ascii")
            self.__host_ip = host_ip
            self.__pub_address = None
            self.__pub_socket = None
            self.__context = None
            self._set_pub_socket()
            self.__parameter_db = make_db(spec=api.config.ParameterDB(type_=DEFAULT_PARAMETER_DB_TYPE),
                                          worker_info=api.config.WorkerInformation(
                                              experiment_name=sub_req['experiment_name'],
                                              trial_name=sub_req['trial_name']),
                                          user_namespace=sub_req['user_name'])
            logger.debug(f"a new pub instance is created")

        def __del__(self):
            if self.__pub_socket is not None:
                self.__pub_socket.close()
                self.__context.destroy()
            logger.debug(f"a pub instance is destroyed")

        def __get_parameter(self):
            try:
                return self.__parameter_db.get(name=self.__sub_req['policy_name'],
                                               identifier=self.__sub_req['tag_name'],
                                               mode='bytes')
            except FileNotFoundError:
                return b""

        def _set_pub_socket(self):
            # ctx = zmq.Context()
            self.__context = zmq.Context()
            self.__pub_socket = self.__context.socket(zmq.PUB)
            self.__pub_socket.setsockopt(zmq.RATE, 80000)

            if self.__sub_req['client_type'] == 'cluster':
                self.__pub_address = f"epgm://{{}};" \
                              f"239.192.{random.randint(1, 11)}.{random.randint(1, 255)}:{random.randint(3000, 6000)}"
                self.__pub_socket.bind(self.__pub_address.format(self.__host_ip))
            elif self.__sub_req['client_type'] == 'local':
                port = self.__pub_socket.bind_to_random_port(f"tcp://{self.__host_ip}")
                self.__pub_address = f"tcp://{self.__host_ip}:{port}"
            logger.info(f"publishing on {self.__pub_address}")

        def __publish(self):
            while not self.__interrupt:
                msg = self.__get_parameter()
                check_sum = hashlib.sha256(msg).hexdigest()
                if msg == b"" or check_sum == self.__last_sum:
                    time.sleep(1)
                else:
                    if self.__pub_address.startswith("tcp"):
                        self.__pub_socket.send(self.__topic + msg)
                    elif self.__pub_address.startswith("epgm"):
                        msg_length = len(msg)
                        chunks = math.ceil(len(msg) / MULTICAST_PACKAGE_SIZE)
                        for c in range(chunks):
                            start = c * MULTICAST_PACKAGE_SIZE  # where the slice message start
                            end = min((c + 1) * MULTICAST_PACKAGE_SIZE,
                                      msg_length)  # where the slice message end
                            # multicast_pkg includes the following in turn:
                            # the order of the current slice message in the complete message
                            # total number of slice messages of the complete message
                            # length of complete message
                            # start position(byte) of the current slice message in the complete message
                            # start position(byte) of the current slice message in the complete message
                            # times of sending parameter
                            # content of current slice message
                            multicast_pkg = self.__topic + \
                                            check_sum.encode("ascii") + \
                                            "0x{0:0{1}X}".format(c, MULTICAST_METADATA_HEXLENGTH - 2).\
                                                encode("ascii") + \
                                            "0x{0:0{1}X}".format(chunks, MULTICAST_METADATA_HEXLENGTH - 2).\
                                                encode("ascii") + \
                                            "0x{0:0{1}X}".format(msg_length, MULTICAST_METADATA_HEXLENGTH - 2).\
                                                encode("ascii") + \
                                            "0x{0:0{1}X}".format(start, MULTICAST_METADATA_HEXLENGTH - 2).\
                                                encode("ascii") + \
                                            "0x{0:0{1}X}".format(end, MULTICAST_METADATA_HEXLENGTH - 2).\
                                                encode("ascii") + \
                                            "0x{0:0{1}X}".format(self.__serving_count,
                                                                 MULTICAST_METADATA_HEXLENGTH - 2).encode("ascii") \
                                            + msg[start: end]
                            self.__pub_socket.send(multicast_pkg)
                            time.sleep(MULTICAST_SEND_INTERVAL)
                        self.__serving_count += 1
                    self.__last_sum = check_sum
                    logger.debug("send a new parameter")

        def start(self):
            self.__interrupt = False
            self.__thread = threading.Thread(target=self.__publish)
            self.__thread.start()

        def stop(self):
            self.__interrupt = True
            self.__thread.join()

        def get_pub_address(self):
            return self.__pub_address

        def get_topic(self):
            return self.__topic

    def __init__(self, address=get_host_ip(), port="3000"):
        self.__listen_address = address  # one ip address of host which may have multiple ip address
        self.__listen_port = port
        self.__listen_socket = None
        self.__context = None
        self.__active_clients_ttl = {}
        self.__next_inspection_time = -1
        self.__active_clients_sub: Dict[str, List[str]] = {}
        self.__subscriptions: Dict[str, ParameterService.PubInstance] = {}
        self.__sleeping_clients = []  # clients which don't have any subscription
        self.__running = True

    def __del__(self):
        if self.__listen_socket is not None:
            self.__listen_socket.close()
            self.__context.destroy()

    def _set_listen_socket(self):
        try:
            # ctx = zmq.Context()
            self.__context = zmq.Context()
            self.__listen_socket = self.__context.socket(zmq.REP)
            self.__listen_socket.bind(f"tcp://{self.__listen_address}:{self.__listen_port}")
            self.__listen_socket.setsockopt(zmq.RCVTIMEO, 3000)
            logger.info(f"listening on {self.__listen_address}:{self.__listen_port}")
        except zmq.ZMQError as e:
            logger.error(f"zmq error: {e}")
            raise e

    def _receive_request_from_client(self):
        """
        every request must have a type
        """
        try:
            msg = self.__listen_socket.recv().decode("ascii")
            req = json.loads(msg)
            if req['type'] == "connect":
                logger.debug(f"received {req['type']} request")
            else:
                logger.debug(f"received {req['type']} request from {req['client_id']}")
            return req
        except zmq.ZMQError as e:
            raise e
        except (UnicodeError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"invalid request")
            raise e

    def _send_response_to_client(self, rep):
        self.__listen_socket.send(json.dumps(rep).encode("ascii"))
        logger.debug(f"send response {rep} to client")

    def _new_pub_instance(self, sub_req, host_ip=get_host_ip()):
        pub_instance = self.PubInstance(sub_req, host_ip)
        pub_instance.start()
        return pub_instance

    @staticmethod
    def get_sub_key(sub_req):
        sub = {
            "client_type": sub_req["client_type"],
            "user_name": sub_req['user_name'],
            "experiment_name": sub_req['experiment_name'],
            "trial_name": sub_req['trial_name'],
            "policy_name": sub_req['policy_name'],
            "tag_name": sub_req['tag_name'],
        }
        sub_str = json.dumps(sub, separators=(',', ':'), sort_keys=True)
        return hashlib.sha256(sub_str.encode("ascii")).hexdigest()[:PARAMETER_SUBSCRIPTION_ID_HEXLENGTH]

    def _response_subscribe_request(self, sub_req):
        if not {'client_type', 'client_id', 'user_name', 'experiment_name', 'trial_name', 'policy_name', 'tag_name'} \
                <= set(sub_req.keys()):
            logger.error("invalid subscribe request")
            rep = {
                'status': 'error',
                'comment': 'invalid request',
            }
            return rep

        client_id = sub_req['client_id']
        if client_id not in self.__active_clients_sub.keys() and client_id not in self.__sleeping_clients:
            logger.error('nonexistent client')
            rep = {
                'status': 'error',
                'comment': 'nonexistent client',
            }
            return rep

        sub_key = self.get_sub_key(sub_req)
        if sub_key not in self.__subscriptions.keys():
            pub_instance = self._new_pub_instance(sub_req, self.__listen_address)
            self.__subscriptions[sub_key] = pub_instance
        pub_address = self.__subscriptions[sub_key].get_pub_address()

        if client_id not in self.__active_clients_sub.keys():
            self.__active_clients_sub[client_id] = [sub_key]
            self.__active_clients_ttl[client_id] = time.time() + SUBSCRIPTION_MAX_TTL_SECONDS
            self.__sleeping_clients.remove(client_id)
        if sub_key not in self.__active_clients_sub[client_id]:
            self.__active_clients_sub[client_id].append(sub_key)
        rep = {
            "status": "ok",
            "pub_address": pub_address,
            "sub_key": sub_key,
        }
        return rep

    def _response_connect_request(self):
        client_id = str(uuid.uuid4())
        rep = {
            "status": "ok",
            "client_id": client_id,
        }
        # self.__clients_sub[client_id] = []
        # self.__clients_ttl[client_id] = time.time() + SUBSCRIPTION_MAX_TTL_SECONDS
        self.__sleeping_clients.append(client_id)
        logger.info(f"a new client {client_id}")
        return rep

    def _has_client_for_sub(self, sub_key):
        for _, subs in self.__active_clients_sub.items():
            if sub_key in subs:
                return True
        return False

    def _response_unsubscribe_request(self, req):
        if not {'sub_key', 'client_id'} <= set(req.keys()):
            logger.error("invalid unsubscribe request")
            rep = {
                'status': 'error',
                'comment': 'invalid request',
            }
            return rep

        client_id = req["client_id"]
        if client_id not in self.__active_clients_sub.keys():
            logger.error('inactive client')
            rep = {
                'status': 'error',
                'comment': 'inactive client',
            }
            return rep

        sub_key = req['sub_key']
        if sub_key not in self.__active_clients_sub[client_id]:
            logger.error('invalid subscription')
            rep = {
                'status': 'error',
                'comment': 'invalid subscription',
            }
            return rep

        self.__active_clients_sub[client_id].remove(sub_key)
        if not self._has_client_for_sub(sub_key):
            self.__subscriptions[sub_key].stop()
            self.__subscriptions.pop(sub_key)
        rep = {"status": "ok"}
        return rep

    def _response_touch_request(self, req):
        if 'client_id' not in req.keys():
            logger.error("invalid touch request")
            rep = {
                'status': 'error',
                'comment': 'invalid request',
            }
            return rep

        client_id = req['client_id']
        if client_id not in self.__active_clients_ttl.keys() and client_id not in self.__sleeping_clients:
            logger.error('nonexistent client')
            rep = {
                'status': 'error',
                'comment': 'nonexistent client',
            }
            return rep

        if client_id not in self.__active_clients_ttl.keys():
            self.__sleeping_clients.remove(client_id)
            self.__active_clients_ttl[client_id] = time.time()
            self.__active_clients_sub[client_id] = []

        self.__active_clients_ttl[client_id] = min(
            time.time() + SUBSCRIPTION_MAX_TTL_SECONDS,
            self.__active_clients_ttl[client_id] + SUBSCRIPTION_EXTEND_TTL_PER_TOUCH)

        rep = {
            'status': 'ok',
        }
        return rep

    def _get_response(self, req):
        if req["type"] == "subscribe":
            rep = self._response_subscribe_request(req)
        elif req["type"] == "connect":
            rep = self._response_connect_request()
        elif req["type"] == "unsubscribe":
            rep = self._response_unsubscribe_request(req)
        elif req["type"] == "touch":
            rep = self._response_touch_request(req)
        else:
            logger.error(f"unsupported request type: {req['type']}")
            rep = {
                "status": "error",
                "comments": "unsupported request type",
            }
        return rep

    def _serve(self, req):
        rep = self._get_response(req)
        self._send_response_to_client(rep)

    def _manage_publishers(self):
        clients_to_remove = []
        for client, ttl in self.__active_clients_ttl.items():
            if ttl < time.time():
                clients_to_remove.append(client)
        for client in clients_to_remove:
            self.__active_clients_ttl.pop(client)
            self.__active_clients_sub.pop(client)
            self.__sleeping_clients.append(client)

        sub_to_remove = []
        for sub_key in self.__subscriptions.keys():
            if not self._has_client_for_sub(sub_key):
                sub_to_remove.append(sub_key)
        for sub in sub_to_remove:
            self.__subscriptions[sub].stop()
            self.__subscriptions.pop(sub)

    def run(self):
        try:
            if self.__listen_socket is None:
                self._set_listen_socket()
        except zmq.ZMQError:
            return

        for _, sub in self.__subscriptions:
            sub.start()

        while self.__running:
            if time.time() > self.__next_inspection_time:
                self._manage_publishers()
                self.__next_inspection_time = time.time() + PUBLISHER_INSPECTION_INTERVAL_SECONDS

            try:
                req = self._receive_request_from_client()
            except zmq.ZMQError:
                continue
            except (UnicodeError, json.JSONDecodeError, KeyError):
                rep = {
                    "status": "error",
                    "comments": "invalid request",
                }
                self._send_response_to_client(rep)
            else:
                self._serve(req)

    def stop_all(self):
        for _, sub in self.__subscriptions.items():
            sub.stop()

    def interrupt(self):
        self.stop_all()
        self.__running = False

    def active_clients_num(self):
        return len(self.__active_clients_sub)

    def sleeping_clients_num(self):
        return len(self.__sleeping_clients)

    def pub_instance_num(self):
        return len(self.__subscriptions)


def make_service(address=get_host_ip(), port="3000"):
    return ParameterService(address, port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--address', default=get_host_ip(), help='listen ip address')
    parser.add_argument('-p', '--port', default="3000", help='port number')
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
        'debug': logging.DEBUG,
    }
    logging.basicConfig(level=log_levels[args.log_level],
                        format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s",
                        datefmt="%Y%m%d-%H:%M:%S")

    service = make_service(args.address, args.port)
    service.run()
