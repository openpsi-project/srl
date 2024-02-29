import argparse
import hashlib
import json
import logging
import socket
import threading
import time
import numpy as np
import zmq
import os

import distributed.base.name_resolve
from api.config import ParameterDB, WorkerInformation
from base.timeutil import FrequencyControl
from distributed.system.parameter_db import make_db, MultiCastParameterServer, PARAMETER_SUBSCRIPTION_ID_HEXLENGTH
from .utils import EXPERIMENT_NAME, IDENTIFIER, get_multicast_param_id

TRAINER_LINGER_SECONDS = 10


class PseudoTrainer:

    def __init__(self, data_size, push_frequency, trial_name, push_type="filesystem", push_only=False):
        self.test_worker_info = WorkerInformation(experiment_name=EXPERIMENT_NAME, trial_name=trial_name)
        self.policy_name = "test"
        self.db = make_db(ParameterDB(type_=ParameterDB.Type.FILESYSTEM), worker_info=self.test_worker_info)
        self.__version = -1
        self.__data_size = data_size
        self.__push_frequency_control = FrequencyControl(frequency_seconds=push_frequency)
        self.__push_type = push_type
        self.__log_frequency_control = FrequencyControl(frequency_seconds=5)
        self.__interrupt = False
        self.__push_only = push_only
        self.logger = logging.getLogger("Pseudo Trainer")
        self.__checker_thread = threading.Thread(target=self.run_version_check, daemon=False)
        self.__push_time_records = []
        self.__data = os.urandom(self.__data_size * 1024 * 1024)

    def run_version_check(self):
        version_diffs = []
        c = zmq.Context()
        s = c.socket(zmq.PULL)
        port = s.bind_to_random_port("tcp://*")
        hostname = socket.gethostbyname(socket.gethostname())
        try:
            distributed.base.name_resolve.delete(
                f"benchmark/{self.test_worker_info.experiment_name}/{self.test_worker_info.trial_name}/sample_stream"
            )
        except:
            pass
        distributed.base.name_resolve.add(
            f"benchmark/{self.test_worker_info.experiment_name}/{self.test_worker_info.trial_name}/sample_stream",
            f"tcp://{hostname}:{port}",
            keepalive_ttl=3,
            delete_on_exit=True)

        self.logger.info(f"start version checker on tcp://{hostname}:{port}")

        s.setsockopt(zmq.RCVTIMEO, 1000)
        while not self.__interrupt:
            try:
                data_str = s.recv().decode("ascii")
                version = json.loads(data_str)["version"]
                version_diffs.append(self.__version - version)
                self.logger.debug(f"Version diff: {self.__version - version}")
            except zmq.ZMQError:
                time.sleep(0.01)
            if self.__log_frequency_control.check() and len(version_diffs) > 0:
                self.logger.info(f"Average Version difference {np.mean(version_diffs):.2f}")

        self.logger.info(f"Average Version difference {np.mean(version_diffs):.2f}")
        s.close(linger=False)
        c.term()

    def push_one(self):
        self.__version += 1
        start = time.monotonic_ns()
        self.db.push(
            name=self.policy_name,
            checkpoint={
                "data": self.__data,
                "version": self.__version
            },
            version=str(self.__version),
            tags=str(self.__version) + "tag",
        )
        push_time = (time.monotonic_ns() - start) / 1e9
        self.__push_time_records.append(push_time)
        self.logger.debug(f"pushed parameter version {self.__version}: {push_time:.3f} seconds")

    def run_multicast(self):
        HASH_ID = hashlib.sha256(get_multicast_param_id(self.test_worker_info.trial_name).encode(
            "ascii")).hexdigest()[:PARAMETER_SUBSCRIPTION_ID_HEXLENGTH]
        TOPIC = HASH_ID.encode("ascii")

        server = MultiCastParameterServer.ServingInstance(
            hash_id=HASH_ID,
            host_experiment_name=self.test_worker_info.experiment_name,
            host_trial_name=self.test_worker_info.trial_name,
            topic=TOPIC,
            db=self.db,
            args=dict(name=self.policy_name, identifier=IDENTIFIER),
        )
        self.logger.info("waiting for 30 seconds for clients to connect.")
        time.sleep(30)
        server.start()
        return server

    def run(self, max_version=100, max_time=200):
        assert max_version is not None and max_time is not None
        self.db.purge(self.test_worker_info.experiment_name, self.test_worker_info.trial_name)
        if not self.__push_only:
            self.__checker_thread.start()
        if self.__push_type == "multicast":
            server = self.run_multicast()
        ddl = time.time() + max_time
        while self.__version <= max_version and time.time() < ddl:
            if self.__push_frequency_control.check():
                self.push_one()
            else:
                time.sleep(0.002)
        self.__interrupt = True
        if self.__push_type == "multicast":
            server.stop()
        self.logger.info(f"Final verison: {self.__version}")
        self.logger.info(f"Time per push: {np.mean(self.__push_time_records):.2f} seconds")
        self.logger.info(f"Job done. Lingering for {TRAINER_LINGER_SECONDS} seconds before purging.")
        time.sleep(TRAINER_LINGER_SECONDS)
        self.logger.info(
            f"purging {self.test_worker_info.experiment_name} {self.test_worker_info.trial_name} from db."
            f"Please wait for the program to exit...")
        self.db.purge(self.test_worker_info.experiment_name, self.test_worker_info.trial_name)
        self.logger.info(
            f"purged {self.test_worker_info.experiment_name} {self.test_worker_info.trial_name} from db."
            f"Thank you for waiting.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="pseudo_trainer")
    parser.add_argument("--mode", type=str, default="filesystem", required=False)
    parser.add_argument("--push_frequency", type=float, required=True, help="seconds per push")
    parser.add_argument("--data_size", type=int, required=True, help="size of each ckpt, in MegaBytes")
    parser.add_argument("--max_version", type=int, required=True, help="maximum version for this run")
    parser.add_argument("--max_time", type=int, required=True, help="maximum time this run.")
    parser.add_argument("--trial_name", type=str, required=True, help="trial name.")
    parser.add_argument("--LOGLEVEL", type=str, required=False, default="INFO")
    parser.add_argument("--push_only", action="store_true", default=False, required=False)
    args = parser.parse_args()
    logging.basicConfig(level=args.LOGLEVEL)
    t = PseudoTrainer(data_size=args.data_size,
                      push_frequency=args.push_frequency,
                      push_type=args.mode,
                      trial_name=args.trial_name,
                      push_only=args.push_only)
    t.run(max_version=args.max_version, max_time=args.max_time)
