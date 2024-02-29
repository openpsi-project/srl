import argparse
import json
import logging
import time

import numpy as np
import zmq

import distributed.base.name_resolve
from api.config import ParameterDB, WorkerInformation
from base.timeutil import FrequencyControl
from distributed.system.parameter_db import make_db, MultiCastParameterServiceClient
from .utils import EXPERIMENT_NAME, IDENTIFIER, get_sub_request


class PseudoWorker:

    def __init__(self, pull_frequency, post_frequency, step_frequency, trial_name, pull_type="filesystem"):
        self.test_worker_info = WorkerInformation(experiment_name=EXPERIMENT_NAME, trial_name=trial_name)
        self.policy_name = "test"
        self.db = make_db(ParameterDB(type_=ParameterDB.Type.FILESYSTEM), worker_info=self.test_worker_info)
        self.__version = -1
        self.logger = logging.getLogger("Pseudo Worker")
        self.__pull_frequency_control = FrequencyControl(frequency_seconds=pull_frequency, initial_value=True)
        self.__post_frequency_control = FrequencyControl(frequency_seconds=post_frequency, initial_value=True)
        self.__step_frequency_control = FrequencyControl(frequency_seconds=step_frequency, initial_value=True)
        self.__pull_type = pull_type

    def multicast_callback(self, param):
        self.__version = param["version"]

    def run(self, max_version=100, max_time=200):
        c = zmq.Context()
        s = c.socket(zmq.PUSH)
        addr = distributed.base.name_resolve.wait(
            f"benchmark/{self.test_worker_info.experiment_name}/{self.test_worker_info.trial_name}/sample_stream"
        )
        self.logger.debug(f"connecting to {addr}")
        s.connect(addr)
        s.setsockopt(zmq.SNDTIMEO, 1000)
        samples = []
        if self.__pull_type == "multicast":
            client = MultiCastParameterServiceClient.Subscription(
                sub_request=get_sub_request(self.test_worker_info.trial_name),
                host_experiment_name=self.test_worker_info.experiment_name,
                host_trial_name=self.test_worker_info.trial_name,
                callback=self.multicast_callback,
                recv_timeo=5000,
                rank=0)
            client.start()
            self.logger.info("waiting for 30 seconds for clients to connect.")
            time.sleep(30)

        ddl = time.time() + max_time
        while self.__version <= max_version and time.time() < ddl:
            if self.__pull_type == "filesystem" and self.__pull_frequency_control.check():
                self.__version = self.db.get(self.policy_name, block=True, identifier=IDENTIFIER)["version"]
                self.logger.debug(f"Pulled parameter version {self.__version}")
            if self.__step_frequency_control.check():
                samples.append(self.__version)
            if self.__post_frequency_control.check():
                v = np.mean(samples)
                self.logger.debug(f"Send samples version {v} of length {len(samples)}")
                samples = []
                msg = json.dumps({"version": v})
                s.send(msg.encode("ascii"))
        if self.__pull_type == "multicast":
            client.stop()
        s.close(linger=False)
        c.term()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="pseudo_worker")
    parser.add_argument("--mode", type=str, default="filesystem", required=False)
    parser.add_argument("--pull_frequency", type=float, required=True, help="time interval per pull seconds.")
    parser.add_argument("--post_frequency", type=int, required=True, help="time interval per post seconds")
    parser.add_argument("--step_frequency", type=float, required=True, help="time interval per step")
    parser.add_argument("--max_version", type=int, required=True, help="maximum version for this run")
    parser.add_argument("--max_time", type=int, required=True, help="maximum time this run.")
    parser.add_argument("--trial_name", type=str, required=True, help="trial name.")
    parser.add_argument("--LOGLEVEL", type=str, required=False, default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.LOGLEVEL)
    w = PseudoWorker(pull_frequency=args.pull_frequency,
                     post_frequency=args.post_frequency,
                     step_frequency=args.step_frequency,
                     pull_type=args.mode,
                     trial_name=args.trial_name)
    w.run(max_version=args.max_version, max_time=args.max_time)
