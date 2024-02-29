import base.shared_memory as shared_memory
import distributed.system.worker_base as worker_base
import api.config as config_package

import logging
import time

logger = logging.getLogger("SharedMemoryWorker")


# To write a worker you need to modify:
# 0. write worker class from worker_base.WorkerBase
# 1. api.config: add Scheduling method, add ExperimentConfig, Experiment scheduling add config class for this worker
# 2. distributed.system.__init__: add worker specs
# 3. apps.main: add main_start worker type
# 4. write corresponding experiment config
# TODO: In the long run, modify these into one method like `register_worker`
class SharedMemoryWorker(worker_base.Worker):
    """ An intermediate server for shared memory communication between 
    workers launched by multiprocessing.
    """

    def __init__(self, server=None, lock=None):
        super().__init__(server, lock)

        self.__experiment_name = None
        self.__trial_name = None
        self.__stream_name = None
        self.__qsize = None
        self.__reuses = None
        self.__type = None

        self.__dock_server = None
        self.__start_time = None

    def _configure(self, config: config_package.SharedMemoryWorker):
        self.__experiment_name = config.worker_info.experiment_name
        self.__trial_name = config.worker_info.trial_name
        self.__stream_name = config.stream_name
        self.__qsize = config.qsize
        self.__reuses = config.reuses

        if config.type_ == config_package.SharedMemoryWorker.Type.SAMPLE:
            self.__dock_server = shared_memory.SharedMemoryDockServer(self.__experiment_name,
                                                                      self.__trial_name, self.__stream_name,
                                                                      self.__qsize, self.__reuses)
            self.__type = "SAMPLE"
        elif config.type_ == config_package.SharedMemoryWorker.Type.INFERENCE:
            self.__dock_server = shared_memory.PinnedSharedMemoryServer(self.__experiment_name,
                                                                        self.__trial_name, self.__stream_name)
            self.__type = "INFERENCE"
        else:
            raise NotImplementedError()

        logger.info("SharedMemoryWorker dock server configured")
        self.__dock_server.start()
        logger.info("Dock server started")

        r = config.worker_info
        self.__start_time = time.monotonic()
        return r

    def _poll(self):
        # Do nothing in main thread, and run server in threads
        time.sleep(10)
        # logger.info(f"{self.__type} Shared memory server handled {self.__dock_server.messages_handled} messages."
        #             f" Throughput: {self.__dock_server.messages_handled/(time.monotonic() - self.__start_time)}")
        # logger.info(f"Timing: {self.__dock_server.timing}")

        return worker_base.PollResult(sample_count=0, batch_count=0)
