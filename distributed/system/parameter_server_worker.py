import time

from distributed.system.parameter_db import make_server
from distributed.system.worker_base import Worker, PollResult
import api.config


class ParameterServerWorker(Worker):

    def __init__(self, server=None, lock=None):
        super(ParameterServerWorker, self).__init__(server, lock)
        self.__parameter_server = None
        self.config = None

    def _reconfigure(self, **kwargs) -> api.config.WorkerInformation:
        pass

    def _configure(self, config) -> api.config.WorkerInformation:
        self.config = config
        self.__parameter_server = make_server(spec=config.parameter_server, worker_info=config.worker_info)
        self.__parameter_server.update_subscription()
        return self.config.worker_info

    def _poll(self) -> PollResult:
        self.__parameter_server.update_subscription()
        time.sleep(5)
        return PollResult(sample_count=1, batch_count=1)

    def start(self):
        self.__parameter_server.run()
        super(ParameterServerWorker, self).start()
