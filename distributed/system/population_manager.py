import copy

from distributed.system.worker_base import PollResult
import api.config
import api.pbt
import distributed.system.sample_stream
import distributed.system.worker_base
import distributed.system.worker_control


class PopulationManager(distributed.system.worker_base.Worker):

    def __init__(self, server=None):
        super(PopulationManager, self).__init__(server)
        self.config = None
        self.population = None
        self.__population_algorithm = None
        self.__population_sample_stream = None
        self.__control = None
        self.__handlers = {
            "self_exit": self.exit,
        }

    def _configure(self, cfg: api.config.PopulationManager) -> api.config.WorkerInformation:
        self.config = cfg
        self.population = copy.deepcopy(cfg.population)
        self.__population_algorithm = api.pbt.make(cfg.population_algorithm)
        self.__population_algorithm.configure(cfg.actors, cfg.policies, cfg.trainers, cfg.eval_managers)
        self.__population_sample_stream = distributed.system.sample_stream.make_consumer(
            cfg.population_sample_stream, worker_info=cfg.worker_info)
        self.__control = control = distributed.system.worker_control.make_control(
            experiment_name=cfg.worker_info.experiment_name,
            trial_name=cfg.worker_info.trial_name,
        )
        self.logger.info("Connecting to workers...")
        try:
            control.connect([control.name("actor", i) for i in range(len(cfg.actors))])
            control.connect([control.name("policy", i) for i in range(len(cfg.policies))])
            control.connect([control.name("trainer", i) for i in range(len(cfg.trainers))])
            control.connect([control.name("eval_manager", i) for i in range(len(cfg.eval_managers))])
        except (TimeoutError, KeyboardInterrupt) as e:
            self.logger.error("Failed connecting to worker: %s", e)
            raise e

        r = self.config.worker_info
        return r

    def _poll(self) -> PollResult:
        sample_count = 0
        batch_count = 0
        try:
            sample = self.__population_sample_stream.consume()
            sample_count += 1
        except distributed.system.sample_stream.NothingToConsume:
            return PollResult(sample_count=sample_count, batch_count=batch_count)

        requests = self.__population_algorithm.step(sample)
        if requests is not None:
            for command, kwargs in requests.items():
                if command in self.__handlers.keys():
                    self.logger.info(f"{command} population manager.")
                    self.__handlers[command](**kwargs)
                else:
                    self.logger.info(f"{command} workers.")
                    self.__control.group_request(command, **kwargs)
                    self._print_request(command, **kwargs)
            batch_count += 1
        return PollResult(sample_count=sample_count, batch_count=batch_count)

    def _print_request(self, command, worker_names=None, worker_kwargs=None):
        self.logger.debug(f"command: {command}")
        if worker_names is not None:
            self.logger.debug(f"worker_names: {worker_names}")
        if worker_kwargs is not None:
            self.logger.debug(f"worker_kwargs: {worker_kwargs}")
