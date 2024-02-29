from typing import List, Dict, Tuple, Optional
from datetime import datetime
import dataclasses
import enum
import logging
import time

from distributed.system.worker_base import WorkerServerStatus as Wss
from distributed.system import RL_WORKERS
import api.config
import base.names as names
import distributed.base.name_resolve
import distributed.base.monitoring
import distributed.system.worker_base
import distributed.system.worker_control

CONNECTION_RETRY_AFTER_SECONDS = 360

logger = logging.getLogger("controller")


@dataclasses.dataclass
class TrialStatus:
    experiment_name: str
    trial_name: str
    running_workers: Dict[str, List[str]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TrialHistory:
    experiment_name: str
    trial_name: str
    age_days: int


class ControllerExitStatus(enum.Enum):
    SUCCESS = 0
    TIMEOUT = 1
    INTERRUPTED = 9
    FAIL = 101
    LOST = 102
    UNKNOWN = 404


class Controller:
    # TODO: make threaded controller

    def __init__(self, experiment_name, trial_name, mode="local"):
        """Initialization method of controller.
        Args:
            experiment_name: name of the experiment, as registered to experiments.config_.ALL_EXPERIMENT_CLASSES.
            trial_name: the unique name of this trial.
            mode: scheduler mode. Currently supported are "local" and "slurm"
        """
        assert "_" not in experiment_name, f"_ not allowed in experiment_name (args: -e) " \
                                           f"{experiment_name}, use '-' instead."
        assert "_" not in trial_name, f"_ not allowed in trial_name (args: -f) {trial_name}, use '-' instead."
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.__mode = mode

        self.__control = distributed.system.worker_control.make_control(experiment_name=self.experiment_name,
                                                                        trial_name=self.trial_name)
        logger.info("Experiment: %s %s", self.experiment_name, self.trial_name)

    def reconnect(self):
        """Automatically reconnect to workers. And list all jobs to scheduler.
        """
        self.__control.auto_connect()

    def start(self, experiment, ignore_worker_error=False):
        """Start an experiment.
        Args:
            experiment: An experiment class, with `initial_setup` method returning workers configurations.
            ignore_worker_error: If True, do not stop experiment when part of worker(s) fail.
        """
        if ignore_worker_error:
            check_worker_status = ()
            remove_worker_status = (Wss.COMPLETED, Wss.ERROR, Wss.LOST, Wss.UNKNOWN)
        else:
            check_worker_status = (Wss.ERROR, Wss.LOST, Wss.UNKNOWN)
            remove_worker_status = (Wss.COMPLETED,)

        scheduling: api.config.ExperimentScheduling = experiment.scheduling_setup()
        setup = experiment.initial_setup()
        # Scheduling and connecting to workers.
        workers_configs = [(k, getattr(setup, v.config_field_name), getattr(scheduling, v.config_field_name))
                           for k, v in RL_WORKERS.items()]
        for name, config, schedule in workers_configs:
            count = sum([s.count for s in schedule]) if isinstance(schedule, list) else schedule.count
            if len(config) != count:
                logger.error("Scheduling and config mismatch, interrupting all workers.")
                self.interrupt()
                raise IndexError(f"Configuration has {len(config)} {name} workers, {count} scheduled.")

        setup.set_worker_information(experiment_name=self.experiment_name, trial_name=self.trial_name)
        logger.info(
            "Made initial setup: %d actors, %d policies, %d trainers, %d eval_managers, %d population_manager"
            " %d buffer_workers", len(setup.actors), len(setup.policies), len(setup.trainers),
            len(setup.eval_managers), len(setup.population_manager), len(setup.buffers))

        # State clean-up.
        logger.info("Cleaning up previous states")
        distributed.base.name_resolve.clear_subtree(names.trial_root(self.experiment_name, self.trial_name))
        distributed.base.name_resolve.add(names.trial_registry(self.experiment_name, self.trial_name),
                                          value=datetime.now().strftime("%Y%m%d"),
                                          delete_on_exit=False,
                                          replace=True)
        distributed.base.name_resolve.add(names.worker_status(experiment_name=self.experiment_name,
                                                              trial_name=self.trial_name,
                                                              worker_name="ctl"),
                                          value="READY",
                                          delete_on_exit=True)

        while True:
            try:
                logger.info("Connecting to workers...")
                self.__control.connect([
                    self.__control.name(name, i) for name, cfgs, _ in workers_configs
                    for i in range(len(cfgs))
                ],
                                       progress=True,
                                       timeout=CONNECTION_RETRY_AFTER_SECONDS,
                                       raises_timeout_error=True)
                break

            except TimeoutError:
                logger.info("Connecting to workers timeout. Retrying...")
            except KeyboardInterrupt as e:
                logger.info("Interrupted by user. Stopping all and exiting...")
                raise e

        distributed.base.name_resolve.delete(
            names.worker_status(experiment_name=self.experiment_name,
                                trial_name=self.trial_name,
                                worker_name="ctl"))

        def configure_worker_with_config(name, cfgs):
            logger.info(f"Configuring Workers: {name}...")
            self.__control.group_request(
                "configure",
                worker_names=[self.__control.name(name, i) for i in range(len(cfgs))],
                worker_kwargs=[dict(config=cfg) for cfg in cfgs],
                progress=True)

        # make worker_configs to dict
        worker_configs = {k: v for k, v, _ in workers_configs}

        # Configure workers.
        try:
            if "shared_memory_worker" in worker_configs:
                configure_worker_with_config("shared_memory_worker", worker_configs["shared_memory_worker"])
                worker_configs.pop("shared_memory_worker")

            for name, cfgs in worker_configs.items():
                configure_worker_with_config(name, cfgs)

        except Exception as e:
            logger.error("Configuring Failed. Exiting Workers.")
            self.interrupt(wait_timeout=120)
            raise e

        # # Configure monitoring.
        logger.info("Configuring monitoring")
        mon_addresses = []
        mon_repo = distributed.base.monitoring.TargetRepository()
        workers = None
        for _ in range(10):
            rs = self.__control.group_request("start_monitoring", worker_names=workers, timeout=60)
            workers = []
            for r in rs:
                if r.timed_out:
                    workers.append(r.worker_name)
                else:
                    mon_addresses.append(f"{r.result.host}:{r.result.prometheus_port}")
            if len(workers) == 0:
                break
            logger.warning("Failed start monitoring for %d workers, reconnecting and trying again",
                           len(workers))
            self.__control.connect(workers, reconnect=True)
        else:
            raise RuntimeError("Failed to start monitoring.")

        with mon_repo.add_target_group(f"{self.experiment_name}.{self.trial_name}",
                                       mon_addresses,
                                       delete_on_exit=True):
            logger.info("Start workers...")
            self.__control.group_request("start")
            logger.info("Started.")
            try:
                self.wait(timeout=None, check_status=check_worker_status, remove_status=remove_worker_status)
            except distributed.system.worker_base.WorkerException as e:
                logger.error(e)
                self.interrupt(wait_timeout=30)
            except KeyboardInterrupt:
                logger.info("Interrupted.")
                self.interrupt(wait_timeout=30)

    def wait(self, timeout: Optional[int], check_status: Tuple[Wss, ...], remove_status: Tuple[Wss, ...]):
        deadline = None if timeout is None else time.time() + timeout
        left = set(self.__control.worker_names)
        num_jobs_left = len(left)
        logger.info(f"Waiting for {num_jobs_left} jobs.")
        current_status = {name: Wss.UNKNOWN for name in self.__control.worker_names}
        while len(left) > 0:
            logger.debug(
                f"JOBS LEFT: {[str(len([l for l in left if job_type in l])) + ' ' + job_type for job_type in set([job_id.split('/')[0] for job_id in left])]}"
            )
            if len(left) < num_jobs_left:
                num_jobs_left = len(left)
                logger.info(f"Waiting for {num_jobs_left} jobs.")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timeout waiting for {self.experiment_name, self.trial_name}: {', '.join(sorted(left))}")
            for worker_name, worker_status in self.__control.pulse().items():
                if worker_status in check_status:
                    raise distributed.system.worker_base.WorkerException(worker_name, worker_status,
                                                                         "experiment is running.")
                if worker_status in remove_status:
                    if worker_name in current_status:
                        logger.debug(f"Worker {worker_name} is {worker_status}. Removed from waiting list.")
                        current_status.pop(worker_name)
                    else:
                        pass
                else:
                    if current_status.get(worker_name, None) != worker_status:
                        current_status.update({worker_name: worker_status})
                        logger.debug(f"Update worker status: {worker_name} -> {worker_status}")

            left = set(current_status.keys())
            time.sleep(10)

    def stop(self):
        """Stop the experiment.
        Note:
            This method assumes that the controller and scheduler is connected to the correct workers. To ensure this,
            call controller.reconnect before your call controller.stop.
        """
        raise NotImplementedError()

    def interrupt(self, wait_timeout=120):
        """Interrupt the experiment.
        """
        logger.info("Interrupting experiment")
        self.__control.group_request("interrupt", wait_response=False)
        try:
            self.wait(timeout=wait_timeout,
                      check_status=(),
                      remove_status=(Wss.ERROR, Wss.LOST, Wss.COMPLETED, Wss.INTERRUPTED))
        except TimeoutError:
            raise RuntimeError(f"Fail to interrupt workers, timeout={wait_timeout}.")
