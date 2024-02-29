from typing import Optional, Dict, Any, List
import dataclasses
import enum
import logging
import prometheus_client
import queue
import socket
import threading
import time
import wandb

from api import config as config_pkg
from base.gpu_utils import set_cuda_device
import base.cluster
import base.names
import base.name_resolve
import base.network

logger = logging.getLogger("worker")


class WorkerException(Exception):

    def __init__(self, worker_name, worker_status, scenario):
        super(WorkerException, self).__init__(f"Worker {worker_name} is {worker_status} while {scenario}")
        self.worker_name = worker_name
        self.worker_status = worker_status
        self.scenario = scenario


class WorkerServerStatus(str, enum.Enum):
    """List of all possible Server status. This is typically set by workers hosting the server, and
    read by the controller.
    """
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"

    UNKNOWN = "UNKNOWN"  # CANNOT be set.
    INTERRUPTED = "INTERRUPTED"
    ERROR = "ERROR"
    LOST = "LOST"  # CANNOT be set


class WorkerServer:
    """The abstract class that defines how a worker exposes RPC stubs to the controller.
    """

    def __init__(self, worker_name):
        """Specifies the name of the worker that WorkerControlPanel can used to find and manage.
        Args:
            worker_name: Typically "<worker_type>/<worker_index>".
        """
        self.worker_name = worker_name

    def register_handler(self, command, fn):
        """Registers an RPC command. The handler `fn` shall be called when `self.handle_requests()` sees an
        incoming command of the registered type.
        """
        raise NotImplementedError()

    def handle_requests(self, max_count=None):
        raise NotImplementedError()

    def set_status(self, status: WorkerServerStatus):
        raise NotImplementedError()


@dataclasses.dataclass
class MonitoringInformation:
    host: str
    prometheus_port: int


@dataclasses.dataclass
class PollResult:
    # Number of total samples and batches processed by the worker. Specifically:
    # - For an actor worker, sample_count = batch_count = number of env.step()-s being executed.
    # - For a policy worker, number of inference requests being handled, versus how many batches were made.
    # - For a trainer worker, number of samples & batches fed into the trainer (typically GPU).
    sample_count: int
    batch_count: int


class Worker:
    """The worker base class that provides general methods and entry point.

    For simplicity, we use a single-threaded pattern in implementing the worker RPC server. Logic
    of every worker are executed via periodical calls to the poll() method, instead of inside
    another thread or process (e.g. the gRPC implementation). A subclass only needs to implement
    poll() without duplicating the main loop.

    The typical code on the worker side is:
        worker = make_worker()  # Returns instance of Worker.
        worker.run()
    and the later is standardized here as:
        while exit command is not received:
            if worker is started:
                worker.poll()
    """

    def __init__(self, server: Optional[WorkerServer] = None):
        """Initializes a worker server.

        Args:
            server: The RPC server API for the worker to register handlers and poll requests.
        """
        self.__running = False
        self.__exiting = False
        self.config = None
        self.__is_configured = False
        self._monitoring_info = None

        self._server = server
        if server is not None:
            server.register_handler('configure', self.configure)
            server.register_handler('reconfigure', self.reconfigure)
            server.register_handler('start_monitoring', self.start_monitoring)
            server.register_handler('start', self.start)
            server.register_handler('pause', self.pause)
            server.register_handler('exit', self.exit)
            server.register_handler('interrupt', self.interrupt)
            server.register_handler('ping', lambda: "pong")

        self.logger = logging.getLogger("worker")
        self.__worker_type = None
        self.__worker_index = None
        self.__monitoring_enabled = False
        self.__last_successful_poll_time = None

        # Monitoring related.
        self._start_time_ns = None
        self.__metric_sample_count: Optional[prometheus_client.Counter] = None
        self.__metric_batch_count: Optional[prometheus_client.Counter] = None
        self.__metric_wait_seconds: Optional[prometheus_client.Histogram] = None
        self.__wandb_run = None
        self.__wandb_args = None
        self.__log_wandb = None
        self.__wandb_last_log_time_ns = None

        self.__set_status(WorkerServerStatus.READY)

    def __set_status(self, status: WorkerServerStatus):
        if self._server is not None:
            self.logger.debug(f"Setting worker server status to {status}")
            self._server.set_status(status)

    def __del__(self):
        if self.__wandb_run is not None:
            self.__wandb_run.finish()

    @property
    def is_configured(self):
        return self.__is_configured

    @property
    def wandb_run(self):
        if self.__wandb_run is None:
            wandb.login()
            for _ in range(10):
                try:
                    self.__wandb_run = wandb.init(dir=base.cluster.get_user_tmp(),
                                                  config=self.config,
                                                  resume="allow",
                                                  **self.__wandb_args)
                    break
                except wandb.errors.UsageError as e:
                    time.sleep(5)
            else:
                raise e
        return self.__wandb_run

    def _new_wandb_run(self, new_wandb_args):
        self.__wandb_run.finish()
        self.__wandb_run = None
        self.__wandb_args.update(new_wandb_args)

    def _reconfigure(self, **kwargs) -> config_pkg.WorkerInformation:
        """Implemented by sub-classes."""
        raise NotImplementedError()

    def _configure(self, config) -> config_pkg.WorkerInformation:
        """Implemented by sub-classes."""
        raise NotImplementedError()

    def _poll(self) -> PollResult:
        """Implemented by sub-classes."""
        raise NotImplementedError()

    def _stats(self) -> Dict[str, Any]:
        """Implemented by sub-classes."""
        return {}

    def configure(self, config):
        assert not self.__running
        self.logger.info("Configuring with: %s", config)

        r = self._configure(config)
        self.__worker_type = r.worker_type
        self.__worker_index = r.worker_index
        self.logger = logging.getLogger(r.worker_type + "-worker")
        if r.host_key is not None:
            self.__host_key(
                base.names.worker_key(experiment_name=r.experiment_name,
                                      trial_name=r.trial_name,
                                      key=r.host_key))
        if r.watch_keys is not None:
            keys = [r.watch_keys] if isinstance(r.watch_keys, str) else r.watch_keys
            self.__watch_keys([
                base.names.worker_key(experiment_name=r.experiment_name, trial_name=r.trial_name, key=k)
                for k in keys
            ])

        self.__wandb_run = None  # This will be lazy created by self.wandb_run().
        self.__log_wandb = (self.__worker_index == 0
                            and self.__worker_type == "trainer") if r.log_wandb is None else r.log_wandb
        self.__wandb_args = dict(
            entity=r.wandb_entity,
            project=r.wandb_project or f"{r.experiment_name}",
            group=r.wandb_group or r.trial_name,
            job_type=r.wandb_job_type or f"{r.worker_type}",
            name=r.wandb_name or f"{r.policy_name or r.worker_index}",
            id=
            f"{r.experiment_name}_{r.trial_name}_{r.policy_name or 'unnamed'}_{r.worker_type}_{r.worker_index}",
            settings=wandb.Settings(start_method="fork"),
        )

        self.__is_configured = True
        self.logger.info("Configured successfully")

    def reconfigure(self, **kwargs):
        assert not self.__running
        self.__is_configured = False
        self.logger.info(f"Reconfiguring with: {kwargs}")
        self._reconfigure(**kwargs)
        self.__is_configured = True
        self.logger.info("Reconfigured successfully")

    def start(self):
        self.logger.info("Starting worker")
        self.__running = True
        self.__set_status(WorkerServerStatus.RUNNING)

    def pause(self):
        self.logger.info("Pausing worker")
        self.__running = False
        self.__set_status(WorkerServerStatus.PAUSED)

    def exit(self):
        self.logger.info("Exiting worker")
        self.__set_status(WorkerServerStatus.COMPLETED)
        self.__exiting = True

    def interrupt(self):
        self.logger.info("Worker interrupted by remote control.")
        self.__set_status(WorkerServerStatus.INTERRUPTED)
        raise WorkerException(worker_name="worker",
                              worker_status=WorkerServerStatus.INTERRUPTED,
                              scenario="running")

    def run(self):
        self._start_time_ns = time.monotonic_ns()
        self.logger.info("Running worker now")
        try:
            while not self.__exiting:
                if self.__running:
                    if not self.__is_configured:
                        raise RuntimeError("Worker is not configured")
                    start_time = time.monotonic_ns()
                    r = self._poll()
                    if r.sample_count == r.batch_count == 0:
                        if self.__worker_type != "actor":
                            time.sleep(0.005)
                    else:
                        if self.__monitoring_enabled:
                            # Record monitoring metrics.
                            wait_seconds = 0.0
                            if self.__last_successful_poll_time is not None:
                                # Account the waiting time since the last successful step.
                                wait_seconds = (start_time - self.__last_successful_poll_time) / 1e9
                            self.__last_successful_poll_time = time.monotonic_ns()
                            self.__metric_sample_count.inc(r.sample_count)
                            self.__metric_batch_count.inc(r.batch_count)
                            self.__metric_wait_seconds.observe(wait_seconds)
                            self.__maybe_log_wandb()
                else:
                    time.sleep(0.05)
                if self._server is not None:
                    self._server.handle_requests()
        except KeyboardInterrupt:
            self.exit()
        except Exception as e:
            if isinstance(e, WorkerException):
                raise e
            self.__set_status(WorkerServerStatus.ERROR)
            raise e

    def __host_key(self, key: str):
        self.logger.info(f"Hosting key: {key}")
        base.name_resolve.add(key, "up", keepalive_ttl=15, replace=True, delete_on_exit=True)

    def __watch_keys(self, keys: List[str]):
        self.logger.info(f"Watching keys: {keys}")
        base.name_resolve.watch_names(keys, call_back=self.exit)

    def __maybe_log_wandb(self):
        if not self.__log_wandb:
            return
        now = time.monotonic_ns()
        if self.__wandb_last_log_time_ns is not None:  # Log with a frequency.
            if (now - self.__wandb_last_log_time_ns) / 1e9 < _WANDB_LOG_FREQUENCY_SECONDS:
                return
        self.__wandb_last_log_time_ns = now
        duration = (now - self._start_time_ns) / 1e9
        stats = {
            "samples": self.__metric_sample_count._value.get() / duration,
            "batches": self.__metric_batch_count._value.get() / duration,
            "idleTime": self.__metric_wait_seconds._sum.get() / duration,
        }
        stats.update(self._stats())
        self.wandb_run.log(stats)


class MappingThread:
    """Wrapped of a mapping thread.
    A mapping thread gets from up_stream_queue, process data, and puts to down_stream_queue.
    """

    def __init__(self,
                 map_fn,
                 interrupt_flag,
                 upstream_queue,
                 downstream_queue: queue.Queue = None,
                 cuda_device=None):
        """Init method of MappingThread for Policy Workers.

        Args:
            map_fn: mapping function.
            interrupt_flag: main thread sets this value to True to interrupt the thread.
            upstream_queue: the queue to get data from.
            downstream_queue: the queue to put data after processing. If None, data will be discarded after processing.
        """
        self.__map_fn = map_fn
        self.__interrupt = interrupt_flag
        self.__upstream_queue = upstream_queue
        self.__downstream_queue = downstream_queue
        self.__thread = threading.Thread(target=self._run, daemon=True)
        self.__cuda_device = cuda_device

    def is_alive(self) -> bool:
        """Check whether the thread is alive.

        Returns:
            alive: True if the wrapped thread is alive, False otherwise.
        """
        return self.__interrupt or self.__thread.is_alive()

    def start(self):
        """Start the wrapped thread.
        """
        self.__thread.start()

    def join(self):
        """Join the wrapped thread.
        """
        self.__thread.join()

    def _run(self):
        if self.__cuda_device is not None:
            set_cuda_device(self.__cuda_device)
        while not self.__interrupt:
            self._run_step()

    def _run_step(self):
        try:
            data = self.__upstream_queue.get(timeout=1)
            data = self.__map_fn(data)
            if self.__downstream_queue is not None:
                self.__downstream_queue.put(data)
        except queue.Empty:
            pass

    def stop(self):
        """Stop the wrapped thread.
        """
        self.__interrupt = True
        if self.__thread.is_alive():
            self.__thread.join()
