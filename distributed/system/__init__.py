import collections
import dataclasses
import importlib
import logging
import os
import traceback

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))


@dataclasses.dataclass
class WorkerSpec:
    """Description of a worker implementation.
    """
    short_name: str  # short name is used in file names.
    config_field_name: str  # Used in experiment/scheduling configuration(api.config).
    class_name: str  # The class name of the implementation.
    module: str  # The module path to find the worker class.

    def load_worker(self):
        module = importlib.import_module(self.module)
        return getattr(module, self.class_name)


actor_worker = WorkerSpec(short_name='aw',
                          class_name="ActorWorker",
                          config_field_name="actors",
                          module="distributed.system.actor_worker")
buffer_worker = WorkerSpec(short_name='bw',
                           class_name="BufferWorker",
                           config_field_name="buffers",
                           module="distributed.system.buffer_worker")
eval_manager = WorkerSpec(short_name='em',
                          class_name="EvalManager",
                          config_field_name="eval_managers",
                          module="distributed.system.eval_manager")
policy_worker = WorkerSpec(short_name='pw',
                           class_name="PolicyWorker",
                           config_field_name="policies",
                           module="distributed.system.policy_worker")
trainer_worker = WorkerSpec(short_name='tw',
                            class_name="TrainerWorker",
                            config_field_name="trainers",
                            module="distributed.system.trainer_worker")
population_manager = WorkerSpec(short_name="pm",
                                class_name="PopulationManager",
                                config_field_name="population_manager",
                                module="distributed.system.population_manager")
parameter_server = WorkerSpec(short_name="ps",
                              class_name="ParameterServerWorker",
                              config_field_name="parameter_server_worker",
                              module="distributed.system.parameter_server_worker")
shared_memory_worker = WorkerSpec(short_name="sms",
                                  class_name="SharedMemoryWorker",
                                  config_field_name="shared_memory_worker",
                                  module="distributed.system.shared_memory_worker")

RL_WORKERS = collections.OrderedDict()
RL_WORKERS["parameter_server"] = parameter_server
RL_WORKERS["trainer"] = trainer_worker
RL_WORKERS["buffer"] = buffer_worker
RL_WORKERS["policy"] = policy_worker
RL_WORKERS["eval_manager"] = eval_manager
RL_WORKERS["population_manager"] = population_manager
RL_WORKERS["actor"] = actor_worker
RL_WORKERS["shared_memory_worker"] = shared_memory_worker


def run_worker(worker_type,
               experiment_name,
               trial_name,
               worker_name,
               mp_locks=None,
               env_vars={},
               timeout=None):
    """Run one worker
    Args:
        worker_type: string, one of the worker types listed above,
        experiment_name: string, the experiment this worker belongs to,
        trial_name: string, the specific trial this worker belongs to,
        worker_name: name given to the worker, typically "<worker_type>/<worker_index>"
    """
    for k, v in env_vars.items():
        os.environ[k] = v

    worker_class = RL_WORKERS[worker_type].load_worker()
    server = make_worker_server(experiment_name=experiment_name,
                                trial_name=trial_name,
                                worker_name=worker_name,
                                timeout=timeout)
    worker = worker_class(server=server, lock=mp_locks)
    try:
        worker.run()
    except Exception as e:
        logging.error("Worker %s failed with exception: %s", worker_name, e)
        logging.error(traceback.format_exc())
        raise e


def make_controller(*args, **kwargs):
    """Make a distributed reinforcement learning controller.
    Returns:
        a controller.
    """
    module = importlib.import_module("distributed.system.controller")
    return getattr(module, "Controller")(*args, **kwargs)


def make_worker_server(*args, **kwargs):
    """Make a worker server, so we can establish remote control to the worker.
    Returns:
        a worker server.
    """
    module = importlib.import_module("distributed.system.worker_control")
    return getattr(module, "make_server")(*args, **kwargs)
