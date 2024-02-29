import argparse
import json
import logging
import multiprocessing
import os

multiprocessing.set_start_method("spawn", force=True)

# import codespace.experiment
# import codespace.implementation

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
logger = logging.getLogger("Main-Workers")


def main_reset_name_resolve(args):
    import distributed.base.name_resolve
    import base.names
    distributed.base.name_resolve.clear_subtree(
        base.names.trial_root(experiment_name=args.experiment_name, trial_name=args.trial_name))


def main_worker(args):
    import base.gpu_utils
    base.gpu_utils.isolate_cuda_device(args.worker_type, args.experiment_name, args.trial_name)

    from legacy import algorithm, environment, population_based_training, experiments
    import api.config
    from distributed.system import run_worker
    experiments = api.config.make_experiment(args.experiment_name)
    if len(experiments) != 1:
        raise RuntimeError("Only one experiment is supported for now.")
    timeout = experiments[0].initial_setup().timeout_seconds
    logger.info(f"Run {args.worker_type} worker with args: %s", args)
    assert not args.experiment_name.startswith("/"), args.experiment_name
    if args.group_size == 1:
        run_worker(worker_type=args.worker_type,
                   experiment_name=args.experiment_name,
                   trial_name=args.trial_name,
                   worker_name=f"{args.worker_type}/{args.group_id}",
                   timeout=timeout)
    else:
        id_start = args.group_id * args.group_size
        id_end = (args.group_id + 1) * args.group_size
        workers = []
        for worker_id in range(id_start, id_end):
            worker_name = f"{args.worker_type}/{worker_id}"
            worker_args = dict(worker_type=args.worker_type,
                               experiment_name=args.experiment_name,
                               trial_name=args.trial_name,
                               worker_name=worker_name,
                               timeout=timeout)
            p = multiprocessing.Process(target=run_worker, kwargs=worker_args)
            p.name = worker_name
            p.start()
            workers.append(p)

        logger.info(f"Waiting for {id_end-id_start} {args.worker_type} workers: [{id_start}-{id_end}]")

        worker_exit_code = 0
        while not worker_exit_code:
            for p in workers:
                if p.exitcode is None:
                    p.join(timeout=5)
                elif p.exitcode == 0:
                    pass
                else:
                    logger.error(f"{p.name} exitcode: {p.exitcode}")
                    worker_exit_code = p.exitcode
                    break
        for p in workers:
            if p.is_alive():
                p.kill()
        exit(worker_exit_code)


def mixed_main_worker(args):
    from legacy import algorithm, environment, population_based_training, experiments
    from distributed.system import run_worker
    from base.lock import ClientServerLock
    """ Only used by MixedSlurmSchedulerClient. 
    args:
    experiment_name: Experiment name.
    trial_name: Trial name.
    remote_config: A dict specifying how much workers of each type should be run on this remote instance.
                   Keys are worker types. Values are number of workers of corresponding type.
                   Example: {"tw": 1, "pw": 2} specify that this remote instance should run one tw and 2 pws.
    """
    # here args should be changed into json dict to support mixed worker group
    # logger.info(f"Run {args.worker_type} worker with args: %s", args)
    assert not args.experiment_name.startswith("/"), \
           f"Experiment name should not start with \"/\": {args.experiment_name}"
    # remote_config format should be {worker_type : worker_num}
    if os.path.isfile(args.remote_config):
        remote_config = json.load(open(args.remote_config, "r"))
    else:
        remote_config = json.loads(args.remote_config)
    workers = []
    locks = [ClientServerLock() for _ in range(8)]
    for worker_type, (id_start, id_end) in remote_config.items():
        for worker_id in range(id_start, id_end):
            worker_name = f"{worker_type}/{worker_id}"
            worker_args = dict(worker_type=worker_type,
                               experiment_name=args.experiment_name,
                               trial_name=args.trial_name,
                               worker_name=worker_name,
                               mp_locks=locks)
            p = multiprocessing.Process(target=run_worker, kwargs=worker_args)
            p.name = worker_name
            p.start()
            workers.append(p)
        logger.info(f"Waiting for {id_end-id_start} {worker_type} workers: [{id_start}-{id_end}]")

    worker_exit_code = 0
    while not worker_exit_code:
        for p in workers:
            if p.exitcode is None:
                p.join(timeout=5)
            elif p.exitcode == 0:
                pass
            else:
                logger.error(f"{p.name} exitcode: {p.exitcode}")
                worker_exit_code = p.exitcode
                break
    for p in workers:
        if p.is_alive():
            p.kill()
    exit(worker_exit_code)


def main_controller(args):
    """
    Args:
        args: argparse result including:
            experiment_name:
            trial_name:
            config_index: the index of experiment configuration (experiment may return multiple configurations)
            ignore_worker_error: bool, if False, stop the experiment when any worker(s) fail.
    """
    from legacy import algorithm, environment, population_based_training, experiments
    from distributed.system import make_controller
    import api.config
    logger.info("Running controller with args: %s", args)
    assert not args.experiment_name.startswith("/"), args.experiment_name
    controller = make_controller(experiment_name=args.experiment_name, trial_name=args.trial_name)
    experiments = api.config.make_experiment(args.experiment_name)
    controller.start(
        experiment=experiments[int(args.config_index)],
        ignore_worker_error=args.ignore_worker_error,
    )


def main():
    logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))

    parser = argparse.ArgumentParser(prog="marl")
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("controller", help="run a controller of experiment")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.add_argument("--config_index", type=int, required=True)
    subparser.add_argument("--ignore_worker_error", action="store_true")
    subparser.add_argument('--raise_worker_error', dest='ignore_worker_error', action='store_false')
    subparser.set_defaults(feature=False)
    subparser.set_defaults(func=main_controller)

    subparser = subparsers.add_parser("worker", help="run a standalone worker")
    subparser.add_argument("--worker_type", '-w', type=str, required=True)
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.add_argument("--group_id", "-i", type=int, required=True)
    subparser.add_argument("--group_size", "-g", type=int, required=False, default=1)
    subparser.set_defaults(func=main_worker)

    # To ensure old way works, add mixed_worker for new scheduling only.
    subparser = subparsers.add_parser("mixed_worker", help="run a standalone worker")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    # --remote_config argument could be a json string or a json file name.
    subparser.add_argument("--remote_config", "-r", type=str, required=False, default="\{\}")
    subparser.set_defaults(func=mixed_main_worker)

    subparser = subparsers.add_parser("reset_name_resolve", help="reset name resolve repo for a trial")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.set_defaults(func=main_reset_name_resolve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
