from typing import List
import api.config as config_package
import argparse
import logging
import getpass
import os
import re

# import apps.viewer  # TODO: apps.viewer
from distributed.system import RL_WORKERS
import api.config
import distributed.infra.scheduler.client
# import codespace.experiment
import legacy.experiments

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"

logger = logging.getLogger("main")


def main_start(args):

    def __submit_workers(worker_type, scheduling_configs, partition, ps_node) -> List[str]:
        assert worker_type in [
            "actor", "policy", "trainer", "eval_manager", "population_manager", "buffer", "parameter_server",
            "shared_memory_worker"
        ], worker_type
        if len(scheduling_configs) == 0:
            return []

        short_name = RL_WORKERS[worker_type].short_name
        start_index = 0
        scheduled_jobs = []
        for sch_cfg in scheduling_configs:
            sch_cfg: api.config.TasksGroup
            job_environs = {
                "PYTHONPATH": os.path.dirname(os.path.dirname(__file__)),
                "NCCL_IB_DISABLE": "1",
                "NCCL_P2P_DISABLE": "1",
                "NCCL_IGNORE_DISABLED_P2P": "1",
                "WANDB_MODE": args.wandb_mode,
                "LOGLEVEL": args.LOGLEVEL,
            }
            cmd = f"python3 {'' if args.debug else '-O'} -m apps.remote worker -w {worker_type} " \
                  f"-e {args.experiment_name} -f {trial_name} -i {{index}}"
            logger.debug(f"Scheduling worker {worker_type}, {scheduling_configs}")

            nodelist = sch_cfg.scheduling.node_list
            exclude = sch_cfg.scheduling.exclude
            if ps_node is not None:
                if worker_type == "parameter_server":
                    nodelist = ps_node
                    exclude = None
                elif worker_type in ("actor", "policy", "buffer"):
                    exclude = ps_node if exclude is None else exclude + "," + ps_node

            scheduled_jobs.append(
                scheduler.submit_array(short_name,
                                       cmd,
                                       count=sch_cfg.count,
                                       cpu=sch_cfg.scheduling.cpu,
                                       gpu=sch_cfg.scheduling.gpu,
                                       gpu_type=sch_cfg.scheduling.gpu_type,
                                       mem=sch_cfg.scheduling.mem,
                                       partition=partition,
                                       container_image=args.image_name or sch_cfg.scheduling.container_image,
                                       node_list=nodelist,
                                       node_type=sch_cfg.scheduling.node_type,
                                       exclude=exclude,
                                       start_index=start_index,
                                       environment_vars=job_environs,
                                       expr_name=args.experiment_name,
                                       trial_name=trial_name,
                                       debug=args.debug))
        return scheduled_jobs

    trial_name = args.trial_name or f"test-{getpass.getuser()}"
    experiments = config_package.make_experiment(args.experiment_name)
    logger.info(
        f"Experiment has {len(experiments)} sub configurations: {[e.__class__.__name__ for e in experiments]}"
    )
    skip_indices = [int(i)
                    for i in args.skip_sub_config.split(",")] if args.skip_sub_config is not None else []

    scheduler = distributed.infra.scheduler.client.make(mode=args.mode,
                                                        job_name=f"{args.experiment_name}_{trial_name}")
    assert len(experiments) == 1, "Single experiments experiments only."

    for i, experiment in enumerate(experiments):
        if i in skip_indices:
            logger.info(f"Skipping sub configuration: {experiment.__class__.__name__}")
            continue

        # Parameter Server Worker uses PGM protocol and does not support back-loop. ps_node must be specified.
        setup = experiment.scheduling_setup()
        if setup.parameter_server_worker:
            if args.ps_node is None:
                raise KeyError(
                    "--ps_node must be specified when experiment contains parameter server worker.")

        simple_env_vars = {
            "PYTHONPATH": os.path.dirname(os.path.dirname(__file__)),
            "LOGLEVEL": args.LOGLEVEL
        }
        logger.info(f"Resetting name resolving repo...")
        scheduler.submit(
            "setup",
            f"python3 {'' if args.debug else '-O'} -m apps.remote reset_name_resolve -e {args.experiment_name} -f {trial_name}",
            environment_vars=simple_env_vars,
            expr_name=args.experiment_name,
            trial_name=trial_name,
            debug=args.debug)
        scheduler.wait(timeout=360, update=True)
        logger.info(f"Resetting name resolving repo... Done.")

        logger.info(f"Running sub configuration: {experiment.__class__.__name__}")
        # Schedule controller
        job_ids = [
            scheduler.submit_array(
                task_name="ctl",
                cmd=
                f"python3 {'' if args.debug else '-O'} -m apps.remote controller -e {args.experiment_name} -f "
                f"{trial_name} --config_index {i} "
                f"--{'ignore_worker_error' if args.ignore_worker_error else 'raise_worker_error'}",
                count=1,
                cpu=1,
                gpu=0,
                mem=128,
                container_image=args.image_name or setup.controller_image,
                environment_vars=simple_env_vars,
                expr_name=args.experiment_name,
                trial_name=trial_name,
                debug=args.debug)
        ]

        workers_configs = ((k, getattr(setup, v.config_field_name)) for k, v in RL_WORKERS.items())

        submitted_workers = []
        for name, scheduling_setup in workers_configs:
            if not isinstance(scheduling_setup, list):
                scheduling_setup = [scheduling_setup]
            job_ids.extend(__submit_workers(name, scheduling_setup, args.partition, args.ps_node))
            if scheduling_setup:
                submitted_workers.append(name)

        job_name = f"{args.experiment_name}_{trial_name}"
        for name in submitted_workers:
            logger.info(
                f"To check {name} output: \n"
                f"\t`tail -f {distributed.infra.scheduler.client.log_path(job_name=job_name, task_name=RL_WORKERS[name].short_name)}`"
            )
        logger.info(
            f"To check controller output: \n"
            f"\t`tail -f {distributed.infra.scheduler.client.log_path(job_name=job_name, task_name='ctl')}`")
        logger.info(f"To stop experiment: `scancel {','.join([j for j in job_ids if j is not None])}`")

        setup: api.config.ExperimentConfig = experiment.initial_setup()
        try:
            scheduler.wait(timeout=setup.timeout_seconds)
        except (KeyboardInterrupt, distributed.infra.scheduler.client.TaskException):
            scheduler.stop_all()
            raise


def main_stop(args):
    mode = args.mode or "slurm"
    assert mode == "slurm", "Only slurm experiment is supported."
    scheduler = distributed.infra.scheduler.client.make(mode=args.mode,
                                                        job_name=f"{args.experiment_name}_{args.trial_name}")
    scheduler.find_all()
    scheduler.stop_all()


def main_find_config(args):
    exp_names = [x for x in config_package.ALL_EXPERIMENT_CLASSES if re.match(args.regex, x)]
    if len(exp_names) == 0:
        print("No matched experiment names.")
    if len(exp_names) > 20:
        response = input(f"Found {len(exp_names)} experiments, list all?(y/n)")
        if response != "y":
            return
    for exp_name in exp_names:
        print(exp_name)


def main():
    parser = argparse.ArgumentParser(prog="marl")
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("start", help="starts an experiment")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True, help="name of the experiment")
    subparser.add_argument("--trial_name",
                           "-f",
                           type=str,
                           default=None,
                           help="trial name; by default uses '<USER>-test'")
    subparser.add_argument(
        "--mode",
        default="slurm",
        choices=["slurm", "slurm_mix", "slurm_trivial", "slurm_gpu", "slurm_one", "slurm_pw"])
    subparser.add_argument("--partition", default="dev", help="slurm partition to schedule the trial")
    subparser.add_argument("--skip_sub_config",
                           "-s",
                           type=str,
                           default=None,
                           help="comma delimited indices of sub experiments.")
    subparser.add_argument("--wandb_mode",
                           type=str,
                           default="disabled",
                           choices=["online", "offline", "disabled"])
    subparser.add_argument("--image_name",
                           type=str,
                           required=False,
                           default=None,
                           help="if specified, all workers will use this image. Useful in CI/CD pipeline.")
    subparser.add_argument("--ps_node", type=str, default=None)
    subparser.add_argument("--LOGLEVEL", type=str, default="INFO")
    subparser.add_argument("--ignore_worker_error", action="store_true")
    subparser.add_argument("--debug",
                           action="store_true",
                           help="If True, activate all assertions in the code.")
    ## pass experiment config by arguments

    subparser.set_defaults(ignore_worker_error=False)
    subparser.set_defaults(func=main_start)

    subparser = subparsers.add_parser("stop", help="stops an experiment. only slurm experiment is supported.")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True, help="name of the experiment")
    subparser.add_argument("--trial_name", "-f", type=str, required=True, help="name of the trial")
    subparser.add_argument("--mode", default="slurm", choices=["local", "slurm"])
    subparser.set_defaults(func=main_stop)

    subparser = subparsers.add_parser("find_config",
                                      help="find configuration by matching regular expression.")
    subparser.add_argument("--regex", "-r", type=str, required=True)
    subparser.set_defaults(func=main_find_config)

    # subparser = subparsers.add_parser("view", help="render the model of a trial")
    # subparser.add_argument("--experiment_name", "-e", type=str, default=None)
    # subparser.add_argument("--trial_name", "-f", type=str, default=None)
    # subparser.add_argument("--episodes", type=int, default=5)
    # subparser.add_argument("--ckpt_versions", nargs='+', default=None)
    # subparser.set_defaults(func=main_view)

    args = parser.parse_args()
    logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=getattr(args, "LOGLEVEL", "INFO"))
    args.func(args)


if __name__ == "__main__":
    main()
