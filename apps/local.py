import argparse
import logging
import multiprocessing
import os

import api.config
from legacy import algorithm, environment, population_based_training, experiments
# import codespace.experiment
# import codespace.implementation
import base.name_resolve
import local.system

base.name_resolve.reconfigure("memory")
LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"

logger = logging.getLogger("Marl")


def run_worker(worker_type, config, device: str):
    if "MARL_CUDA_DEVICES" in os.environ.keys():
        os.environ.pop("MARL_CUDA_DEVICES")
    if device.isdigit() or device == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    else:
        raise ValueError(f"Invalid Device for {worker_type}: {device}")

    workers = {
        "actor": local.system.basic.actor_worker.ActorWorker,
        "policy": local.system.basic.policy_worker.PolicyWorker,
        "trainer": local.system.basic.trainer_worker.TrainerWorker,
        "eval_manager": local.system.basic.eval_manager.EvalManager,
    }
    worker_class = workers[worker_type]
    worker = worker_class()
    worker.configure(config=config)
    worker.start()
    worker.run()


def config_local_worker_and_run(worker_type, worker_configs, devices):
    logger.info(f"Running {len(worker_configs)} {worker_type} workers on {devices if devices else 'cpu'}.")
    devices = devices.split(",")
    ps = [
        multiprocessing.Process(target=run_worker, args=(worker_type, c, devices[i % len(devices)]))
        for i, c in enumerate(worker_configs)
    ]
    return ps


def run_local(args):
    exps = api.config.make_experiment(args.experiment_name)
    if len(exps) > 1:
        raise NotImplementedError()
    exp: api.config.Experiment = exps[0]
    setup = exp.initial_setup()
    setup.set_worker_information(args.experiment_name, args.trial_name)
    logger.info(
        f"Running {exp.__class__.__name__} experiment_name: {args.experiment_name} trial_name {args.trial_name}"
    )

    logger.info("Making Sample Streams.")
    sample_streams = {}

    for e in setup.eval_managers:
        if isinstance(e.eval_sample_stream, str):
            if e.eval_sample_stream not in sample_streams.keys():
                q = multiprocessing.Queue()
                sample_streams[e.eval_sample_stream] = local.system.basic.local_sample.make_local_pair(q)
            e.eval_sample_stream = sample_streams[e.eval_sample_stream][1]
        else:
            raise NotImplementedError()

    for t in setup.trainers:
        if isinstance(t.sample_stream, str):
            if t.sample_stream not in sample_streams.keys():
                q = multiprocessing.Queue()
                sample_streams[t.sample_stream] = local.system.basic.local_sample.make_local_pair(q)
            t.sample_stream = sample_streams[t.sample_stream][1]
        else:
            raise NotImplementedError()

    inference_streams = {}
    for a in setup.actors:
        for i, inf in enumerate(a.inference_streams):
            if isinstance(inf, str):
                req_q = multiprocessing.Queue()
                resp_q = multiprocessing.Queue()
                a.inference_streams[i] = local.system.basic.local_inference.make_local_client(req_q, resp_q)
                if inf not in inference_streams.keys():
                    inference_streams[inf] = [(req_q, resp_q)]
                else:
                    inference_streams[inf].append((req_q, resp_q))
        for i, spl in enumerate(a.sample_streams):
            if isinstance(spl, str):
                a.sample_streams[i] = sample_streams[spl][0]

    for i, p in enumerate(setup.policies):
        if isinstance(p.inference_stream, str):
            if p.inference_stream not in inference_streams.keys():
                raise KeyError(p.inference_stream)
            p.inference_stream = local.system.basic.local_inference.make_local_server(
                req_qs=[q[0] for q in inference_streams[p.inference_stream][i::len(setup.policies)]],
                resp_qs=[q[1] for q in inference_streams[p.inference_stream][i::len(setup.policies)]])
        else:
            raise NotImplementedError()

    workers = config_local_worker_and_run("actor", setup.actors, args.actor_devices) + \
        config_local_worker_and_run("policy", setup.policies, args.policy_devices) + \
        config_local_worker_and_run("trainer", setup.trainers, args.trainer_devices)

    for w in workers:
        w.start()

    for w in workers:
        w.join(timeout=3000)


def main():
    parser = argparse.ArgumentParser(prog="marl")
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("run", help="starts a basic experiment")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True, help="name of the experiment")
    subparser.add_argument("--trial_name", "-f", type=str, required=True, help="name of the trial")
    subparser.add_argument("--wandb_mode",
                           type=str,
                           default="disabled",
                           choices=["online", "offline", "disabled"])
    subparser.add_argument("--trainer_devices", type=str, required=False, default="")
    subparser.add_argument("--policy_devices", type=str, required=False, default="")
    subparser.add_argument("--buffer_devices", type=str, required=False, default="")
    subparser.add_argument("--actor_devices", type=str, required=False, default="")

    subparser.add_argument("--LOGLEVEL", type=str, default="INFO")
    subparser.set_defaults(func=run_local)

    args = parser.parse_args()
    logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=getattr(args, "LOGLEVEL", "INFO"))
    args.func(args)


if __name__ == '__main__':
    main()
