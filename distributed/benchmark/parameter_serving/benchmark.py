"""This benchmark simulates a simple RL-training loop and measures the stalesss.
See README.md for more information.
"""
import argparse
import os
import signal
import socket
import subprocess
import time

from .utils import EXPERIMENT_NAME


trainer_command = "python3 -m distributed.benchmark.parameter_serving.pseudo_trainer " \
                  "--data_size {data_size} " \
                  "--push_frequency {push_frequency} --max_version {max_version} --max_time {max_time} " \
                  "--LOGLEVEL {LOGLEVEL} --mode {mode} --trial_name {trial_name}"

worker_command = "python3 -m distributed.benchmark.parameter_serving.pseudo_worker " \
                 "--pull_frequency {pull_frequency} " \
                 "--post_frequency {post_frequency} " \
                 "--step_frequency {step_frequency} --max_version {max_version} --max_time {max_time} " \
                 "--LOGLEVEL {LOGLEVEL} --mode {mode} --trial_name {trial_name}"

package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
srun_command = f"srun -n {{ntasks}} --job-name={{job_name}} --container-image=marl/marl-cpu --container-mount-home " \
               f"--export=PYTHONPATH={os.path.dirname(__file__)} --container-mounts=/data {{flags}} " \
               f"{{cmd}}"

if __name__ == '__main__':
    is_frl = socket.gethostname().startswith("frl") or socket.gethostname().startswith(
        "dev") or socket.gethostname().startswith("ctrl")
    print(f"is_frl: {is_frl}", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="filesystem", required=False)
    parser.add_argument("--push_frequency", type=float, required=True, help="seconds per push")
    parser.add_argument("--data_size", type=int, required=True, help="size of each ckpt, in MegaBytes")
    parser.add_argument("--pull_frequency", type=float, required=True, help="time interval per pull seconds.")
    parser.add_argument("--post_frequency", type=int, required=True, help="time interval per post seconds")
    parser.add_argument("--step_frequency", type=float, required=True, help="time interval per step")
    parser.add_argument("--max_version", type=int, required=True, help="maximum version for this run")
    parser.add_argument("--max_time", type=int, required=True, help="maximum time this run.")
    parser.add_argument("--workers", type=int, required=False, default=1)
    parser.add_argument("--trainer_node", type=str, default="frl8g135", required=False)
    parser.add_argument("--workers-per-node", type=int, default=1, required=False)
    parser.add_argument("--trainer-cpus", type=int, default=4, required=False)
    parser.add_argument("--trial_name", type=str, required=True)
    parser.add_argument("--LOGLEVEL", type=str, required=False, default="INFO")
    args = parser.parse_args()
    trainer_command = trainer_command.format(data_size=args.data_size,
                                             push_frequency=args.push_frequency,
                                             max_version=args.max_version,
                                             max_time=args.max_time,
                                             LOGLEVEL=args.LOGLEVEL,
                                             mode=args.mode,
                                             trial_name=args.trial_name)
    worker_command = worker_command.format(pull_frequency=args.pull_frequency,
                                           post_frequency=args.post_frequency,
                                           step_frequency=args.step_frequency,
                                           max_version=args.max_version,
                                           max_time=args.max_time,
                                           LOGLEVEL=args.LOGLEVEL,
                                           mode=args.mode,
                                           trial_name=args.trial_name)

    if is_frl:
        cmd0 = srun_command.format(
            ntasks=1,
            cmd=f"python3 -m apps.remote reset_name_resolve -e {EXPERIMENT_NAME} -f {args.trial_name}",
            flags="",
            job_name="parameter_serving_benchmark_setup",
        )
        process0 = subprocess.Popen(cmd0, shell=True)
        process0.wait()

        trainer_command = srun_command.format(
            ntasks=1,
            cmd=trainer_command,
            flags=f"--nodelist={args.trainer_node} -c {args.trainer_cpus}",
            job_name="parameter_serving_benchmark_trainer",
        )
        worker_command = srun_command.format(
            ntasks=args.workers,
            cmd=worker_command,
            flags=f"--exclude={args.trainer_node} --ntasks-per-node={args.workers_per_node}",
            job_name="parameter_serving_benchmark_worker",
        )
        print(trainer_command)
        print(worker_command)
        process1 = subprocess.Popen(trainer_command, shell=True)
        process2 = subprocess.Popen(worker_command, shell=True)

        try:
            process1.wait()
            process2.wait()
        except KeyboardInterrupt:
            killtrainer = subprocess.Popen("scancel -n parameter_serving_benchmark_trainer", shell=True)
            killworker = subprocess.Popen("scancel -n parameter_serving_benchmark_worker", shell=True)
            process1.wait(timeout=10)
            process2.wait(timeout=10)
    else:
        processes = []
        processes.append(subprocess.Popen(trainer_command, shell=True))
        for _ in range(args.workers):
            processes.append(subprocess.Popen(worker_command, shell=True))
        try:
            for p in processes:
                p.wait()
        except KeyboardInterrupt:
            for p in processes:
                p.send_signal(sig=signal.SIGINT)
                p.wait(timeout=10)
