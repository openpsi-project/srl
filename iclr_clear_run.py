from typing import List
import os
import subprocess

TRIAL_NAME = "iclr24-benchmark"


def get_runned_exp_names():
    log_exps = []
    for logfile in os.listdir("z_iclr24logs/archive"):
        if logfile.startswith('.nfs'):
            continue
        log_exps.append(logfile.split('.out')[0].strip())
    for logfile in os.listdir("z_iclr24logs"):
        if logfile.startswith('.nfs'):
            continue
        if os.path.isfile(f"z_iclr24logs/{logfile}"):
            assert logfile.split('.out')[0].strip() not in log_exps, (logfile, log_exps)
            log_exps.append(logfile.split('.out')[0].strip())
    return log_exps


def main():
    log_exps = get_runned_exp_names()

    squeue_result = subprocess.check_output(r"squeue -u fw -o '%.18i %.80j' -h", shell=True,
                                            text=True).strip().split("\n")
    for squeue_line in squeue_result:
        if not squeue_line.strip():
            continue
        job_id, job_name = squeue_line.strip().split()
        if not any(job_name.startswith(exp_name) for exp_name in log_exps) and TRIAL_NAME in job_name:
            print(f"Cancel job: {job_id}")
            os.system(f"scancel {job_id}")

    # clear other checkpoint/log dirs (i.e., debug runs)
    print("Clearing logs and checkpoints...")
    for dir_name in os.listdir("/data/marl/logs/iclr24-benchmark/fw/"):
        exp_name, trial_name = dir_name.split("_")
        if trial_name != TRIAL_NAME or exp_name not in log_exps:
            print(f"Removing log dir: /data/marl/logs/iclr24-benchmark/fw/{dir_name}")
            os.system(f"rm -rf /data/marl/logs/iclr24-benchmark/fw/{dir_name}")

    for exp_name in os.listdir("/data/marl/checkpoints/iclr24-benchmark/fw/"):
        if exp_name not in log_exps:
            print(f"Removing checkpoint dir: /data/marl/checkpoints/iclr24-benchmark/fw/{exp_name}")
            os.system(f"rm -rf /data/marl/checkpoints/iclr24-benchmark/fw/{exp_name}")
        else:
            for trial_name in os.listdir(f"/data/marl/checkpoints/iclr24-benchmark/fw/{exp_name}"):
                if trial_name != TRIAL_NAME:
                    print(f"Removing checkpoint dir: "
                          f"/data/marl/checkpoints/iclr24-benchmark/fw/{exp_name}/{trial_name}")
                    os.system(f"rm -rf /data/marl/checkpoints/iclr24-benchmark/fw/{exp_name}/{trial_name}")


if __name__ == "__main__":
    main()