from typing import List
import subprocess
import time
import os
from queue import Queue
import random

TRIAL_NAME = "iclr24-benchmark"


def _parse_nodelist(nodelist: str) -> List[str]:
    return subprocess.check_output([
        'scontrol',
        'show',
        'hostnames',
        nodelist,
    ]).decode('utf-8').strip().split('\n')


def get_all_possible_exp_names():
    seeds = list(range(1, 6))
    scales = [0.5, 1, 2, 4, 8]
    atari5_games = ['BattleZone', 'DoubleDunk', 'Phoenix', 'NameThisGame', 'Qbert']
    atari_dqn_exp_names = [
        f"{game}-apex-s{seed}-x{scale}" for game in atari5_games for scale in scales for seed in seeds
    ]
    atari_ppo_exp_names = [
        f"{game}-ppo-s{seed}-x{scale}" for game in atari5_games for scale in scales for seed in seeds
    ]
    atari_muzero_exp_names = [
        f"{game}-muzero-s{seed}-x{scale}" for game in atari5_games for scale in scales for seed in seeds
    ]
    grf_games = ['3v1', 'Corner', 'CAeasy', 'CAhard']
    grf_ppo_exp_names = [
        f"{game}-ppo-s{seed}-x{scale}" for game in grf_games for scale in scales for seed in seeds
    ]
    grf_qmix_exp_names = [
        f"{game}-vdn-s{seed}-x{scale}" for game in grf_games for scale in scales for seed in seeds
    ]
    return atari_dqn_exp_names + atari_ppo_exp_names + atari_muzero_exp_names + grf_ppo_exp_names + grf_qmix_exp_names


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


ALL_EXP_NAMES = get_all_possible_exp_names()


def main(seeds):
    scales = [0.5, 1, 2, 4, 8]

    atari5_games = ['BattleZone', 'DoubleDunk', 'Phoenix', 'NameThisGame', 'Qbert']

    atari_dqn_exp_names = [
        f"{game}-apex-s{seed}-x{scale}" for game in atari5_games for scale in scales for seed in seeds
    ]
    atari_ppo_exp_names = [
        f"{game}-ppo-s{seed}-x{scale}" for game in atari5_games for scale in scales for seed in seeds
    ]
    atari_muzero_exp_names = [
        f"{game}-muzero-s{seed}-x{scale}" for game in atari5_games for scale in [8] for seed in seeds
    ]
    # atari_muzero_exp_names = []

    grf_games = ['3v1', 'Corner', 'CAeasy', 'CAhard']
    grf_ppo_exp_names = [
        f"{game}-ppo-s{seed}-x{scale}" for game in grf_games for scale in scales for seed in seeds
    ]
    grf_qmix_exp_names = [
        f"{game}-vdn-s{seed}-x{scale}" for game in grf_games for scale in scales for seed in seeds
    ]

    cur_exp_names = atari_dqn_exp_names + atari_ppo_exp_names + atari_muzero_exp_names + grf_ppo_exp_names + grf_qmix_exp_names

    def is_concerned_job_name(job_name):
        if not (':' in job_name and "_" in job_name):
            return False
        exp_name, t = job_name.split(":")[0].split('_')
        return (exp_name in ALL_EXP_NAMES) and t == TRIAL_NAME

    def extract_exp_name(job_name):
        return job_name.split(":")[0].split('_')[0]

    # Function to submit a SLURM job
    def submit_slurm_job(experiment_name, trial_name):
        print(f"submit {experiment_name}")
        # if 'muzero' not in experiment_name:
        cmd = f"nohup python3 -m apps.main start -e {experiment_name} -f {trial_name} > z_iclr24logs/{experiment_name}.out &"
        # else:
        #     sinfo_lines = subprocess.check_output("sinfo -h", shell=True, text=True).strip().split('\n')
        #     for line in sinfo_lines:
        #         *_, status, nodelist = line.strip().split()
        #         if status == 'mix':
        #             mix1gnodes = list(filter(lambda x: "1g" in x, _parse_nodelist(nodelist)))
        #             break
        #     ps_node = random.choice(mix1gnodes)
        #     print("MuZero ps_node:", ps_node)
        #     cmd = f"nohup python3 -m apps.main start -e {experiment_name} -f {trial_name} --ps_node {ps_node} > z_iclr24logs/{experiment_name}.out &"
        subprocess.run(cmd, shell=True)

    # Function to check the number of running jobs
    def get_running_jobs():
        squeue_result = subprocess.check_output(
            r"squeue -u fw -o '%.18i %.9P %.80j %.8u %.2t %.10M %.6D %R' -h", shell=True, text=True)
        job_names = [x.split()[2] for x in squeue_result.strip().split('\n') if len(x) > 0]
        job_names = set(map(extract_exp_name, filter(is_concerned_job_name, job_names)))
        return list(job_names)

    max_queue_size = 10
    job_queue = Queue(100)
    for job in get_running_jobs():
        job_queue.put(job)
    print("current # jobs: ", job_queue.qsize())

    while True:
        for _ in range(job_queue.qsize()):
            # Check if any jobs have finished and remove them from the queue
            experiment_name = job_queue.get()
            # print(experiment_name)
            current_jobs = get_running_jobs()
            if not any([job.startswith(experiment_name) for job in current_jobs]):
                print(f"{experiment_name} has finished.")
            else:
                # If the job is still running, put it back in the queue
                job_queue.put(experiment_name)

        current_jobs = get_running_jobs()
        assert len(current_jobs) == job_queue.qsize()
        current_job_count = len(current_jobs)
        if current_job_count < max_queue_size:
            # You can customize your experiment and trial names here
            runned_experiment_names = get_runned_exp_names()
            assert len(set(runned_experiment_names)) == len(runned_experiment_names)
            remaining_exps = list(set(cur_exp_names) - set(runned_experiment_names))
            if len(remaining_exps) == 0:
                print("No more experiments to run")
                return
            exp_name = random.choice(remaining_exps)
            submit_slurm_job(exp_name, TRIAL_NAME)
            job_queue.put(exp_name)

        time.sleep(20)


if __name__ == "__main__":
    existing_exp_names = get_runned_exp_names()
    print(f"Planning to run {len(ALL_EXP_NAMES)} experiments, "
          f"{len(existing_exp_names)}/{len(ALL_EXP_NAMES)} "
          f"({len(existing_exp_names)/len(ALL_EXP_NAMES):.2%}) have been launched.")
    for seeds in [[1], [2], [3], [4], [5]]:
        print(f">>>>>>> running seed {seeds}")
        main(seeds)