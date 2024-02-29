from typing import List, Tuple

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
from datetime import datetime
import itertools
import re
import numpy as np
import matplotlib as mpl
import matplotlib.axes as axes
import matplotlib.pyplot as plt

# random + human score
_ATARI_DATA = {
    'BattleZone': (2360.0, 37187.5),
    'DoubleDunk': (-18.6, -16.4),
    'NameThisGame': (2292.3, 8049.0),
    'Phoenix': (761.4, 7242.6),
    'Qbert': (163.9, 13455.0),
}

TRIAL_NAME = "iclr24-benchmark"


def match2float(match: str):
    return float(match.strip().strip(',').strip().split(':')[-1].strip())


def extract_time_from_line(line: str):
    time_pattern = r'^\d+: (\d{8}-\d{2}:\d{2}:\d{2}.\d{3})'
    # Find all matches using the regular expression pattern
    matches = re.findall(time_pattern, line, re.MULTILINE)
    return datetime.strptime(matches[0].split('.')[0], '%Y%m%d-%H:%M:%S')


def extract_reward_version_frames(exp_name: str,
                                  train_time: float = float('inf'),
                                  em_idx: int = 0) -> List[Tuple[float, int, int]]:
    logdir = f"/data/marl/logs/iclr24-benchmark/fw/{exp_name}_{TRIAL_NAME}/em"
    with open(logdir, 'r') as f:
        text = f.read()
    filtered_texts = []
    train_start_time: datetime = None
    for line in text.strip().split('\n'):
        # if exp_name == "DoubleDunk-ppo-s1-x1":
        #     print(line)
        if not line.strip().startswith(f"{em_idx}: "):
            continue
        if "Logging stats" not in line or "fps" in line:
            continue
        if train_start_time is None and 'fps' in line:
            train_start_time = extract_time_from_line(line)
        if train_start_time is not None and (extract_time_from_line(line) -
                                             train_start_time).total_seconds() > train_time:
            break
        filtered_texts.append(line)
    text = '\n'.join(filtered_texts)
    rewards = list(map(match2float, re.findall(r",\s*'episode_return':\s*(-?\d+\.\d+)", text)))
    versions = list(map(match2float, re.findall(r",\s*'version':\s\d+,\s*", text)))
    frames = list(map(match2float, re.findall(r",\s*'frames':\s\d+", text)))
    assert len(rewards) == len(versions) == len(frames)
    return list(zip(rewards, versions, frames))


def get_trial_reward(rewards: List[int]) -> float:
    return np.mean(rewards[-10:])


def main_atari(ax):
    algorithms = ['PPO', 'ApexDQN']
    scales = [0.5, 1, 2, 4, 8]
    seeds = list(range(1, 6))
    games = ['BattleZone', 'DoubleDunk', 'NameThisGame', 'Phoenix', 'Qbert']
    scores = {k: np.zeros((len(seeds), len(games), len(scales)), dtype=np.float64) for k in algorithms}
    frames = {k: np.zeros((len(seeds), len(games), len(scales)), dtype=np.float64) for k in algorithms}
    versions = {k: np.zeros((len(seeds), len(games), len(scales)), dtype=np.float64) for k in algorithms}
    for (i, scale), (j, seed), algo, (k, game) in itertools.product(enumerate(scales), enumerate(seeds),
                                                                    algorithms, enumerate(games)):
        exp_algo_identifier = 'apex' if algo == 'ApexDQN' else algo.lower()
        exp_name = f"{game}-{exp_algo_identifier}-s{seed}-x{scale}"
        rewards_versions_frames = extract_reward_version_frames(exp_name)
        reward = get_trial_reward([x[0] for x in rewards_versions_frames])
        v, f = rewards_versions_frames[-1][1:]
        scores[algo][j, k,
                     i] = (reward - _ATARI_DATA[game][0]) / (_ATARI_DATA[game][1] - _ATARI_DATA[game][0])
        frames[algo][j, k, i] = f
        versions[algo][j, k, i] = v

    iqm = lambda scores: np.array(
        [metrics.aggregate_iqm(scores[..., scale]) for scale in range(scores.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(scores, iqm, reps=50000)
    # for k, v in iqm_scores.items():
    #     iqm_scores[k] = v - v[..., 0:1]
    #     iqm_cis[k] -= v[..., 0:1]
    plot_utils.plot_sample_efficiency_curve(
        np.array(scales) * 2,
        iqm_scores,
        iqm_cis,
        algorithms=algorithms,
        labelsize=20,
        ylabel="IQM Human\nNormalized Score",
        xlabel="Scale of Batch Size & Workers",
        ax=ax,
    )
    ax.set_xticks(np.arange(0, 17, 2))
    ax.legend(fontsize=20)
    print(iqm_scores, iqm_cis)
    return ax


def main_football(ax):
    algorithms = ['MAPPO', 'VDN']
    scales = [0.5, 1, 2, 4, 8]
    seeds = list(range(1, 5))
    games = ['3v1', 'Corner', 'CAeasy', 'CAhard']
    scores = {k: np.zeros((len(seeds), len(games), len(scales)), dtype=np.float64) for k in algorithms}
    frames = {k: np.zeros((len(seeds), len(games), len(scales)), dtype=np.float64) for k in algorithms}
    versions = {k: np.zeros((len(seeds), len(games), len(scales)), dtype=np.float64) for k in algorithms}
    for (i, scale), (j, seed), algo, (k, game) in itertools.product(enumerate(scales), enumerate(seeds),
                                                                    algorithms, enumerate(games)):
        exp_algo_identifier = algo.lower() if algo != 'MAPPO' else 'ppo'
        exp_name = f"{game}-{exp_algo_identifier}-s{seed}-x{scale}"
        rewards_versions_frames = extract_reward_version_frames(exp_name)
        reward = get_trial_reward([x[0] for x in rewards_versions_frames])
        v, f = rewards_versions_frames[-1][1:]
        scores[algo][j, k, i] = reward
        frames[algo][j, k, i] = f
        versions[algo][j, k, i] = v

    iqm = lambda scores: np.array(
        [metrics.aggregate_iqm(scores[..., scale]) for scale in range(scores.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(scores, iqm, reps=50000)
    # for k, v in iqm_scores.items():
    #     iqm_scores[k] = v - v[..., 0:1]
    #     iqm_cis[k] -= v[..., 0:1]
    plot_utils.plot_sample_efficiency_curve(
        np.array(scales) * 2,
        iqm_scores,
        iqm_cis,
        algorithms=algorithms,
        labelsize=20,
        xlabel="Scale of Batch Size & Workers",
        ylabel="IQM Win Rate",
        ax=ax,
    )
    ax.set_xticks(np.arange(0, 17, 2))
    ax.legend(fontsize=20)
    print(iqm_scores, iqm_cis)
    return ax


if __name__ == "__main__":
    # print(extract_reward_version_frames("Corner-ppo-s1-x8"))
    fig = plt.figure(figsize=(10, 3.5))
    ax = plt.subplot(1, 2, 1)
    main_atari(ax)
    ax = plt.subplot(1, 2, 2)
    main_football(ax)
    plt.tight_layout()
    fig.savefig("reward_vs_scale.pdf", bbox_inches='tight', pad_inches=0.1)