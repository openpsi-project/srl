import pandas as pd
import numpy as np

from typing import List, Tuple

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
from datetime import datetime
import itertools
import re
import numpy as np
import numpy as np
import scipy.stats as st


def get_ci(data, axis):
    return st.t.interval(confidence=0.95,
                         df=len(data[axis]) - 1,
                         loc=np.mean(data, axis=axis),
                         scale=st.sem(data, axis=axis))


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


def main_atari():
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
        scores[algo][j, k, i] = reward
        frames[algo][j, k, i] = f
        versions[algo][j, k, i] = v

    # print(scores, versions, frames)

    scores_median = {}
    scores_std = {}
    for k, score in scores.items():
        scores_median[k] = np.median(score, axis=0)
        scores_std[k] = np.std(score, axis=0)

    for (k, v1), (_, v2) in zip(scores_median.items(), scores_std.items()):
        print(k)
        df = [[] for _ in range(v1.shape[0])]
        for j in range(v1.shape[1]):
            for i in range(v1.shape[0]):
                df[i].append(f"{v1[i, j]:.1f}")
        df = pd.DataFrame(df)
        df.index = ['BattleZone', 'DoubleDunk', 'NameThisGame', 'Phoenix', 'Qbert']
        df.columns = list(map(lambda x: f"scale x{str(int(x))}", np.array(scales) * 2))
        print(df.to_latex())


def main_football():
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

    print(scores, versions, frames)
    scores_median = {}
    scores_std = {}
    for k, score in scores.items():
        scores_median[k] = np.median(score, axis=0)
        scores_std[k] = np.std(score, axis=0)

    for (k, v1), (_, v2) in zip(scores_median.items(), scores_std.items()):
        print(k)
        df = [[] for _ in range(v1.shape[0])]
        for j in range(v1.shape[1]):
            for i in range(v1.shape[0]):
                df[i].append(f"{v1[i, j]:.2f}")
        df = pd.DataFrame(df)
        df.index = ['3v1', 'Corner', 'CAeasy', 'CAhard']
        df.columns = list(map(lambda x: f"scale x{str(int(x))}", np.array(scales) * 2))
        print(df.to_latex())


if __name__ == "__main__":
    # print(extract_reward_version_frames("Corner-ppo-s1-x8"))
    # main_atari()
    main_football()