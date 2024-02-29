"""Simple demo program for an environment.
"""
import argparse
import copy
import pickle
import random
import os
import socket
import time
import warnings

import numpy as np

from base.namedarray import recursive_aggregate
import api.config
import api.policy
import api.environment as env_base
import legacy.experiments.hns  # TODO: this violates the convention.

os.environ["MKL_NUM_THREADS"] = '1'


def run_game(env):
    obs = []
    rs = env.reset()
    obs.append(rs)
    print("Starting game...")
    reward = 0
    current_round = 0
    rounds = 10000

    while current_round < rounds:
        if hasattr(rs[0].obs, "available_action"):
            actions = [env.action_spaces[i].sample(rs[i].obs.available_action) for i in range(len(rs))]
        else:
            actions = [env.action_spaces[i].sample() for i in range(len(rs))]
        rs = env.step(actions)
        obs.append(rs)
        reward += rs[0].reward
        if rs[0].done:
            obs_len = len(obs)
            obs = []
            # break
            # env.reset()
            obs.append(env.reset())
            current_round += 1
            print(f"Round {current_round}: Episode ended, reward = {rs[0].reward}. Episode len = {obs_len}")
        try:
            env.render()
        except NotImplementedError:
            warnings.warn("Rendering not implemented for Environment")
        time.sleep(0.05)
    print(f"Game ended. Reward = {reward}")


def run_fps_test(env, dumps=False, batch_size=0):
    print("Starting Inference test...")
    steps = episodes = requests = pickle_time = batch_time = 0
    rs = env.reset()
    start_time = time.monotonic_ns()
    ps = np.random.random((1, 64)).astype(np.float32)
    for _ in range(128):
        episodes += 1
        while True:
            if hasattr(rs[0].obs, "available_action"):
                actions = [env.action_spaces[i].sample(rs[i].obs.available_action) for i in range(len(rs))]
            else:
                actions = [env.action_spaces[i].sample() for i in range(len(rs))]
            rs = env.step(actions)

            if dumps:
                pickle_start = time.monotonic_ns()
                [
                    pickle.dumps(
                        api.policy.RolloutRequest(
                            obs=rs[i].obs,
                            policy_state=ps,
                            is_evaluation=np.array([True], dtype=np.uint8),
                            # Cause we use recursive aggregate, we need an identifier for default policy state.
                            on_reset=np.array([True], dtype=np.uint8),
                        )) for i in range(len(rs))
                ]
                pickle_time += (time.monotonic_ns() - pickle_start) / 1e6

            steps += 1
            requests += len(rs)
            if all(r.done for r in rs):
                rs = env.reset()
                break
    current = time.monotonic_ns()
    duration = (current - start_time) / 1e9

    if dumps:
        print(f" Pickling/step={pickle_time/steps:.2f}ms")
    print(
        f" Games={episodes}, Steps/Second={steps / duration:.1f}, "
        f"game_dur={duration / episodes:.3f}s, Req/Second={requests/duration:.1f}, Req/Step:{requests/steps:.1f}",
        flush=True)


def run_batching_test(env, batch_size):
    batch_time = 0
    rs = env.reset()
    ps = np.random.random((1, 64)).astype(np.float32)
    if batch_size > 0:
        for i in range(100):
            batch_start = time.monotonic_ns()
            recursive_aggregate(
                [
                    api.policy.RolloutRequest(
                        obs=rs[0].obs,
                        policy_state=ps,
                        is_evaluation=np.array([True], dtype=np.uint8),
                        # Cause we use recursive aggregate, we need an identifier for default policy state.
                        on_reset=np.array([True], dtype=np.uint8),
                    ) for _ in range(batch_size)
                ],
                np.stack)
            batch_time += (time.monotonic_ns() - batch_start) / 1e6
    if batch_size:
        print(f" BatchSize {batch_size}"
              f" Aggregation/batch={batch_time/100:.2f}ms")


def parse_kwargs(argument):
    """Parser console keyword argument
    Args:
        argument: -A argument of console input. delimited by ";" then ":"
    Returns:
        kwargs: dict
    """
    return {f[0]: f[1] for x in argument.split(";") for f in x.split(":")}


def main():
    parser = argparse.ArgumentParser(prog="env-demo")
    parser.add_argument("env_name", nargs="?", default="smac")
    parser.add_argument("-a", "--args", nargs="*", default=None)
    parser.add_argument("-A", "--kwargs", default=None, help="Outer_delimiter = ; , Inner_delimiter = :")
    parser.add_argument("--fps_test", action="store_true")
    args = parser.parse_args()
    if args.args is None:
        args.args = []
    if args.kwargs is not None:
        args.kwargs = parse_kwargs(args.kwargs)
    else:
        args.kwargs = {}

    if args.env_name == "atari" and not args.kwargs:
        args.kwargs["game_name"] = "Boxing-v0"
    elif args.env_name == "gym_mujoco" and not args.kwargs:
        args.kwargs["scenario"] = "Humanoid-v3"
    elif args.env_name == "football" and not args.kwargs:
        args.kwargs["env_name"] = "academy_3_vs_1_with_keeper"
        args.kwargs["representation"] = "simple115v2"
        args.kwargs['number_of_left_players_agent_controls'] = 3
        args.kwargs['number_of_right_players_agent_controls'] = 0
    elif args.env_name == "smac" and not args.kwargs:
        args.kwargs["map_name"] = "5m_vs_6m"
    elif args.env_name == 'hide_and_seek' and not args.kwargs:
        args.kwargs = copy.deepcopy(legacy.experiments.hns._HNS_RANDOMWALL_CONFIG)
        args.kwargs['scenario_name'] = 'hide_and_seek'
        args.kwargs['max_n_agents'] = 6
        args.kwargs['n_hiders'] = random.randint(1, 3)
        args.kwargs['n_seekers'] = random.randint(1, 3)
        args.kwargs['horizon'] = 240
    elif args.env_name == "nel12" and not args.kwargs:
        args.kwargs["lc_addr"] = "10.210.5.151"
        args.kwargs["lc_port"] = 4444
        args.kwargs["obs_port"] = 12151
        args.kwargs["action_port"] = 13151
        args.kwargs["init_port"] = 14151
        args.kwargs["rlsdk_addr"] = socket.gethostbyname(socket.gethostname())
    else:
        raise NotImplementedError("env_name not valid")

    env_config = api.config.Environment(
        type_=args.env_name,
        args=args.kwargs,
    )
    print(f"Initializing game {args.env_name} with args: {args.args} kwargs: {args.kwargs}...")
    env = env_base.make(env_config)
    print("Action space:", len(env.action_spaces), env.action_spaces)

    if args.fps_test:
        run_fps_test(env)
        run_fps_test(env, dumps=True)
        run_batching_test(env, batch_size=32)
        run_batching_test(env, batch_size=64)
        run_batching_test(env, batch_size=128)
        run_batching_test(env, batch_size=256)
        run_batching_test(env, batch_size=512)
    else:
        run_game(env)


if __name__ == '__main__':
    main()
