import logging
import numpy as np
import os
import re
import socket
import time
import torch

import algorithm
import base.namedarray
from codespace import config as config_package
import environment

logger = logging.getLogger('viewer')


class ConfigError(RuntimeError):
    pass


def match_policies(agent_count, agent_specs, policy_names):
    rs1 = [None for _ in range(agent_count)]
    policy_matches = {k: [] for k in policy_names}
    for i in range(len(agent_specs)):
        for j in range(agent_count):
            if rs1[j] is None and re.fullmatch(agent_specs[i].index_regex, str(j)):
                rs1[j] = policy_names[i]
                policy_matches[policy_names[i]].append(j)

    for j in range(agent_count):
        if rs1[j] is None:
            raise ConfigError(f'Agent {j} has no matched agent_spec')
    for k, agent_indices in policy_matches.items():
        if not agent_indices:
            raise ConfigError(f'Policy {k} has no matched agents')
    return policy_matches


def main(args):
    experiment = config_package.make_experiment(args.experiment_name)
    setup = experiment.initial_setup()

    logger.info("Initializing policies...")
    policies = {}

    for policy_worker_cfg in setup.policies:
        if policy_worker_cfg.policy_name not in policies:
            policies[policy_worker_cfg.policy_name] = algorithm.policy.make(policy_worker_cfg.policy)

    if not socket.gethostname().startswith("frl") and not socket.gethostname().startswith('DESKTOP-PR1HLOG'):
        if not args.ckpt_versions:
            raise ConfigError("Please specify the directories of checkpoints.")
        elif not isinstance(args.ckpt_versions, list) or len(args.ckpt_versions) != len(policies):
            raise ConfigError(
                "The length of specified checkpoint names does not equal to the number of policies.")
        ckpt_dirs = args.ckpt_versions
    else:
        if args.ckpt_versions is None:
            ckpt_versions = ['latest' for _ in range(len(policies))]
        else:
            if not isinstance(args.ckpt_versions, list) or len(args.ckpt_versions) != len(policies):
                raise ConfigError(
                    "The length of specified checkpoint names does not equal to the number of policies.")
            ckpt_versions = ['latest' if name is None else name for name in args.ckpt_versions]
        ckpt_dirs = [
            os.path.join("/data/marl/checkpoints", args.experiment_name, args.trial_name, policy_name,
                         version) for policy_name, version in zip(policies, ckpt_versions)
        ]

    logger.info("Loading state dict...")
    for (policy_name, policy), ckpt_dir in zip(policies.items(), ckpt_dirs):
        state_dict = torch.load(ckpt_dir, map_location=policy.device)['state_dict']
        policy.set_state_dict(state_dict)
    logger.info("Successfully loaded state dict!")

    # for simplicity we just initialize the environment of the first actor worker
    env = environment.env_base.make(setup.actors[0].env)
    agent_specs = setup.actors[0].agent_specs
    logger.info("Successfully initialized environment!")

    policy_agents = match_policies(env.agent_count, agent_specs, list(policies.keys()))
    episode_infos = []
    for ep in range(args.episodes):
        logger.info(f"Start Episode: {ep + 1}/{args.episodes}")
        rs = env.reset()
        ep_step = 0
        policy_states = {k: None for k in policies}
        dones = [False for _ in range(env.agent_count)]

        while not all(dones):
            actions = [None for _ in range(env.agent_count)]
            for policy_name, policy in policies.items():
                agent_indices = policy_agents[policy_name]

                # batching
                rsubset = [rs[i] for i in agent_indices]
                bs = len(rsubset)

                # make request
                obs = base.namedarray.recursive_aggregate([x.obs for x in rsubset], np.stack)
                policy_state = policy_states[policy_name]
                is_evaluation = np.ones((bs, 1), dtype=np.uint8)
                on_reset = (ep_step == 0) * np.ones((bs, 1), dtype=np.uint8)
                # the following will not be used except for specifying the batch size of a request
                client_id = request_id = received_time = np.random.rand(bs, 1)
                request = algorithm.policy.RolloutRequest(obs, policy_state, is_evaluation, on_reset,
                                                          client_id, request_id, received_time)

                rollout_result = policy.rollout(request)

                # update policy state
                policy_states[policy_name] = rollout_result.policy_state
                # update rollout result
                for result_idx, agent_idx in zip(range(len(agent_indices)), agent_indices):
                    actions[agent_idx] = rollout_result[agent_idx].action
            assert all([action is not None for action in actions]), actions
            rs = env.step(actions)

            env.render()
            time.sleep(0.02)

            ep_step += 1
            dones = [r.done.all() for r in rs]

        episode_infos.append(rs[0].info)

    avg_info = base.namedarray.recursive_apply(base.namedarray.recursive_aggregate(episode_infos, np.stack),
                                               np.mean)

    logger.info('-' * 20)
    for k, v in avg_info.items():
        key = ' '.join(k.split('_')).title()
        logger.info("{}: \t{:.2f}".format(key, v.item()))
    logger.info('-' * 20)