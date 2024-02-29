import copy
import random

from api.config import *

_HNS_RANDOMWALL_CONFIG = {
    'max_n_agents': 6,
    # grab
    'grab_box': True,
    'grab_out_of_vision': False,
    'grab_selective': False,
    'grab_exclusive': False,
    # lock
    'lock_box': True,
    'lock_ramp': True,
    'lock_type': 'all_lock_team_specific',
    'lock_out_of_vision': False,
    # horizon
    'n_substeps': 15,
    'horizon': 240,
    'prep_fraction': 0.4,
    # map
    'scenario': 'randomwalls',
    'n_rooms': 4,
    'rew_type': 'joint_zero_sum',
    'random_room_number': True,
    'prob_outside_walls': 0.5,
    'restrict_rect': [-6.0, -6.0, 12.0, 12.0],
    'hiders_together_radius': 0.5,
    'seekers_together_radius': 0.5,
    # box
    'n_boxes': [3, 9],
    'n_elongated_boxes': [3, 9],
    'box_only_z_rot': True,
    'boxid_obs': False,
    # ramp
    'n_ramps': 2,
    # food
    'n_food': 0,
    'max_food_health': 40,
    'food_radius': 0.5,
    'food_box_centered': True,
    'food_together_radius': 0.25,
    'food_respawn_time': 5,
    'food_rew_type': 'joint_mean',
    # observations
    'n_lidar_per_agent': 30,
    'visualize_lidar': False,
    'prep_obs': True
}

RANDOMWALL_OBSERVATION_SPACE = {
    'agent_qpos_qvel': (5, 10),
    'box_obs': (9, 15),
    'lidar': (1, 30),
    'mask_aa_obs': (5,),
    'mask_aa_obs_spoof': (5,),
    'mask_ab_obs': (9,),
    'mask_ab_obs_spoof': (9,),
    'mask_ar_obs': (2,),
    'mask_ar_obs_spoof': (2,),
    'observation_self': (10,),
    'ramp_obs': (2, 15)
}

_HNS_QUADRANT_CONFIG = {
    # Agents
    'n_hiders': 2,
    'n_seekers': 2,
    # grab
    'grab_box': True,
    'grab_out_of_vision': False,
    'grab_selective': False,
    'grab_exclusive': False,
    # lock
    'lock_box': True,
    'lock_ramp': False,
    'lock_type': "all_lock_team_specific",
    'lock_out_of_vision': False,
    # Scenario
    'n_substeps': 15,
    'horizon': 80,
    'scenario': 'quadrant',
    'prep_fraction': 0.4,
    'rew_type': "joint_zero_sum",
    'restrict_rect': [0.1, 0.1, 5.9, 5.9],
    'p_door_dropout': 0.5,
    'quadrant_game_hider_uniform_placement': True,
    # Objects
    'n_boxes': 2,
    'box_only_z_rot': True,
    'boxid_obs': False,
    'n_ramps': 1,
    'penalize_objects_out': True,
    # Food
    'n_food': 0,
    # Observations
    'n_lidar_per_agent': 30,
    'visualize_lidar': False,
    'prep_obs': True,
}

QUADRANT_OBSERVATION_SPACE = {
    'agent_qpos_qvel': (3, 10),
    'box_obs': (2, 12),
    'lidar': (1, 30),
    'mask_aa_obs': (3,),
    'mask_ab_obs': (2,),
    'mask_ar_obs': (1,),
    'observation_self': (10,),
    'ramp_obs': (1, 9)
}

ACTION_SPACE = {'move_x': 11, 'move_y': 11, 'move_z': 11, 'lock': 2, 'grab': 2}


class HideAndSeekMiniExperiment(Experiment):

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(
                count=2,
                scheduling=Scheduling.actor_worker_default(container_image="fw/marl-cpu-hns-new"),
            ),
            policies=TasksGroup(
                count=1,
                scheduling=Scheduling.policy_worker_default(gpu=0.12, container_image="fw/marl-gpu-new"),
            ),
            trainers=TasksGroup(
                count=1,
                scheduling=Scheduling.trainer_worker_default(container_image="fw/marl-gpu-new"),
            ),
        )

    def initial_setup(self):
        policy_name = "default"
        inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)

        policy = Policy(type_="hns",
                        args=dict(
                            obs_space=RANDOMWALL_OBSERVATION_SPACE,
                            act_space=ACTION_SPACE,
                            chunk_len=10,
                            seed=1,
                        ))
        env_config = copy.deepcopy(_HNS_RANDOMWALL_CONFIG)
        env_config['scenario_name'] = 'hide_and_seek'
        env_config['max_n_agents'] = 4
        env_config['n_hiders'] = 2
        env_config['n_seekers'] = 2
        # TODO: force terimination in actor worker instead of in hns env
        env_config['horizon'] = 239  # such that reset + step = 240
        return ExperimentConfig(
            actors=[
                ActorWorker(
                    env=Environment(type_="hide_and_seek", args=env_config),
                    inference_streams=[inference_stream],
                    sample_streams=[sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=False,
                            send_after_done=False,
                            sample_steps=240,
                        )
                    ],
                    ring_size=32,
                    max_num_steps=2000,
                ) for _ in range(2)
            ],
            policies=[
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=inference_stream,
                    parameter_db=parameter_db,
                    max_inference_delay=0.05,
                    pull_frequency_seconds=0.5,
                    pull_max_failures=100,
                    policy=policy,
                ) for _ in range(1)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(
                        max_size=16,
                        reuses=4,
                        batch_size=2,
                    ),
                    policy_name=policy_name,
                    trainer=Trainer(type_="mappo"),
                    policy=policy,
                    sample_stream=sample_stream,
                    parameter_db=parameter_db,
                ) for _ in range(1)
            ],
        )


class HideAndSeekQuadrantExperiment(Experiment):

    def __init__(self):
        self.num_actors = 768
        self.num_policies = 16
        self.num_trainers = 8

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(
                count=self.num_actors,
                scheduling=Scheduling.actor_worker_default(partition="cpu",
                                                           container_image="fw/marl-cpu-hns-new"),
            ),
            policies=TasksGroup(
                count=self.num_policies,
                scheduling=Scheduling.policy_worker_default(gpu=0.12, container_image="fw/marl-gpu-new"),
            ),
            trainers=TasksGroup(
                count=self.num_trainers,
                scheduling=Scheduling.trainer_worker_default(container_image="fw/marl-gpu-new"),
            ),
        )

    def initial_setup(self):
        policy_name = "default"

        seed = 1
        ring_size = 64
        inference_splits = 2
        policy = Policy(type_="hns",
                        args=dict(
                            obs_space=QUADRANT_OBSERVATION_SPACE,
                            act_space=ACTION_SPACE,
                            chunk_len=10,
                            seed=seed,
                        ))
        env_config = copy.deepcopy(_HNS_QUADRANT_CONFIG)
        env_config['scenario_name'] = 'hide_and_seek'
        # TODO: force terimination in actor worker instead of in hns env
        env_config['horizon'] = 79  # such that reset + step = 80
        trainer = Trainer(type_="mappo",
                          args=dict(
                              discount_rate=0.998,
                              gae_lambda=0.95,
                              eps_clip=0.2,
                              value_loss='mse',
                              value_loss_weight=0.5,
                              entropy_bonus_weight=0.01,
                              optimizer='adam',
                              optimizer_config=dict(lr=3e-4, weight_decay=1e-6),
                              popart=True,
                              max_grad_norm=5.0,
                          ))
        return ExperimentConfig(
            actors=[
                ActorWorker(
                    env=Environment(type_="hide_and_seek", args=env_config),
                    inference_streams=[policy_name],
                    sample_streams=[policy_name],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=False,
                            send_after_done=False,
                            sample_steps=80,
                        )
                    ],
                    max_num_steps=2000,
                    ring_size=ring_size,
                    inference_splits=inference_splits,
                ) for k in range(self.num_actors)
            ],
            policies=[
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=policy_name,
                    max_inference_delay=0.05,
                    pull_frequency_seconds=0.5,
                    pull_max_failures=100000,
                    policy=policy,
                ) for j in range(self.num_policies)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(
                        max_size=8,
                        reuses=4,
                        batch_size=64000 // (_HNS_QUADRANT_CONFIG['horizon'] // 10) // self.num_trainers,
                    ),
                    policy_name=policy_name,
                    trainer=trainer,
                    policy=policy,
                    log_frequency_seconds=1,
                    push_frequency_seconds=0.5,
                    sample_stream=policy_name,
                ) for _ in range(self.num_trainers)
            ],
        )


class HideAndSeekRandomWallExperiment(Experiment):

    def __init__(self):
        self.num_actors = 768
        self.num_policies = 16
        self.num_trainers = 8

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(
                count=self.num_actors,
                scheduling=Scheduling.actor_worker_default(container_image="fw/marl-cpu-hns-new"),
            ),
            policies=TasksGroup(
                count=self.num_policies,
                scheduling=Scheduling.policy_worker_default(gpu=0.12, container_image="fw/marl-gpu-new"),
            ),
            trainers=TasksGroup(
                count=self.num_trainers,
                scheduling=Scheduling.trainer_worker_default(container_image="fw/marl-gpu-new"),
            ),
        )

    def initial_setup(self):
        policy_name = "default"

        seed = 1
        ring_size = 96
        inference_splits = 2
        policy = Policy(type_="hns",
                        args=dict(
                            obs_space=RANDOMWALL_OBSERVATION_SPACE,
                            act_space=ACTION_SPACE,
                            chunk_len=10,
                            seed=seed,
                        ))
        env_configs = []
        for _ in range(self.num_actors):
            env_config = copy.deepcopy(_HNS_RANDOMWALL_CONFIG)
            env_config['scenario_name'] = 'hide_and_seek'
            env_config['max_n_agents'] = 6
            env_config['n_hiders'] = random.randint(1, 3)
            env_config['n_seekers'] = random.randint(1, 3)
            # TODO: force terimination in actor worker instead of in hns env
            env_config['horizon'] = 239  # such that reset + step = 240
            env_configs.append(env_config)
        trainer = Trainer(type_="mappo",
                          args=dict(
                              discount_rate=0.998,
                              gae_lambda=0.95,
                              eps_clip=0.2,
                              value_loss='mse',
                              value_loss_weight=0.5,
                              entropy_bonus_weight=0.01,
                              optimizer='adam',
                              optimizer_config=dict(lr=3e-4, weight_decay=1e-6),
                              popart=True,
                              max_grad_norm=5.0,
                          ))
        return ExperimentConfig(
            actors=[
                ActorWorker(
                    env=Environment(type_="hide_and_seek", args=env_config),
                    inference_streams=[policy_name],
                    sample_streams=[policy_name],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=False,
                            send_after_done=False,
                            sample_steps=240,
                        )
                    ],
                    max_num_steps=2000,
                    ring_size=ring_size,
                    inference_splits=inference_splits,
                ) for env_config in env_configs
            ],
            policies=[
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=policy_name,
                    max_inference_delay=0.05,
                    pull_frequency_seconds=0.5,
                    pull_max_failures=100,
                    policy=policy,
                ) for j in range(self.num_policies)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(
                        max_size=8,
                        reuses=4,
                        batch_size=72000 // (_HNS_QUADRANT_CONFIG['horizon'] // 10) // self.num_trainers,
                    ),
                    policy_name=policy_name,
                    trainer=trainer,
                    policy=policy,
                    log_frequency_seconds=1,
                    push_frequency_seconds=0.5,
                    sample_stream=policy_name,
                ) for _ in range(self.num_trainers)
            ],
        )


register_experiment("hns-mini", HideAndSeekMiniExperiment)
register_experiment("hns-quadrant", HideAndSeekQuadrantExperiment)
register_experiment("hns-randomwall", HideAndSeekRandomWallExperiment)
