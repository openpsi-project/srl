import copy

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

ACTOR_IMAGE = "marl/marl-cpu-hns"
GPU_IMAGE = "marl/marl-gpu"


class HideAndSeekBenchmarkExperiment(Experiment):

    def __init__(self):
        self.aws, self.pws, self.tws = 32, 0, 1

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(count=self.aws,
                              scheduling=Scheduling.actor_worker_default(cpu=2,
                                                                         partition="cpu",
                                                                         container_image=ACTOR_IMAGE)),
            trainers=TasksGroup(
                count=self.tws,
                scheduling=Scheduling.trainer_worker_default(cpu=8, container_image=GPU_IMAGE, mem=10 * 1024),
            ))

    def initial_setup(self):
        policy_name = "default"
        policy = Policy(type_="hns",
                        args=dict(
                            obs_space=RANDOMWALL_OBSERVATION_SPACE,
                            act_space=ACTION_SPACE,
                            chunk_len=10,
                            seed=1,
                        ))
        inference_stream = InferenceStream(type_=InferenceStream.Type.INLINE,
                                           policy=policy,
                                           policy_name=policy_name,
                                           stream_name="")
        # sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)

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
                    ring_size=1,
                    max_num_steps=2000,
                ) for _ in range(self.aws)
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
                ) for _ in range(self.pws)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(
                        max_size=128,
                        reuses=1,
                        batch_size=100,
                    ),
                    policy_name=policy_name,
                    trainer="mappo",
                    policy=policy,
                    sample_stream=policy_name,
                    parameter_db=parameter_db,
                ) for _ in range(self.tws)
            ],
        )


register_experiment("hns-benchmark", HideAndSeekBenchmarkExperiment)
