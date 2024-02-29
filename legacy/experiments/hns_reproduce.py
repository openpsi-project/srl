import copy
import random
import functools
import itertools
import numpy as np

from api.config import *

_HNS_RANDOMWALL_CONFIG = {
    'max_n_agents': 5,
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


class HideAndSeekRandomWallExperiment(Experiment):
    # 8=64k, 4=32k, 16=128k

    def __init__(
        self,
        inline,
        scale_factor,
        seed=1,
        use_ps=False,
        cpu_scale=1.0,
        large_map=False,
        map_scale=2.0,
        fixed_horizon=True,
        half_tw=False,
    ):
        self.seed = seed
        self.scale_factor = scale_factor
        self.cpu_scale = cpu_scale
        self.num_actors = int((240 if inline else 120) * self.scale_factor * cpu_scale)
        self.num_policies = (0 if inline else 8) * self.scale_factor
        self.num_buffers = (0 if not inline else 16) * self.scale_factor
        self.num_trainers = 1 * self.scale_factor
        self.half_tw = half_tw
        if half_tw:
            self.num_trainers //= 2
        self.inline = inline
        self.use_ps = use_ps
        self.large_map = large_map
        self.map_scale = map_scale
        self.fixed_horizon = fixed_horizon
        if not inline and cpu_scale != 1.0:
            raise RuntimeError()

    def scheduling_setup(self) -> ExperimentScheduling:
        scheduling = ExperimentScheduling(
            actors=TasksGroup(
                count=self.num_actors,
                scheduling=Scheduling.actor_worker_default(
                    cpu=1,
                    partition='cpu',
                    exclude='frl1g064',
                    # node_list='frl1g[038-064,066-102]',
                    mem=2 * 1024,
                    # exclude='frl1g106',  # ps_node
                    # node_list="frl1g[066-102]", # fullsize
                    container_image="fw/marl-cpu-hns-new",
                ),
            ),
            trainers=TasksGroup(
                count=self.num_trainers,
                scheduling=Scheduling.trainer_worker_default(
                    cpu=8,
                    container_image="fw/marl-gpu-new",
                    mem=60 * 1024,
                    node_list='frl8a[138-141]',
                    # node_list="frl8a139",
                    partition='dev',
                    gpu_type='tesla'),
            ),
        )
        if self.use_ps:
            scheduling.parameter_server_worker = TasksGroup(
                count=1,
                scheduling=Scheduling.parameter_server_worker_default(mem=100 * 1024),
            )
        if not self.inline:
            scheduling.policies = TasksGroup(
                count=self.num_policies + 8,
                scheduling=Scheduling.policy_worker_default(
                    cpu=3,
                    gpu=0.25,
                    mem=20 * 1024,
                    # partition='cpu',
                    # node_list='frl1g[066-087],frl2g[007-023,026-029]', # full size
                    # node_list='frl2g[010-017]', # 8 cards
                    node_list="frl1g[066-097],frl2g[014-029]",
                    exclude='frl1g064',
                    container_image="fw/marl-gpu-new",
                    gpu_type='geforce',
                ),
            )
        return scheduling

    @property
    def wandb_group(self):
        name = (f"bs{int(self.scale_factor * 8)}k"
                if not self.inline else f"bs{int(self.scale_factor * 8)}k-inline-aw{self.num_actors}")
        if self.large_map:
            name += f"-lmx{self.map_scale}"
        if self.half_tw:
            name += "-htw"
        if self.fixed_horizon:
            name += "-fh"
        return name

    def initial_setup(self):
        policy_name = "default"
        burn_in_steps = 0

        seed = self.seed
        ring_size = 1 if self.inline else 20
        sample_steps = 160
        policy = Policy(type_="hns",
                        args=dict(
                            obs_space=RANDOMWALL_OBSERVATION_SPACE,
                            act_dims=[11, 11, 11, 2, 2],
                            seed=seed,
                            chunk_len=10,
                        ))
        env_configs = []
        for rank in range(self.num_actors):
            env_config = copy.deepcopy(_HNS_RANDOMWALL_CONFIG)
            env_config['scenario_name'] = 'hide_and_seek'
            if self.large_map:
                map_scale = self.map_scale
                if not self.fixed_horizon:
                    env_config['horizon'] = int(env_config['horizon'] * map_scale)
                env_config['restrict_rect'] = (np.array(env_config['restrict_rect']) * map_scale).tolist()
                env_config['hiders_together_radius'] *= map_scale
                env_config['seekers_together_radius'] *= map_scale
                env_config['floor_size'] = 6.0 * map_scale
                env_config['grid_size'] = int(30 * map_scale)
            env_config['n_hiders'] = random.randint(1, 3)
            env_config['n_seekers'] = random.randint(1, 3)
            env_config['seed'] = seed + rank * 10000
            env_configs.append(env_config)
        trainer = Trainer(type_="mappo",
                          args=dict(
                              discount_rate=0.998,
                              gae_lambda=0.95,
                              eps_clip=0.2,
                              clip_value=False,
                              dual_clip=False,
                              value_loss='mse',
                              value_loss_weight=0.5,
                              entropy_bonus_weight=0.01,
                              optimizer='adam',
                              optimizer_config=dict(lr=3e-4, eps=1e-5, weight_decay=1e-6),
                              popart=True,
                              max_grad_norm=5.0,
                              bootstrap_steps=1,
                              burn_in_steps=burn_in_steps,
                              ppo_epochs=1,
                          ))
        if not self.inline:
            inference_stream = policy_name
        else:
            if self.use_ps:
                inference_stream = InferenceStream(
                    type_=InferenceStream.Type.INLINE,
                    policy=policy,
                    policy_name=policy_name,
                    stream_name="",
                    pull_interval_seconds=None,
                    parameter_service_client=ParameterServiceClient(),
                )
            else:
                inference_stream = InferenceStream(
                    type_=InferenceStream.Type.INLINE,
                    policy=policy,
                    policy_name=policy_name,
                    stream_name="",
                    pull_interval_seconds=1.0,
                )

        sample_producer = SampleStream(type_=SampleStream.Type.NAME,
                                       stream_name=policy_name,
                                       serialization_method="pickle_compress")
        sample_consumer = SampleStream(type_=SampleStream.Type.NAME,
                                       stream_name=policy_name,
                                       serialization_method="pickle_compress")

        return ExperimentConfig(
            actors=[
                ActorWorker(
                    env=[
                        Environment(type_="hide_and_seek",
                                    args={
                                        **env_config, 'seed': env_config['seed'] + rank * 123
                                    }) for rank in range(ring_size)
                    ],
                    # decorrelate_seconds=(aw_i // self.num_trainers) * 30 /
                    # (self.num_actors // self.num_trainers),
                    inference_streams=[inference_stream],
                    sample_streams=[sample_producer],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=False,
                            send_after_done=False,
                            sample_steps=sample_steps,
                            bootstrap_steps=1,
                            burn_in_steps=burn_in_steps,
                        )
                    ],
                    max_num_steps=2000,
                    ring_size=ring_size,
                ) for aw_i, env_config in enumerate(env_configs)
            ],
            policies=[
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=inference_stream,
                    policy=policy,
                    max_inference_delay=0.05,
                    pull_frequency_seconds=0.2,
                    parameter_service_client=ParameterServiceClient() if self.use_ps else None,
                ) for j in range(self.num_policies + 8)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(
                        max_size=16,
                        reuses=4,
                        batch_size=int(8e3 * self.scale_factor) // (sample_steps // 10) // self.num_trainers,
                    ),
                    policy_name=policy_name,
                    trainer=trainer,
                    policy=policy,
                    log_frequency_seconds=10,
                    push_frequency_seconds=0.5,
                    preemption_steps=200,
                    push_tag_frequency_minutes=30,
                    sample_stream=sample_consumer,
                    worker_info=WorkerInformation(
                        wandb_project="hns-reproduce-debug",
                        wandb_name=f"seed{self.seed}",
                        wandb_group=self.wandb_group,
                        wandb_job_type="tw",
                        log_wandb=(i == 0),
                    ),
                ) for i in range(self.num_trainers)
            ],
            parameter_server_worker=[ParameterServerWorker()] if self.use_ps else [],
            timeout_seconds=7 * 86400,
        )


for seed, scale in itertools.product([1, 2, 3], [1, 4, 8, 16, 32]):
    register_experiment(
        f"hns-reproduce-s{seed}x{scale}",
        functools.partial(HideAndSeekRandomWallExperiment, False, scale_factor=scale, seed=seed))
    for map_scale in [1.5, 2.0]:
        register_experiment(
            f"hns-reproduce-s{seed}x{scale}-lmx{map_scale}",
            functools.partial(HideAndSeekRandomWallExperiment,
                              False,
                              scale_factor=scale,
                              seed=seed,
                              large_map=True,
                              map_scale=map_scale,
                              fixed_horizon=False))
        register_experiment(
            f"hns-reproduce-s{seed}x{scale}-lmx{map_scale}-fh",
            functools.partial(HideAndSeekRandomWallExperiment,
                              False,
                              scale_factor=scale,
                              seed=seed,
                              large_map=True,
                              map_scale=map_scale,
                              fixed_horizon=True))
        register_experiment(
            f"hns-reproduce-s{seed}x{scale}-inline-cpux2-lmx{map_scale}-fh",
            functools.partial(HideAndSeekRandomWallExperiment,
                              True,
                              use_ps=False,
                              cpu_scale=2.0,
                              scale_factor=scale,
                              seed=seed,
                              large_map=True,
                              map_scale=map_scale,
                              fixed_horizon=True))

    register_experiment(
        f"hns-reproduce-s{seed}x{scale}-inline",
        functools.partial(HideAndSeekRandomWallExperiment,
                          True,
                          scale_factor=scale,
                          seed=seed,
                          use_ps=False,
                          fixed_horizon=False))
    register_experiment(
        f"hns-reproduce-s{seed}x{scale}-inline-cpux2",
        functools.partial(HideAndSeekRandomWallExperiment,
                          True,
                          scale_factor=scale,
                          seed=seed,
                          use_ps=False,
                          cpu_scale=2.0,
                          fixed_horizon=False))
