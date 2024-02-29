import dataclasses
import functools
import itertools
import math

from api.config import *

atari_action_dims = {
    'adventure': 18,
    'airraid': 6,
    'alien': 18,
    'amidar': 10,
    'assault': 7,
    'asterix': 9,
    'asteroids': 14,
    'atlantis': 4,
    'bankheist': 18,
    'battlezone': 18,
    'beamrider': 9,
    'berzerk': 18,
    'bowling': 6,
    'boxing': 18,
    'breakout': 4,
    'carnival': 6,
    'centipede': 18,
    'choppercommand': 18,
    'crazyclimber': 9,
    'defender': 18,
    'demonattack': 6,
    'doubledunk': 18,
    'elevatoraction': 18,
    'enduro': 9,
    'fishingderby': 18,
    'freeway': 3,
    'frostbite': 18,
    'gopher': 8,
    'gravitar': 18,
    'hero': 18,
    'icehockey': 18,
    'jamesbond': 18,
    'journeyescape': 16,
    'kangaroo': 18,
    'krull': 18,
    'kungfumaster': 14,
    'montezumarevenge': 18,
    'mspacman': 9,
    'namethisgame': 6,
    'phoenix': 8,
    'pitfall': 18,
    'pong': 6,
    'pooyan': 6,
    'privateeye': 18,
    'qbert': 6,
    'riverraid': 18,
    'roadrunner': 18,
    'robotank': 18,
    'seaquest': 18,
    'skiing': 3,
    'solaris': 18,
    'spaceinvaders': 6,
    'stargunner': 18,
    'tennis': 18,
    'timepilot': 10,
    'tutankham': 8,
    'upndown': 6,
    'venture': 18,
    'videopinball': 9,
    'wizardofwor': 10,
    'yarsrevenge': 18,
    'zaxxon': 18
}


class AtariMiniExperiment(Experiment):

    def __init__(self):
        self.aws = 1
        self.pws = 1
        self.tws = 1

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(count=self.aws, scheduling=Scheduling.actor_worker_default()),
            policies=TasksGroup(count=self.pws, scheduling=Scheduling.policy_worker_default()),
            trainers=TasksGroup(count=self.tws, scheduling=Scheduling.trainer_worker_default()),
        )

    def initial_setup(self):
        policy_name = "default"
        inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)
        policy = Policy(type_="atari_naive_rnn", args=dict(
            action_space=18,
            rnn_hidden_dim=128,
        ))
        return ExperimentConfig(
            actors=[
                ActorWorker(
                    env=Environment(
                        type_="atari",
                        args=dict(
                            game_name="Boxing-v0",
                            render=False,  # Switch this to True to display actor games.
                            pause=False,
                        )),
                    inference_streams=[inference_stream],
                    sample_streams=[sample_stream],
                    agent_specs=[AgentSpec(
                        index_regex=".*",
                        inference_stream_idx=0,
                        sample_stream_idx=0,
                    )],
                    ring_size=1,
                ) for _ in range(1)
            ],
            policies=[
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=inference_stream,
                    parameter_db=parameter_db,
                    policy=policy,
                ) for _ in range(1)
            ],
            trainers=[
                TrainerWorker(
                    policy_name=policy_name,
                    trainer="mappo",
                    policy=policy,
                    sample_stream=sample_stream,
                    parameter_db=parameter_db,
                ) for _ in range(1)
            ],
        )


class AtariExperiment(Experiment):

    def __init__(self):
        self.aws = 16
        self.pws = 8
        self.tws = 2

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(count=self.aws, scheduling=Scheduling.actor_worker_default(cpu=1, mem=1024)),
            policies=TasksGroup(count=self.pws,
                                scheduling=Scheduling.policy_worker_default(gpu=0.25, mem=1024 * 10)),
            trainers=TasksGroup(count=self.tws,
                                scheduling=Scheduling.trainer_worker_default(cpu=4, mem=1024 * 10)),
        )

    def initial_setup(self):
        policy_name = "default"
        inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)
        policy = Policy(type_="atari_naive_rnn",
                        args=dict(
                            action_dim=6,
                            obs_dim={"obs": (3, 96, 96)},
                            rnn_hidden_dim=32,
                        ))
        trainer = Trainer(type_="mappo",
                          args=dict(
                              discount_rate=0.98,
                              gae_lambda=0.97,
                              eps_clip=0.2,
                              value_loss='huber',
                              value_loss_weight=0.5,
                              value_loss_config=dict(delta=10.0,),
                              entropy_bonus_weight=0.02,
                              optimizer='adam',
                              optimizer_config=dict(lr=1e-4),
                              max_grad_norm=10.0,
                              entropy_decay_per_steps=1000,
                              entropy_bonus_decay=0.99,
                              bootstrap_steps=50,
                          ))
        return ExperimentConfig(
            actors=[
                ActorWorker(env=Environment(type_="atari",
                                            args=dict(game_name="PongNoFrameskip-v4", obs_shape=(96, 96))),
                            inference_streams=[inference_stream],
                            sample_streams=[sample_stream],
                            agent_specs=[
                                AgentSpec(
                                    index_regex=".*",
                                    inference_stream_idx=0,
                                    sample_stream_idx=0,
                                    send_full_trajectory=False,
                                    send_after_done=False,
                                    sample_steps=100,
                                    bootstrap_steps=50,
                                )
                            ],
                            max_num_steps=20000,
                            inference_splits=4,
                            ring_size=40) for _ in range(self.aws)
            ],
            policies=[
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=inference_stream,
                    parameter_db=parameter_db,
                    policy=policy,
                    pull_max_failures=1000,
                    pull_frequency_seconds=30,
                ) for _ in range(self.pws)
            ],
            trainers=[
                TrainerWorker(
                    buffer_args=dict(max_size=20, reuses=2, batch_size=20),
                    policy_name=policy_name,
                    trainer=trainer,
                    policy=policy,
                    sample_stream=sample_stream,
                    parameter_db=parameter_db,
                ) for _ in range(self.tws)
            ],
        )


class AtariMuzeroReanalyzeExperiment(Experiment):

    def __init__(self, env_name: str, seed: int = 1, scale: int = 1):
        self.env_name = env_name
        self.training_steps = 100000
        self.buffer_size = 40000
        self.sample_steps = 80

        self.rollout_update_interval = 100
        self.reanalyze_update_interval = 200

        self.num_aw = 1
        self.num_eval_aw = 10
        self.num_pw = max(1, int(scale * 1))
        self.num_eval_pw = 1
        self.num_bw = max(1, int(scale * 8))
        self.num_tw = 1
        self.num_em = 3

        self.do_reanalyze = True
        self.ring_size = int(2 * scale)
        self.inference_splits = 4
        self.batch_size = int(scale * 32)
        self.num_unroll_steps = 5
        self.td_steps = 5

        self.seed = seed

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(count=self.num_aw + self.num_eval_aw * 2,
                              scheduling=Scheduling.actor_worker_default(
                                  cpu=1,
                                  mem=150 * self.ring_size,
                                  container_image='marl/marl-cpu-blosc',
                              )),
            policies=TasksGroup(count=self.num_pw + self.num_eval_pw * 2,
                                scheduling=Scheduling.policy_worker_default(
                                    gpu=0.2,
                                    cpu=2,
                                    node_type=['g1', 'g2'],
                                    mem=1024 * 10,
                                    exclude='frl8g[134,136-137]',
                                    container_image='marl/marl-gpu-blosc',
                                )),
            trainers=TasksGroup(count=self.num_tw,
                                scheduling=Scheduling.trainer_worker_default(
                                    gpu=1,
                                    cpu=16,
                                    node_type=['g1', 'g2'],
                                    mem=1024 * 100,
                                    exclude='frl8g[134,136-137]',
                                    container_image='marl/marl-gpu-blosc',
                                )),
            buffers=TasksGroup(count=self.num_bw,
                               scheduling=Scheduling.buffer_worker_default(
                                   gpu=0.25,
                                   node_type=["g1", "g2"],
                                   mem=1024 * 30,
                                   cpu=3,
                                   exclude='frl8g[134,136-137]',
                                   container_image='marl/marl-gpu-blosc',
                               )),
            eval_managers=TasksGroup(count=self.num_em,
                                     scheduling=Scheduling.eval_manager_default(
                                         mem=4 * 1024,
                                         container_image='marl/marl-cpu-blosc',
                                     )),
            # parameter_server_worker=TasksGroup(count=2,
            #                                    scheduling=Scheduling.parameter_server_worker_default(mem=20 *
            #                                                                                          1024)),
            controller_image='marl/marl-cpu-blosc',
        )

    def initial_setup(self):
        policy_name = "default"

        # system experiments
        num_aw = self.num_aw
        num_pw = self.num_pw
        ring_size = self.ring_size
        inference_splits = self.inference_splits

        rollout_sample_stream_type = "BROADCAST"
        # rollout_sample_stream_type = "ROUND_ROBIN"

        # training
        if rollout_sample_stream_type == "ROUND_ROBIN":
            rollout_sample_stream = SampleStream(type_=SampleStream.Type.NAME_ROUND_ROBIN,
                                                 stream_name=f"rollout_{policy_name}",
                                                 serialization_method='obs_compress')
            buffer_size = self.buffer_size // self.num_bw // self.sample_steps
        else:
            rollout_sample_stream = SampleStream(type_=SampleStream.Type.NAME_BROADCAST,
                                                 stream_name=f"rollout_{policy_name}",
                                                 serialization_method='obs_compress')
            buffer_size = self.buffer_size // self.sample_steps
        buffer_rollout_sample_stream = SampleStream(type_=SampleStream.Type.NAME,
                                                    stream_name=f"rollout_{policy_name}",
                                                    serialization_method='obs_compress')
        train_sample_stream = SampleStream(type_=SampleStream.Type.NAME,
                                           stream_name=f"train_{policy_name}",
                                           serialization_method='obs_compress')
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)

        label = f"-a{self.rollout_update_interval}-r{self.reanalyze_update_interval}-merge-test"

        import os
        wandb_label = os.environ.get("WANDB_LABEL", "null")

        label = f"-a{self.rollout_update_interval}-r{self.reanalyze_update_interval}-{wandb_label}"

        wandb_args = dict(wandb_entity="garrett4wade",
                          wandb_project="sampled-muzero-atari",
                          wandb_group=f"{self.env_name}" + ("-reanalyze" if self.do_reanalyze else "") +
                          f"-{num_aw}actors" + label,
                          log_wandb=True,
                          wandb_config=dict(label=wandb_label, env_name=self.env_name))

        return ExperimentConfig(
            actors=[
                ActorWorker(  # Training.
                    env=self.make_env(rank=i, evaluation=False),
                    inference_streams=[policy_name],
                    sample_streams=[rollout_sample_stream, f"train_log_{policy_name}"],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=[0, 1],
                            sample_steps=self.sample_steps,
                            bootstrap_steps=self.num_unroll_steps + self.td_steps +
                            1,  # num_unroll_steps + td_steps + done
                            send_after_done=False,
                            send_full_trajectory=False,
                        )
                    ],
                    ring_size=ring_size,
                    inference_splits=inference_splits) for i in range(num_aw)
            ] + [
                ActorWorker(  # Evaluation with mcts
                    env=self.make_env(rank=i + num_aw, evaluation=True),
                    inference_streams=[f"eval_mcts_{policy_name}"],
                    sample_streams=[f"eval_mcts_{policy_name}"],
                    agent_specs=[
                        AgentSpec(
                            index_regex="0",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=True,
                            deterministic_action=True,
                            send_concise_info=True,
                        ),
                    ],
                    ring_size=ring_size,
                    inference_splits=1) for i in range(self.num_eval_aw)
            ] + [
                ActorWorker(  # Evaluation without mcts
                    env=self.make_env(rank=i + self.num_eval_aw + num_aw, evaluation=True),
                    inference_streams=[f"eval_nomcts_{policy_name}"],
                    sample_streams=[f"eval_nomcts_{policy_name}"],
                    agent_specs=[
                        AgentSpec(index_regex="0",
                                  inference_stream_idx=0,
                                  sample_stream_idx=0,
                                  send_full_trajectory=True,
                                  deterministic_action=True,
                                  send_concise_info=True),
                    ],
                    ring_size=ring_size,
                    inference_splits=1) for i in range(self.num_eval_aw)
            ],
            policies=[
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=policy_name,
                    parameter_db=parameter_db,
                    pull_max_failures=1000,
                    pull_frequency_seconds=0.05,
                    policy=self.make_policy(rank=i, evaluation=False, use_mcts=True),
                    # parameter_service_client=ParameterServiceClient(),
                ) for i in range(num_pw)
            ] + [  # Eval policy with mcts
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=f"eval_mcts_{policy_name}",
                    parameter_db=parameter_db,
                    pull_max_failures=999,
                    pull_frequency_seconds=5,
                    policy=self.make_policy(rank=i + num_pw, evaluation=True, use_mcts=True),
                    policy_identifier="evaluation_mcts",
                    # parameter_service_client=ParameterServiceClient(),
                ) for i in range(self.num_eval_pw)
            ] + [  # Eval policy without mcts
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=f"eval_nomcts_{policy_name}",
                    parameter_db=parameter_db,
                    pull_max_failures=999,
                    pull_frequency_seconds=5,
                    policy=self.make_policy(
                        rank=i + self.num_eval_pw + num_pw, evaluation=True, use_mcts=False),
                    policy_identifier="evaluation_nomcts",
                    # parameter_service_client=ParameterServiceClient(),
                ) for i in range(self.num_eval_pw)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='simple_queue',
                    buffer_args=dict(max_size=8),
                    policy_name=policy_name,
                    trainer=self.make_trainer(),
                    policy=self.make_policy(
                        rank=0, evaluation=False, use_mcts=False
                    ),  # evaluation & use_mcts could be arbitrary, they do not affect training
                    sample_stream=train_sample_stream,
                    parameter_db=parameter_db,
                    push_tag_frequency_minutes=20,
                    train_for_seconds=12 * 3600,
                    worker_info=WorkerInformation(wandb_job_type="trainer_worker",
                                                  wandb_name=f"tw_{i}",
                                                  **wandb_args)) for i in range(self.num_tw)
            ],
            eval_managers=[
                EvaluationManager(eval_sample_stream=f"eval_mcts_{policy_name}",
                                  parameter_db=parameter_db,
                                  policy_name=policy_name,
                                  eval_tag="evaluation_mcts",
                                  eval_games_per_version=100,
                                  worker_info=WorkerInformation(wandb_job_type="em_mcts",
                                                                wandb_name=f"em_mcts",
                                                                **wandb_args)),
                EvaluationManager(eval_sample_stream=f"eval_nomcts_{policy_name}",
                                  parameter_db=parameter_db,
                                  policy_name=policy_name,
                                  eval_tag="evaluation_nomcts",
                                  eval_games_per_version=100,
                                  worker_info=WorkerInformation(wandb_job_type="em_nomcts",
                                                                wandb_name=f"em_nomcts",
                                                                **wandb_args)),
                EvaluationManager(eval_sample_stream=f"train_log_{policy_name}",
                                  parameter_db=parameter_db,
                                  policy_name=policy_name,
                                  eval_tag="null",
                                  unique_policy_version=False,
                                  eval_games_per_version=10,
                                  worker_info=WorkerInformation(wandb_job_type="train_log",
                                                                wandb_name=f"train_log",
                                                                **wandb_args)),
            ],
            buffers=[
                BufferWorker(
                    buffer_name='simple_replay_buffer',
                    buffer_args=dict(
                        max_size=buffer_size,
                        warmup_transitions=10000,
                        batch_size=self.batch_size,
                        sample_length=self.sample_steps + self.num_unroll_steps + self.td_steps + 1,
                        batch_length=1 + self.num_unroll_steps + self.td_steps +
                        1,  # 1 + num_unroll_steps + td_steps + done
                        seed=i + self.seed * 114514,
                    ),
                    from_sample_stream=buffer_rollout_sample_stream,
                    to_sample_stream=train_sample_stream,
                    unpack_batch_before_post=False,
                    policy_name=policy_name,
                    policy=self.make_policy(rank=0, evaluation=False, use_mcts=True,
                                            is_reanalyze_policy=True),
                    policy_identifier="latest",
                    parameter_db=parameter_db,
                    pull_frequency_seconds=0.5,
                    # parameter_service_client=ParameterServiceClient(),
                ) for i in range(self.num_bw)
            ],
            # parameter_server_worker=[ParameterServerWorker()],
        )

    def make_trainer(self):
        trainer = Trainer(
            type_="muzero",
            args=dict(
                num_unroll_steps=self.num_unroll_steps,
                td_steps=self.td_steps,
                # optimization
                optimizer_name="sgd",
                optimizer_config=dict(lr=0., weight_decay=1e-4, momentum=0.9),
                lr_schedule=[
                    dict(name="linear", num_steps=1000, config=dict(lr_init=0., lr_end=0.2)),
                    dict(name="decay",
                         num_steps=200000,
                         config=dict(lr_init=0.2, lr_decay_rate=0.1, lr_decay_steps=100000)),
                ],
                max_grad_norm=5,
                # self-supervised model
                do_consistency=False,
                # loss
                reward_loss_coeff=1,
                value_loss_coeff=0.25,
                policy_loss_coeff=1,
                consistency_coeff=2,
            ))
        return trainer

    def make_policy(self, rank: int, evaluation: bool, use_mcts: bool, is_reanalyze_policy: bool = False):
        from legacy.algorithm.muzero.utils.scalar_transform import DiscreteSupport
        from legacy.algorithm.muzero.utils.utils import LinearSchedule
        value_support = reward_support = DiscreteSupport(-300, 300, delta=1)
        policy = Policy(
            type_=f"muzero-atari",
            args=dict(
                act_dim=atari_action_dims[self.env_name.split("NoFrameskip")[0].lower()],
                action_space=None,
                obs_shape=(12, 96, 96),  # 12 = 3 x 4 = RGB x stacked_observations
                discount=0.997**4,
                # tree search
                use_mcts=use_mcts,
                value_delta_max=0.01,
                num_simulations=50,
                root_dirichlet_alpha=0.3,
                root_exploration_fraction=0.25,
                pb_c_base=19652,
                pb_c_init=1.25,
                num_threads=16,
                # network initialization
                init_zero=True,
                # frame skip & stack observation
                frame_skip=4,
                stacked_observations=4,
                gray_scale=False,
                # value prefix
                lstm_hidden_size=512,
                use_value_prefix=True,
                value_prefix_horizon=5,
                # siamese
                proj_hid=1024,
                proj_out=1024,
                pred_hid=512,
                pred_out=1024,
                # image augmentation
                use_augmentation=False,
                augmentation=['shift', 'intensity'],
                # reanalyze
                reanalyze_ratio_schedule=LinearSchedule(v_init=0., v_end=float(self.do_reanalyze), t_end=1),
                td_steps=self.td_steps,
                num_unroll_steps=self.num_unroll_steps,
                # value & reward transform
                value_support=value_support,
                reward_support=reward_support,
                # actor exploration
                visit_softmax_temperature_fn=None if evaluation else self.visit_softmax_temperature_fn,
                warm_up_version=0 if evaluation or is_reanalyze_policy else 2000,
                # rollout & reanalyze network update
                rollout_update_interval=self.rollout_update_interval,
                reanalyze_update_interval=self.reanalyze_update_interval,
                # sampled muzero
                mcts_use_sampled_actions=False,
                mcts_num_sampled_actions=0,
                seed=rank * 97 + self.seed * 101))
        return policy

    def make_env(self, rank: int, evaluation: bool = False):
        return Environment(type_="atari",
                           args=dict(game_name=self.env_name,
                                     full_action_space=False,
                                     noop_max=30,
                                     episode_life=not evaluation,
                                     clip_reward=not evaluation,
                                     frame_skip=4,
                                     stacked_observations=4,
                                     max_episode_steps=108000,
                                     gray_scale=False,
                                     obs_shape=(96, 96),
                                     scale=False,
                                     seed=rank * 97 + self.seed * 10085))

    def visit_softmax_temperature_fn(self, version):
        if version < 0.5 * self.training_steps:
            return 1.0
        elif version < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class AtariVanillaPBTMiniExperiment(Experiment):

    def __init__(self):
        self.num_actors = 4
        self.num_policies = 2
        self.num_trainers = 1
        self.num_eval_actors = 2
        self.num_eval_policies = 1

        # Vanilla PBT configuration.
        self.population = ["policy_0", "policy_1"]
        self.initial_hyperparams = dict(
            policy_0=dict(
                eps_clip=0.1,
                delta=9.0,
                entropy_bonus_weight=0.01,
                lr=1e-3,
                max_grad_norm=9.0,
            ),
            policy_1=dict(
                eps_clip=0.2,
                delta=10.0,
                entropy_bonus_weight=0.02,
                lr=1e-4,
                max_grad_norm=10.0,
            ),
        )
        self.population_algorithm = PopulationAlgorithm(
            type_="vanilla_pbt",
            args=dict(
                population=self.population,
                ready_interval=100,
                truncation_ratio=0.5,
                explore_configs=[
                    dict(
                        keys="entropy_bonus_weight",
                        method="perturb",
                        factors=[0.8, 1.2],
                        min_value=0,
                        max_value=1,
                    ),
                    dict(
                        keys=["value_loss_config", "delta"],
                        method="perturb",
                        factors=[0.9, 1.1],
                    ),
                    dict(
                        keys="eps_clip",
                        method="resample",
                        distribution="categorical",
                        values=[0.1, 0.2, 0.3],
                    ),
                    dict(
                        keys="max_grad_norm",
                        method="resample",
                        distribution="uniform",
                        value_range=[8, 12],
                    ),
                    dict(
                        keys=["optimizer_config", "lr"],
                        method="resample",
                        distribution="log_uniform",
                        value_range=[1e-3, 1e-5],
                    ),
                ],
            ),
        )

        self.policy = Policy(type_="atari_naive_rnn", args=dict(action_space=18, rnn_hidden_dim=128))
        self.population_sample_stream = SampleStream(type_=SampleStream.Type.NAMED_EVAL,
                                                     stream_name="population_stream")
        self.null_sample_stream = SampleStream(type_=SampleStream.Type.NULL)

    def make_trainer(self, eps_clip, delta, entropy_bonus_weight, lr, max_grad_norm):
        return Trainer(type_="mappo",
                       args=dict(discount_rate=0.98,
                                 gae_lambda=0.97,
                                 eps_clip=eps_clip,
                                 value_loss='huber',
                                 value_loss_weight=0.5,
                                 value_loss_config=dict(delta=delta),
                                 entropy_bonus_weight=entropy_bonus_weight,
                                 optimizer='adam',
                                 optimizer_config=dict(lr=lr),
                                 max_grad_norm=max_grad_norm,
                                 entropy_decay_per_steps=1000,
                                 entropy_bonus_decay=0.99))

    def initial_setup(self):
        actors = []
        policies = []
        trainers = []
        eval_managers = []

        for idx, policy_name in enumerate(self.population):
            trainer = self.make_trainer(**self.initial_hyperparams[idx])
            inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
            sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
            eval_inference_stream = InferenceStream(type_=InferenceStream.Type.NAME,
                                                    stream_name=f"eval_{policy_name}")
            eval_sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=f"eval_{policy_name}")

            actors.extend([  # Training.
                ActorWorker(
                    env=Environment(type_="atari", args=dict(game_name="Boxing-v0")),
                    inference_streams=[inference_stream],
                    sample_streams=[sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            sample_steps=1000,
                            send_after_done=False,
                            send_full_trajectory=False,
                        )
                    ],
                    max_num_steps=2000,
                    ring_size=20,
                ) for _ in range(self.num_actors)
            ] + [  # Evaluation
                ActorWorker(
                    env=Environment(type_="atari", args=dict(game_name="Boxing-v0")),
                    inference_streams=[eval_inference_stream],
                    sample_streams=[
                        eval_sample_stream, self.population_sample_stream, self.null_sample_stream
                    ],
                    agent_specs=[
                        AgentSpec(
                            index_regex="0",
                            inference_stream_idx=0,
                            sample_stream_idx=[0, 1],
                            send_full_trajectory=True,
                            deterministic_action=True,
                        ),
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=2,
                            send_full_trajectory=True,
                            deterministic_action=True,
                        ),
                    ],
                    ring_size=20,
                ) for _ in range(self.num_eval_actors)
            ])

            policies.extend([  # Training.
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=inference_stream,
                    policy=self.policy,
                    pull_max_failures=1000,
                    pull_frequency_seconds=0.5,
                ) for _ in range(self.num_policies)
            ] + [  # Evaluation.
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=eval_inference_stream,
                    policy=self.policy,
                    policy_identifier="evaluation",
                    pull_max_failures=1000,
                    pull_frequency_seconds=2,
                ) for _ in range(self.num_eval_policies)
            ])

            trainers.extend([
                TrainerWorker(
                    policy_name=policy_name,
                    policy=self.policy,
                    trainer=trainer,
                    sample_stream=sample_stream,
                ) for _ in range(1)
            ])

            eval_managers.extend([
                EvaluationManager(
                    eval_sample_stream=eval_sample_stream,
                    policy_name=policy_name,
                    eval_tag="evaluation",
                    eval_games_per_version=10,
                )
            ])

        population_manager = [
            PopulationManager(
                population=self.population,
                population_algorithm=self.population_algorithm,
                population_sample_stream=self.population_sample_stream,
                actors=actors,
                policies=policies,
                trainers=trainers,
                eval_managers=eval_managers,
            ),
        ]
        return ExperimentConfig(actors=actors,
                                policies=policies,
                                trainers=trainers,
                                eval_managers=eval_managers,
                                population_manager=population_manager)


@dataclasses.dataclass
class AtariPPOExperiment(Experiment):
    game_name: str
    seed: int
    scale: int

    dual_clip: bool = False
    vtrace: bool = False
    popart: bool = False
    bootstrap_steps: int = 50
    data_reuse: int = 5
    compute_gae_on_aw: bool = False
    num_rnn_layers: int = 0
    burn_in_steps: int = 0
    unbiased_popart: bool = False
    chunk_len: int = 40
    tw_preemption_steps: float = math.inf

    gae_lambda: float = 0.97
    shared_backbone: bool = True
    sample_steps: int = 80
    discount_rate: float = 0.99

    rnn_type: str = 'lstm'
    hidden_dim: int = 512

    def __post_init__(self):
        if self.sample_steps % self.chunk_len != 0:
            raise ValueError("sample_steps must be the multiple of chunk_len!")

        self.aws = int(self.scale * 32)
        self.pws = max(1, self.aws // 8)
        self.tws = math.ceil(self.scale / 8)
        self.eval_aws = 64
        self.ring_size = 40

        self.batch_size = int(self.scale * 32)

        self.buffer_max_size = 6

        if self.burn_in_steps > 0 and self.num_rnn_layers < 1:
            self.burn_in_steps = 0
        if self.num_rnn_layers == 0:
            self.chunk_len = 40

    @property
    def wandb_group_name(self):
        postfix = ""
        if not self.dual_clip:
            postfix += "-ndc"
        if self.vtrace:
            postfix += "-vtr"
        if self.popart:
            if self.unbiased_popart:
                postfix += '-upa'
            else:
                postfix += "-pa"
        if self.compute_gae_on_aw:
            postfix += "-awg"
        if self.burn_in_steps > 0:
            postfix += f"-bi{self.burn_in_steps}"
        if self.chunk_len != 40:
            postfix += f"-ch{self.chunk_len}"
        if self.tw_preemption_steps < math.inf:
            postfix += f"-e{self.tw_preemption_steps}"
        if self.shared_backbone:
            postfix += "-sb"

        return (f"aw{self.aws}bo{self.bootstrap_steps}lm{self.gae_lambda}"
                f"b{self.batch_size}r{self.data_reuse}rn{self.num_rnn_layers}st{self.sample_steps}" + postfix)

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(count=self.aws + self.eval_aws,
                              scheduling=Scheduling.actor_worker_default(
                                  cpu=1,
                                  mem=150 * self.ring_size,
                                  container_image='marl/marl-cpu-blosc',
                              )),
            policies=TasksGroup(count=self.pws,
                                scheduling=Scheduling.policy_worker_default(
                                    cpu=2,
                                    gpu=0.125,
                                    mem=1024 * 10,
                                    node_type=['g1', 'g2'],
                                    exclude='frl8g134,frl8g[136-137]',
                                    container_image='marl/marl-gpu-blosc',
                                )),
            trainers=TasksGroup(count=self.tws,
                                scheduling=Scheduling.trainer_worker_default(
                                    cpu=4,
                                    mem=1024 * 60,
                                    node_type=['g1', 'g2', 'g8'] if self.scale < 16 else ['g8'],
                                    container_image='marl/marl-gpu-blosc',
                                )),
            eval_managers=TasksGroup(count=1,
                                     scheduling=Scheduling.eval_manager_default(
                                         cpu=1,
                                         mem=4000,
                                         container_image='marl/marl-cpu-blosc',
                                     )),
            controller_image='marl/marl-cpu-blosc',
        )

    def make_env(self, seed, is_eval):
        # the same as DQN
        return Environment(type_='atari',
                           args=dict(
                               game_name=self.game_name,
                               full_action_space=False,
                               noop_max=30,
                               episode_life=(not is_eval),
                               clip_reward=(not is_eval),
                               frame_skip=4,
                               stacked_observations=4,
                               gray_scale=True,
                               obs_shape=(84, 84),
                               scale=False,
                               seed=seed,
                           ))

    def initial_setup(self):
        import gym
        policy_name = "default"
        policy = Policy(type_="actor-critic",
                        args=dict(
                            obs_dim={"obs": (4, 84, 84)},
                            action_dim=atari_action_dims[self.game_name.split("NoFrameskip")[0].lower()],
                            num_dense_layers=0,
                            hidden_dim=self.hidden_dim,
                            popart=self.popart,
                            layernorm=False,
                            shared_backbone=self.shared_backbone,
                            rnn_type=self.rnn_type,
                            num_rnn_layers=self.num_rnn_layers,
                            seed=self.seed,
                            cnn_layers=dict(obs=[(32, 8, 4, 0, 'zeros'), (64, 4, 2, 0,
                                                                          'zeros'), (64, 3, 1, 0, 'zeros')]),
                            chunk_len=self.chunk_len,
                            denormalize_value_during_rollout=self.compute_gae_on_aw,
                            unbiased_popart=self.unbiased_popart,
                        ))
        trainer = Trainer(type_="mappo",
                          args=dict(
                              discount_rate=self.discount_rate,
                              gae_lambda=self.gae_lambda,
                              eps_clip=0.2,
                              clip_value=True,
                              dual_clip=self.dual_clip,
                              vtrace=self.vtrace,
                              value_loss='huber',
                              value_loss_weight=1.0,
                              value_loss_config=dict(delta=10.0,),
                              entropy_bonus_weight=0.01,
                              optimizer='adam',
                              optimizer_config=dict(lr=5e-4),
                              popart=self.popart,
                              max_grad_norm=40.0,
                              bootstrap_steps=self.bootstrap_steps,
                              recompute_adv_among_epochs=False,
                              recompute_adv_on_reuse=False,
                              burn_in_steps=self.burn_in_steps,
                              normalize_old_value=self.compute_gae_on_aw,
                          ))

        eval_inference_stream = InferenceStream(
            type_=InferenceStream.Type.INLINE,
            stream_name=f"eval_{policy_name}",
            policy=policy,
            policy_name=policy_name,
            policy_identifier='eval',
        )

        wandb_args = dict(
            wandb_project=f'atari_ppo_{self.game_name}',
            wandb_group=self.wandb_group_name,
            wandb_config={f.name: getattr(self, f.name)
                          for f in dataclasses.fields(self)},
        )
        ring_size = self.ring_size
        eval_ring_size = 1

        sample_stream = SampleStream(type_=SampleStream.Type.NAME,
                                     stream_name=policy_name,
                                     serialization_method='obs_compress')
        inference_stream = InferenceStream(
            type_=InferenceStream.Type.NAME,
            stream_name=policy_name,
        )

        return ExperimentConfig(
            actors=[
                ActorWorker(
                    env=[
                        self.make_env(seed=self.seed * 1000 + (ring_size * i + j) * 97, is_eval=False)
                        for j in range(ring_size)
                    ],
                    inference_streams=[inference_stream],
                    sample_streams=[sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            sample_steps=self.sample_steps,
                            bootstrap_steps=self.bootstrap_steps,
                            burn_in_steps=self.burn_in_steps,
                            send_after_done=self.compute_gae_on_aw,
                            compute_gae_before_send=self.compute_gae_on_aw,
                            gae_args=dict(gamma=self.discount_rate, lmbda=self.gae_lambda),
                        )
                    ],
                    max_num_steps=108000,
                    inference_splits=4,
                    ring_size=ring_size,
                ) for i in range(self.aws)
            ] + [
                ActorWorker(
                    env=[
                        self.make_env(seed=self.seed * 1000 + (eval_ring_size * i + j) * 101, is_eval=True)
                        for j in range(eval_ring_size)
                    ],
                    inference_streams=[eval_inference_stream],
                    sample_streams=[f"eval_{policy_name}"],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=True,
                            send_concise_info=True,
                            deterministic_action=True,
                        )
                    ],
                    max_num_steps=108000,
                    ring_size=eval_ring_size,
                ) for i in range(self.eval_aws)
            ],
            policies=[
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=inference_stream,
                    policy=policy,
                    max_inference_delay=0.05,
                    pull_max_failures=100,
                    pull_frequency_seconds=0.01,
                ) for _ in range(self.pws)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(
                        max_size=self.buffer_max_size,
                        reuses=self.data_reuse,
                        batch_size=self.batch_size // self.tws,
                    ),
                    cudnn_benchmark=True,
                    policy_name=policy_name,
                    trainer=trainer,
                    push_frequency_seconds=5,
                    push_frequency_steps=25,
                    log_frequency_seconds=5,
                    policy=policy,
                    preemption_steps=self.tw_preemption_steps,
                    sample_stream=sample_stream,
                    train_for_seconds=8 * 3600,
                    worker_info=WorkerInformation(
                        wandb_name=f"tw_seed{self.seed}",
                        wandb_job_type="tw",
                        log_wandb=(i == 0),
                        **wandb_args,
                    ),
                ) for i in range(self.tws)
            ],
            eval_managers=[
                EvaluationManager(
                    eval_sample_stream=f"eval_{policy_name}",
                    policy_name=policy_name,
                    eval_games_per_version=100,
                    worker_info=WorkerInformation(
                        wandb_job_type="em",
                        wandb_name=f"em_seed{self.seed}",
                        log_wandb=True,
                        **wandb_args,
                    ),
                ),
            ],
        )


@dataclasses.dataclass
class AtariDQNExperiment(Experiment):
    game_name: str
    seed: int
    scale: float

    num_rnn_layers: int = 0
    discount_rate: float = 0.99

    use_per: bool = True
    use_per_weight: bool = True
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_interpolation_eta: float = 0.9

    apply_scalar_transform: bool = False
    episode_life: bool = False

    full_action_space: bool = False
    buffer_total_transitions: int = int(2e6)

    def __post_init__(self):
        self.aws = int(32 * self.scale)
        self.pws = max(1, self.aws // 8)
        self.tws = math.ceil(self.scale / 8)
        self.eval_aws = 64
        self.ring_size = 40

        self.sample_steps = 80
        self.burn_in_steps = 40 if self.num_rnn_layers > 0 else 0
        self.chunk_len = 40
        self.bootstrap_steps = 3
        self.batch_size = int(32 * self.scale)

    @property
    def wandb_group_name(self):
        postfix = ""
        if self.use_per:
            postfix += f"-per{self.priority_alpha}*{self.priority_beta}e{self.priority_interpolation_eta}"
            if self.use_per_weight:
                postfix += 'w'
        if self.apply_scalar_transform:
            postfix += "-ht"
        if self.episode_life:
            postfix += '-epl'
        if self.scale is not None:
            postfix += f"-x{self.scale}"

        return (f"rn{self.num_rnn_layers}" + postfix)

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(
                count=self.aws + self.eval_aws,
                scheduling=Scheduling.actor_worker_default(
                    cpu=1,
                    #   exclude='frl1g[069-072]',
                    mem=150 * self.ring_size,
                    container_image='marl/marl-cpu-blosc',
                )),
            policies=TasksGroup(
                count=self.pws,
                scheduling=Scheduling.policy_worker_default(
                    cpu=2,
                    gpu=0.128,
                    # exclude='frl2g021,frl8g137,frl2g004,frl2g032',
                    exclude='frl8g134,frl8g[136-137]',
                    mem=1024 * 20,
                    container_image='marl/marl-gpu-blosc',
                )),
            trainers=TasksGroup(count=self.tws,
                                scheduling=Scheduling.trainer_worker_default(
                                    cpu=4,
                                    gpu=1,
                                    gpu_type='geforce',
                                    mem=60 * 1024,
                                    node_type=['g1', 'g2', 'g8'] if self.scale < 16 else ['g8'],
                                    container_image='marl/marl-gpu-blosc',
                                )),
            eval_managers=TasksGroup(count=1,
                                     scheduling=Scheduling.eval_manager_default(
                                         cpu=1,
                                         mem=4000,
                                         container_image='marl/marl-cpu-blosc',
                                     )),
            controller_image='marl/marl-cpu-blosc',
        )

    def make_env(self, seed, epsilon, is_eval):
        return Environment(
            type_='atari',
            args=dict(
                game_name=self.game_name,
                full_action_space=self.full_action_space,
                noop_max=30,
                episode_life=((not is_eval) and self.episode_life),
                clip_reward=((not is_eval) and not self.apply_scalar_transform),
                frame_skip=4,
                stacked_observations=4,
                gray_scale=True,
                obs_shape=(84, 84),
                scale=False,
                seed=seed,
                obs_include_last_action=True,
                obs_include_last_reward=True,
                epsilon=epsilon if not is_eval else 0.1,
            ),
        )

    def initial_setup(self):
        import base.timeutil
        policy_name = "default"
        policy = Policy(
            type_="atari-dqn",
            args=dict(act_dim=atari_action_dims[self.game_name.split("NoFrameskip")[0].lower()],
                      dueling=True,
                      num_rnn_layers=self.num_rnn_layers,
                      chunk_len=self.chunk_len,
                      use_double_q=True,
                      seed=self.seed,
                      rnn_type='lstm',
                      rnn_include_last_action=False,
                      rnn_include_last_reward=False,
                      use_env_epsilon=True,
                      epsilon_scheduler=base.timeutil.ChainedScheduler([
                          base.timeutil.LinearScheduler(1.0, int(1e4), 0.1),
                          base.timeutil.ConstantScheduler(0.1, int(1e10))
                      ])),
        )
        trainer = Trainer(
            type_="q-learning",
            args=dict(
                gamma=self.discount_rate,
                bootstrap_steps=self.bootstrap_steps,
                burn_in_steps=self.burn_in_steps,
                use_soft_update=False,
                hard_update_interval=int(2.5e3),
                max_grad_norm=40.0,
                value_loss='smoothl1',
                optimizer='rmsprop',
                optimizer_config=dict(lr=2.5e-4 / 4, eps=1.5e-7, alpha=0.95),
                use_priority_weight=self.use_per_weight,
                priority_interpolation_eta=self.priority_interpolation_eta,
                apply_scalar_transform=self.apply_scalar_transform,
            ),
        )
        if self.use_per:
            buffer_config = dict(
                buffer_name='prioritized_replay_buffer',
                buffer_args=dict(
                    max_size=int(self.buffer_total_transitions) // self.sample_steps,
                    batch_size=self.batch_size // self.tws,
                    sample_length=self.burn_in_steps + self.sample_steps,
                    warmup_transitions=int(5e4),
                    seed=self.seed,
                    burn_in_steps=self.burn_in_steps,
                    alpha=self.priority_alpha,
                    beta=self.priority_beta,
                    beta_scheduler=None,
                    max_priority=1.0,
                    priority_interpolation_eta=self.priority_interpolation_eta,
                ),
            )
        else:
            buffer_config = dict(
                buffer_name='simple_replay_buffer',
                buffer_args=dict(
                    max_size=int(self.buffer_total_transitions) // self.sample_steps,
                    batch_size=self.batch_size,
                    sample_length=self.burn_in_steps + self.sample_steps,
                    batch_length=self.burn_in_steps + self.sample_steps,
                    warmup_transitions=int(5e4),
                    seed=self.seed,
                ),
            )

        eval_inference_stream = InferenceStream(
            type_=InferenceStream.Type.INLINE,
            stream_name=f"eval_{policy_name}",
            policy=policy,
            policy_name=policy_name,
            policy_identifier='eval',
        )

        wandb_args = dict(
            wandb_project=f'atari_dqn_{self.game_name}',
            wandb_group=self.wandb_group_name,
            wandb_config={f.name: getattr(self, f.name)
                          for f in dataclasses.fields(self)},
        )
        ring_size = self.ring_size
        eval_ring_size = 1

        sample_stream = SampleStream(type_=SampleStream.Type.NAME,
                                     stream_name=policy_name,
                                     serialization_method='obs_compress')
        inference_stream = InferenceStream(
            type_=InferenceStream.Type.NAME,
            stream_name=policy_name,
        )

        return ExperimentConfig(
            actors=[
                ActorWorker(
                    env=[
                        self.make_env(
                            seed=self.seed * 1000 + (ring_size * i + j) * 97,
                            epsilon=0.4**(1 + (ring_size * i + j) / (self.aws * ring_size - 1) * 7),
                            is_eval=False,
                        ) for j in range(ring_size)
                    ],
                    inference_streams=[inference_stream],
                    sample_streams=[sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_after_done=False,
                            sample_steps=self.sample_steps,
                            bootstrap_steps=0,
                            burn_in_steps=self.burn_in_steps,
                        )
                    ],
                    max_num_steps=108000,
                    inference_splits=4,
                    ring_size=ring_size,
                ) for i in range(self.aws)
            ] + [
                ActorWorker(
                    env=[
                        self.make_env(
                            seed=self.seed * 1000 + (eval_ring_size * i + j) * 101,
                            epsilon=0.0,  # this will not be used
                            is_eval=True,
                        ) for j in range(eval_ring_size)
                    ],
                    inference_streams=[eval_inference_stream],
                    sample_streams=[f"eval_{policy_name}"],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=True,
                            send_concise_info=True,
                            deterministic_action=True,
                        )
                    ],
                    max_num_steps=108000,
                    ring_size=eval_ring_size,
                ) for i in range(self.eval_aws)
            ],
            policies=[
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=inference_stream,
                    policy=policy,
                    max_inference_delay=0.05,
                    pull_max_failures=100,
                    pull_frequency_seconds=0.01,
                ) for _ in range(self.pws)
            ],
            trainers=[
                TrainerWorker(
                    **buffer_config,
                    cudnn_benchmark=True,
                    policy_name=policy_name,
                    trainer=trainer,
                    push_frequency_seconds=0.01,
                    push_frequency_steps=5,
                    log_frequency_seconds=50,
                    policy=policy,
                    sample_stream=sample_stream,
                    worker_info=WorkerInformation(
                        wandb_name=f"tw_seed{self.seed}",
                        wandb_job_type="tw",
                        log_wandb=(i == 0),
                        **wandb_args,
                    ),
                    train_for_seconds=8 * 3600,
                ) for i in range(self.tws)
            ],
            eval_managers=[
                EvaluationManager(
                    eval_sample_stream=f"eval_{policy_name}",
                    policy_name=policy_name,
                    eval_games_per_version=100,
                    worker_info=WorkerInformation(
                        wandb_job_type="em",
                        wandb_name=f"em_seed{self.seed}",
                        log_wandb=True,
                        **wandb_args,
                    ),
                ),
            ],
        )


register_experiment("atari-mini", AtariMiniExperiment)
register_experiment("atari", AtariExperiment)
register_experiment("atari-pbt-mini", AtariVanillaPBTMiniExperiment)
for game in [
        'Adventure', 'AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis', 'BankHeist',
        'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'Carnival', 'Centipede',
        'ChopperCommand', 'CrazyClimber', 'Defender', 'DemonAttack', 'DoubleDunk', 'ElevatorAction', 'Enduro',
        'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar', 'Hero', 'IceHockey', 'Jamesbond',
        'JourneyEscape', 'Kangaroo', 'Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman', 'NameThisGame',
        'Phoenix', 'Pitfall', 'Pong', 'Pooyan', 'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank',
        'Seaquest', 'Skiing', 'Solaris', 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham',
        'UpNDown', 'Venture', 'VideoPinball', 'WizardOfWor', 'YarsRevenge', 'Zaxxon'
]:
    register_experiment(f"atari-reanalyze-muzero-{game}",
                        functools.partial(AtariMuzeroReanalyzeExperiment, env_name=f"{game}NoFrameskip-v4"))

game_names = [
    'BattleZone',
    'DoubleDunk',
    'NameThisGame',
    'Phoenix',
    'Qbert',
]
seeds = list(range(32))
scales = [0.5, 1, 2, 4, 8, 16, 64]
for game, seed, scale in itertools.product(game_names, seeds, scales):
    register_experiment(
        f"{game}-ppo-s{seed}-x{scale}",
        functools.partial(AtariPPOExperiment, game_name=f"{game}NoFrameskip-v4", seed=seed, scale=scale))
    register_experiment(
        f"{game}-apex-s{seed}-x{scale}",
        functools.partial(AtariDQNExperiment, game_name=f"{game}NoFrameskip-v4", seed=seed, scale=scale))
    register_experiment(
        f"{game}-muzero-s{seed}-x{scale}",
        functools.partial(AtariMuzeroReanalyzeExperiment,
                          env_name=f"{game}NoFrameskip-v4",
                          scale=scale,
                          seed=seed))
