import functools
import itertools

from api.config import *

gym_mujoco_registry = {
    # env_name: (obs_dim, act_dim)
    "Humanoid-v3": (376, 17),
    "HumanoidStandup-v2": (376, 17),
    "HalfCheetah-v3": (17, 6),
    "Ant-v3": (111, 8),
    "Walker2d-v3": (17, 6),
    "Hopper-v3": (11, 3),
    "Swimmer-v3": (8, 2),
    "InvertedPendulum-v2": (4, 1),
    "InvertedDoublePendulum-v2": (11, 1),
    "Reacher-v2": (11, 2),
}


class GymMuJoCoMiniExperiment(Experiment):

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(
                count=2,
                scheduling=Scheduling.actor_worker_default(container_image="marl/marl-cpu-gym_mujoco")),
            policies=TasksGroup(count=1, scheduling=Scheduling.policy_worker_default()),
            trainers=TasksGroup(count=1, scheduling=Scheduling.trainer_worker_default()),
        )

    def initial_setup(self):
        scenario = 'Humanoid-v3'
        policy_name = "default"
        inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)
        policy = Policy(type_="gym_mujoco",
                        args=dict(obs_dim=gym_mujoco_registry[scenario][0],
                                  action_dim=gym_mujoco_registry[scenario][1]))
        return ExperimentConfig(
            actors=[
                ActorWorker(
                    inference_streams=[inference_stream],
                    sample_streams=[sample_stream],
                    env=Environment(type_="gym_mujoco", args=dict(scenario=scenario)),
                    agent_specs=[AgentSpec(
                        index_regex=".*",
                        inference_stream_idx=0,
                        sample_stream_idx=0,
                    )],
                ) for _ in range(2)
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


class GymMuJoCoExperiment(Experiment):

    def __init__(self,
                 scenario: str,
                 multiplier: int,
                 algo_name: str,
                 ppo_epochs=1,
                 ppo_iterations=32,
                 ppg_epochs: int = 6,
                 seed: int = 0):
        self.scenario = scenario
        self.algo_name = algo_name
        assert self.algo_name == 'mappo'
        self.ppg_epochs = ppg_epochs
        self.seed = seed
        self.policy_name = policy_name = "default"
        self.num_actors = 128 * multiplier
        self.num_policies = 4 * multiplier
        self.num_trainer = 1 * multiplier
        self.num_eval_actors = 32
        self.num_eval_pw = 1
        self.ring_size = 128
        self.inference_splits = 4

        self.trainer = Trainer(type_=algo_name,
                               args=dict(
                                   discount_rate=0.99,
                                   gae_lambda=0.95,
                                   vtrace=False,
                                   clip_value=True,
                                   eps_clip=0.2,
                                   value_loss='huber',
                                   value_loss_weight=1,
                                   value_loss_config=dict(delta=10.0,),
                                   entropy_bonus_weight=0.01,
                                   optimizer='adam',
                                   optimizer_config=dict(lr=5e-4),
                                   ppg_optimizer="adam",
                                   ppg_optimizer_config=dict(lr=5e-4),
                                   beta_clone=1,
                                   ppo_epochs=1 if self.algo_name == "mappo" else ppo_epochs,
                                   ppo_iterations=ppo_iterations,
                                   ppg_epochs=self.ppg_epochs,
                                   local_cache_stack_size=None,
                                   popart=True,
                                   max_grad_norm=10.0,
                                   log_per_steps=1,
                                   decay_per_steps=1000,
                                   entropy_bonus_decay=1,
                                   bootstrap_steps=1,
                               ))
        self.policy = Policy(type_="gym_mujoco",
                             args=dict(obs_dim=gym_mujoco_registry[scenario][0],
                                       action_dim=gym_mujoco_registry[scenario][1],
                                       num_rnn_layers=0,
                                       num_dense_layers=2,
                                       popart=True,
                                       activation='tanh',
                                       layernorm=True,
                                       std_type='separate_learnable',
                                       init_log_std=-0.5,
                                       seed=seed))

    def make_env(self, env_seed):
        return Environment(type_="gym_mujoco",
                           args=dict(
                               scenario=self.scenario,
                               action_squash_type=None,
                               seed=env_seed,
                           ))

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(actors=TasksGroup(
            count=self.num_actors + self.num_eval_actors,
            scheduling=Scheduling.actor_worker_default(container_image="marl/marl-cpu-gym_mujoco")),
                                    policies=TasksGroup(
                                        count=self.num_policies + self.num_eval_pw,
                                        scheduling=Scheduling.policy_worker_default(gpu=0.12)),
                                    trainers=TasksGroup(count=self.num_trainer,
                                                        scheduling=Scheduling.trainer_worker_default()),
                                    eval_managers=TasksGroup(count=1,
                                                             scheduling=Scheduling.eval_manager_default()))

    def initial_setup(self):
        return ExperimentConfig(
            actors=[
                ActorWorker(  # Training.
                    env=self.make_env(env_seed=12023 + i),
                    inference_streams=[self.policy_name],
                    sample_streams=[self.policy_name],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            sample_steps=500,
                            bootstrap_steps=1,
                            send_after_done=False,
                            send_full_trajectory=False,
                        )
                    ],
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                ) for i in range(self.num_actors)
            ] + [
                ActorWorker(  # Evaluation
                    env=self.make_env(env_seed=20231 + i),
                    inference_streams=[self.policy_name],
                    sample_streams=["eval" + self.policy_name],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=True,
                            deterministic_action=True,
                        ),
                    ],
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                ) for i in range(self.num_eval_actors)
            ],
            policies=[
                PolicyWorker(
                    policy_name=self.policy_name,
                    inference_stream=self.policy_name,
                    policy=self.policy,
                ) for _ in range(self.num_policies)
            ] + [
                PolicyWorker(
                    policy_name=self.policy_name,
                    inference_stream="eval" + self.policy_name,
                    policy=self.policy,
                    policy_identifier="evaluation",
                ) for _ in range(self.num_eval_pw)
            ],
            trainers=[
                TrainerWorker(buffer_name='priority_queue',
                              buffer_args=dict(
                                  max_size=64,
                                  reuses=1 if self.algo_name == "mappg" else 10,
                                  batch_size=256,
                              ),
                              policy_name=self.policy_name,
                              trainer=self.trainer,
                              policy=self.policy,
                              sample_stream=self.policy_name,
                              push_tag_frequency_minutes=5) for _ in range(self.num_trainer)
            ],
            eval_managers=[
                EvaluationManager(eval_sample_stream="eval" + self.policy_name,
                                  policy_name=self.policy_name,
                                  eval_tag="evaluation",
                                  eval_games_per_version=100,
                                  curriculum_config=Curriculum(type_=Curriculum.Type.Linear,
                                                               name="training",
                                                               stages="self-play",
                                                               conditions=[
                                                                   Condition(
                                                                       type_=Condition.Type.SimpleBound,
                                                                       args=dict(field="episode_return",
                                                                                 lower_limit=6000),
                                                                   )
                                                               ]))
            ])


class GymMujocoSampledMuzeroExperiment(Experiment):

    def __init__(self,
                 scenario: str,
                 num_sampled_actions: int,
                 reanalyze: bool = False,
                 entropy_coeff: float = 1e-2,
                 discretize: bool = False,
                 discrete_num_bins: int = 7,
                 buffer_size: int = 2000000):
        self.scenario = scenario
        self.training_steps = 200000
        self.buffer_size = buffer_size // 10
        self.sample_steps = 200

        self.rollout_update_interval = 100
        self.reanalyze_update_interval = 200

        self.do_reanalyze = reanalyze
        self.num_unroll_steps = 5
        self.td_steps = 5
        self.num_sampled_actions = num_sampled_actions
        self.entropy_coeff = entropy_coeff
        self.discretize = discretize
        self.discrete_num_bins = discrete_num_bins
        # system experiments
        self.num_aw = 32  # 32
        self.num_pw = 16  # 16
        self.num_eval_aw = 1
        self.num_eval_pw = 1
        self.num_bw = 30
        self.num_tw = 1

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(actors=TasksGroup(
            count=self.num_aw + self.num_eval_aw * 2,
            scheduling=Scheduling.actor_worker_default(container_image="marl/marl-cpu-gym_mujoco")),
                                    policies=TasksGroup(
                                        count=self.num_pw + self.num_eval_pw * 2,
                                        scheduling=Scheduling.policy_worker_default(gpu=0.12)),
                                    trainers=TasksGroup(count=self.num_tw,
                                                        scheduling=Scheduling.trainer_worker_default()),
                                    buffers=TasksGroup(count=self.num_bw,
                                                       scheduling=Scheduling.buffer_worker_default(cpu=2,
                                                                                                   gpu=0.15)),
                                    eval_managers=TasksGroup(count=3,
                                                             scheduling=Scheduling.eval_manager_default()))

    def initial_setup(self):
        policy_name = "default"

        ring_size = 32
        inference_splits = 8
        batch_size = 1024

        # rollout_sample_stream_type = "BROADCAST"
        rollout_sample_stream_type = "ROUND_ROBIN"

        # training
        if rollout_sample_stream_type == "ROUND_ROBIN":
            rollout_sample_stream = SampleStream(type_=SampleStream.Type.NAME_ROUND_ROBIN,
                                                 stream_name=f"rollout_{policy_name}")
            buffer_size = self.buffer_size // self.num_bw // self.sample_steps
        else:
            rollout_sample_stream = SampleStream(type_=SampleStream.Type.NAME_BROADCAST,
                                                 stream_name=f"rollout_{policy_name}")
            buffer_size = self.buffer_size // self.sample_steps
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)

        import os
        wandb_label = os.environ.get("WANDB_LABEL", "null")
        label = f"-a{self.rollout_update_interval}-r{self.reanalyze_update_interval}-{wandb_label}"

        wandb_args = dict(  #wandb_entity="samji2000",
            #wandb_project="gym",
            wandb_group=f"{self.scenario}" + ("-reanalyze" if self.do_reanalyze else "") +
            f"-{self.num_aw}actors" + label,
            log_wandb=True,
            wandb_config=dict(label=wandb_label, scenario=self.scenario))

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
                    inference_splits=inference_splits) for i in range(self.num_aw)
            ] + [
                ActorWorker(  # Evaluation with mcts
                    env=self.make_env(rank=i + self.num_aw, evaluation=True),
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
                    env=self.make_env(rank=i + self.num_eval_aw + self.num_aw, evaluation=True),
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
                    pull_frequency_seconds=10,
                    policy=self.make_policy(rank=i, evaluation=False, use_mcts=True),
                ) for i in range(self.num_pw)
            ] + [  # Eval policy with mcts
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=f"eval_mcts_{policy_name}",
                    parameter_db=parameter_db,
                    pull_max_failures=999,
                    pull_frequency_seconds=10,
                    policy=self.make_policy(rank=i + self.num_pw, evaluation=True, use_mcts=True),
                    policy_identifier="evaluation_mcts",
                ) for i in range(self.num_eval_pw)
            ] + [  # Eval policy without mcts
                PolicyWorker(policy_name=policy_name,
                             inference_stream=f"eval_nomcts_{policy_name}",
                             parameter_db=parameter_db,
                             pull_max_failures=999,
                             pull_frequency_seconds=10,
                             policy=self.make_policy(
                                 rank=i + self.num_eval_pw + self.num_pw, evaluation=True, use_mcts=False),
                             policy_identifier="evaluation_nomcts") for i in range(self.num_eval_pw)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='simple_queue',
                    buffer_args=dict(
                        max_size=15,
                        batch_size=batch_size // self.num_tw,
                    ),
                    policy_name=policy_name,
                    trainer=self.make_trainer(),
                    policy=self.make_policy(
                        rank=0, evaluation=False, use_mcts=False
                    ),  # evaluation & use_mcts could be arbitrary, they do not affect training
                    sample_stream=f"train_{policy_name}",
                    parameter_db=parameter_db,
                    push_tag_frequency_minutes=20,
                    worker_info=WorkerInformation(wandb_job_type="trainer_worker",
                                                  wandb_name=f"tw_{i}",
                                                  **wandb_args)) for i in range(self.num_tw)
            ],
            eval_managers=[
                EvaluationManager(eval_sample_stream=f"eval_mcts_{policy_name}",
                                  parameter_db=parameter_db,
                                  policy_name=policy_name,
                                  eval_tag="evaluation_mcts",
                                  unique_policy_version=False,
                                  eval_games_per_version=10,
                                  worker_info=WorkerInformation(wandb_job_type="em_mcts",
                                                                wandb_name=f"em_mcts",
                                                                **wandb_args)),
                EvaluationManager(eval_sample_stream=f"eval_nomcts_{policy_name}",
                                  parameter_db=parameter_db,
                                  policy_name=policy_name,
                                  eval_tag="evaluation_nomcts",
                                  unique_policy_version=False,
                                  eval_games_per_version=10,
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
                        warmup_transitions=1000,
                        batch_size=128,
                        batch_length=1 + self.num_unroll_steps + self.td_steps +
                        1,  # 1 + num_unroll_steps + td_steps + done
                        seed=i,
                    ),
                    from_sample_stream=f"rollout_{policy_name}",
                    to_sample_stream=f"train_{policy_name}",
                    unpack_batch_before_post=True,
                    policy_name=policy_name,
                    policy=self.make_policy(rank=0, evaluation=False, use_mcts=True,
                                            is_reanalyze_policy=True),
                    policy_identifier="latest",
                    parameter_db=parameter_db,
                    pull_frequency_seconds=10,
                ) for i in range(self.num_bw)
            ])

    def make_trainer(self):
        trainer = Trainer(
            type_="muzero",
            args=dict(
                num_unroll_steps=self.num_unroll_steps,
                td_steps=self.td_steps,
                # optimization
                optimizer_name="adam",
                optimizer_config=dict(lr=0., weight_decay=2e-5),
                lr_schedule=[
                    dict(name="cosine",
                         num_steps=self.training_steps,
                         config=dict(lr_init=1e-4, max_steps=self.training_steps)),
                    dict(name="linear",
                         num_steps=0.1 * self.training_steps,
                         config=dict(lr_init=0., lr_end=0.))
                ],
                max_grad_norm=100,
                # self-supervised model
                do_consistency=False,
                # loss
                reward_loss_coeff=1,
                value_loss_coeff=1,
                policy_loss_coeff=1,
                consistency_coeff=1,
                entropy_coeff=self.entropy_coeff,
            ))
        return trainer

    def make_policy(self, rank: int, evaluation: bool, use_mcts: bool, is_reanalyze_policy: bool = False):
        import numpy as np

        import gym
        from legacy.algorithm.muzero.utils.scalar_transform import DiscreteSupport
        from legacy.algorithm.muzero.utils.utils import LinearSchedule
        self.value_support = DiscreteSupport(min=-1000., max=1000., delta=10.)
        self.reward_support = DiscreteSupport(min=-10., max=10., delta=0.5)
        env = gym.make(self.scenario)
        action_space = env.action_space
        if self.discretize:
            action_space = gym.spaces.MultiDiscrete([
                self.discrete_num_bins,
            ] * env.action_space.shape[0])
        policy = Policy(
            type_=f"muzero-mlp",
            args=dict(
                action_space=action_space,
                act_dim=self.num_sampled_actions,
                obs_dim=env.observation_space.shape[0],
                discount=0.99,
                # tree search
                use_mcts=use_mcts,
                value_delta_max=0.06,
                num_simulations=50,
                root_dirichlet_alpha=0.3,
                root_exploration_fraction=0.25,
                pb_c_base=19652,
                pb_c_init=1.25,
                num_threads=16,
                # network initialization
                state_dim=256,  # 512,
                init_zero=True,
                value_fc_layers=[64, 32],
                policy_fc_layers=[64, 32],
                reward_fc_layers=[64, 32],
                num_blocks=10,  # 10,
                # reanalyze
                reanalyze_ratio_schedule=LinearSchedule(v_init=0., v_end=float(self.do_reanalyze), t_end=1),
                td_steps=self.td_steps,
                num_unroll_steps=self.num_unroll_steps,
                # value & reward transform
                value_support=self.value_support,
                reward_support=self.reward_support,
                # actor exploration
                visit_softmax_temperature_fn=None,
                warm_up_version=0 if evaluation or is_reanalyze_policy else 2000,
                # rollout & reanalyze network update
                rollout_update_interval=self.rollout_update_interval,
                reanalyze_update_interval=self.reanalyze_update_interval,
                # sampled muzero
                mcts_use_sampled_actions=True,
                mcts_num_sampled_actions=self.num_sampled_actions,
                # available actions
                use_available_action=False,
                seed=rank,
                obs_type="local"))
        return policy

    def make_env(self, rank: int, evaluation: bool = False):
        return Environment(type_="gym_mujoco",
                           args=dict(scenario=self.scenario,
                                     discretize=self.discretize,
                                     discrete_num_bins=self.discrete_num_bins))

    def visit_softmax_temperature_fn(self, version):
        return 1.0


register_experiment("gym-mujoco-mini", GymMuJoCoMiniExperiment)

scenarios = [k.split('-v')[0] for k in gym_mujoco_registry]
assert len(scenarios) == len(set(scenarios)), 'Only the latest version should be included.'

for scenario in gym_mujoco_registry:
    for m in range(1, 3):
        seed = 1
        ppo_exp_name = f"gym-mujoco-{scenario.split('-v')[0].lower()}-x{m}"
        register_experiment(
            ppo_exp_name,
            functools.partial(
                GymMuJoCoExperiment,
                scenario=scenario,
                multiplier=m,
                algo_name='mappo',
                seed=seed,
            ))

# muzero
for scenario in gym_mujoco_registry:
    args = dict(
        num_sampled_actions=((lambda x: f"s{x}"), [10, 15, 20]),
        reanalyze=((lambda x: "reanalyze" if x else None), [True, False]),
        entropy_coeff=((lambda x: f"ent{x}" if x > 0 else None), [0, 5e-3, 1e-2, 5e-2]),
        discretize=((lambda x: "disc" if x else None), [True, False]),
        buffer_size=((lambda x: f"{x // 1000000}M-buf"), [1000000, 2000000, 3000000, 4000000, 5000000]),
    )
    arg_names = args.keys()
    arg_ranges = [v[1] for k, v in args.items()]
    for arg_values in itertools.product(*arg_ranges):
        exp_name = [f"gym-mujoco-{scenario.split('-v')[0]}-muzero"]
        for arg_name, arg_v in zip(arg_names, arg_values):
            display_str = args[arg_name][0](arg_v)
            if display_str is not None:
                exp_name.append(display_str)
        exp_name = "-".join(exp_name)
        exp_setup = {arg_name: arg_v for arg_name, arg_v in zip(arg_names, arg_values)}
        register_experiment(
            exp_name, functools.partial(
                GymMujocoSampledMuzeroExperiment,
                scenario=scenario,
                **exp_setup,
            ))
