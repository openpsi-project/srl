import functools
import numpy as np

map_agent_registry = {
    # env_name: (left, right, game_length)
    "11_vs_11_competition": (10, 10, 3000),
    "11_vs_11_easy_stochastic": (10, 10, 3000),
    "11_vs_11_hard_stochastic": (10, 10, 3000),
    "11_vs_11_kaggle": (10, 10, 3000),
    "11_vs_11_stochastic": (10, 10, 3000),
    "1_vs_1_easy": (1, 1, 500),
    "5_vs_5": (4, 4, 3000),
    "academy_pass_and_shoot_with_keeper": (2, 1, 400),
    "academy_run_pass_and_shoot_with_keeper": (2, 1, 400),
    "academy_3_vs_1_with_keeper": (3, 1, 400),
    "academy_counterattack_easy": (4, 1, 400),
    "academy_counterattack_hard": (4, 2, 400),
    "academy_corner": (10, 10, 400),
}


class FootballVanillaPBTExperiment(Experiment):

    def __init__(self,
                 pop_size: int,
                 env_name: str = "academy_3_vs_1_with_keeper",
                 representation: str = "simple115v2",
                 algo_name: str = "mappo",
                 ppo_epochs=1,
                 ppo_iterations=32,
                 ppg_epochs: int = 6,
                 seed: int = 0):
        self.env_name = env_name
        self.num_players = map_agent_registry[env_name][0]
        assert representation == "simple115v2"
        self.representation = representation
        self.algo_name = algo_name
        self.ppo_epochs = ppo_epochs
        self.ppo_iterations = ppo_iterations
        self.ppg_epochs = ppg_epochs
        self.seed = seed
        self.num_actors = 120
        self.num_policies = max(self.num_players, 5) // 5 * 4
        self.num_trainers = max(self.num_players, 3) // 3 * 1
        self.num_eval_actors = 40
        self.num_eval_policies = 1
        self.ring_size = 3
        self.inference_splits = 3

        # Vanilla PBT configurations.
        self.population = [f"policy_{i}" for i in range(pop_size)]
        self.initial_hyperparams = [dict(lr=lr) for lr in 10**(-np.linspace(2, 5, pop_size))]
        self.population_algorithm = PopulationAlgorithm(
            type_="vanilla_pbt",
            args=dict(
                population=self.population,
                ready_interval=1000,
                truncation_ratio=0.2,
                explore_configs=[
                    dict(
                        keys=["optimizer_config", "lr"],
                        method="perturb",
                        factors=[0.8, 1.2],
                        min_value=0,
                        max_value=1,
                    ),
                ],
            ),
        )

        self.policy = Policy(
            type_=f"football-simple115-auxiliary" if algo_name == "mappg" else "football-simple115-separate",
            args=dict(num_rnn_layers=1, chunk_len=10, rnn_type="lstm", seed=seed))
        self.population_sample_stream = SampleStream(type_=SampleStream.Type.NAME,
                                                     stream_name="population_stream")
        self.null_sample_stream = SampleStream(type_=SampleStream.Type.NULL)

    def train_env(self, env_seed):
        return Environment(type_="football",
                           args=dict(env_name=self.env_name,
                                     number_of_left_players_agent_controls=self.num_players,
                                     number_of_right_players_agent_controls=0,
                                     representation=self.representation,
                                     rewards="scoring,checkpoints",
                                     seed=env_seed))

    def eval_env(self, env_seed):
        return Environment(type_="football",
                           args=dict(env_name=self.env_name,
                                     number_of_left_players_agent_controls=self.num_players,
                                     number_of_right_players_agent_controls=0,
                                     representation=self.representation,
                                     seed=env_seed))

    def make_trainer(self, lr):
        return Trainer(type_=self.algo_name,
                       args=dict(discount_rate=0.99,
                                 gae_lambda=0.95,
                                 vtrace=False,
                                 clip_value=True,
                                 eps_clip=0.2,
                                 value_loss='huber',
                                 value_loss_weight=1,
                                 value_loss_config=dict(delta=10.0,),
                                 entropy_bonus_weight=0.01,
                                 optimizer='adam',
                                 optimizer_config=dict(lr=lr),
                                 ppg_optimizer="adam",
                                 ppg_optimizer_config=dict(lr=5e-4),
                                 beta_clone=1,
                                 ppo_epochs=1 if self.algo_name == "mappo" else self.ppo_epochs,
                                 ppo_iterations=self.ppo_iterations,
                                 ppg_epochs=self.ppg_epochs,
                                 local_cache_stack_size=None,
                                 popart=True,
                                 max_grad_norm=10.0,
                                 log_per_steps=1,
                                 decay_per_steps=1000,
                                 entropy_bonus_decay=1,
                                 bootstrap_steps=200))

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(
                count=(self.num_actors + self.num_eval_actors) * len(self.population),
                scheduling=Scheduling.actor_worker_default(container_image="marl/marl-cpu-football")),
            policies=TasksGroup(count=(self.num_policies + self.num_eval_policies) * len(self.population),
                                scheduling=Scheduling.policy_worker_default(gpu=0.12)),
            trainers=TasksGroup(count=self.num_trainers * len(self.population),
                                scheduling=Scheduling.trainer_worker_default()),
            eval_managers=TasksGroup(count=len(self.population),
                                     scheduling=Scheduling.eval_manager_default()),
            population_manager=TasksGroup(count=1, scheduling=Scheduling.population_manager_default()))

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
                    env=self.train_env(env_seed=12023 + i),
                    inference_streams=[inference_stream],
                    sample_streams=[sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            sample_steps=600,
                            bootstrap_steps=200,
                            send_after_done=False,
                            send_full_trajectory=False,
                        ),
                    ],
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                ) for i in range(self.num_actors)
            ] + [  # Evaluation.
                ActorWorker(
                    env=self.eval_env(env_seed=12023 + i),
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
                            send_concise_info=True,
                        ),
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=2,
                            send_full_trajectory=True,
                            send_concise_info=True,
                            deterministic_action=True,
                        ),
                    ],
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                ) for i in range(self.num_eval_actors)
            ])

            policies.extend([  # Training.
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=inference_stream,
                    policy=self.policy,
                    pull_max_failures=1000,
                    pull_frequency_seconds=4,
                ) for j in range(self.num_policies)
            ] + [  # Evaluation.
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=eval_inference_stream,
                    policy=self.policy,
                    pull_max_failures=1000,
                    pull_frequency_seconds=4,
                    policy_identifier="evaluation",
                ) for j in range(self.num_eval_policies)
            ])

            trainers.extend([
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(
                        max_size=64,
                        reuses=1 if self.algo_name == "mappg" else 10,
                        batch_size=80,
                    ),
                    policy_name=policy_name,
                    trainer=trainer,
                    policy=self.policy,
                    sample_stream=sample_stream,
                    push_tag_frequency_minutes=20,
                ) for k in range(self.num_trainers)
            ])

            eval_managers.extend([
                EvaluationManager(
                    eval_sample_stream=eval_sample_stream,
                    policy_name=policy_name,
                    eval_tag="evaluation",
                    eval_games_per_version=100,
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
            )
        ]

        return ExperimentConfig(actors=actors,
                                policies=policies,
                                trainers=trainers,
                                eval_managers=eval_managers,
                                population_manager=population_manager)


class FootballPSROSymmetricExperiment(Experiment):

    def __init__(self,
                 env_name: str = "5_vs_5",
                 representation: str = "simple115v2",
                 algo_name: str = "mappo",
                 ppo_epochs=32,
                 ppo_iterations=1,
                 ppg_epochs: int = 6,
                 seed: int = 0):
        self.env_name = env_name
        self.left_players = map_agent_registry[env_name][0]
        self.right_players = map_agent_registry[env_name][1]
        assert representation == "simple115v2"
        self.representation = representation
        self.algo_name = algo_name
        self.ppg_epochs = ppg_epochs
        self.seed = seed
        self.num_actors = 320
        self.num_policies = max(self.left_players, 5) // 5 * 4
        self.num_trainers = max(self.left_players, 3) // 3 * 1
        self.num_eval_actors = 40
        self.num_eval_policies = 1
        self.ring_size = 1
        self.inference_splits = 1

        # PSRO configurations.
        self.index_regex = [
            "|".join([str(i) for i in range(self.left_players)]),
            "|".join([str(i) for i in range(self.left_players, self.left_players + self.right_players)]),
        ]
        self.eval_index_regex = ["0", f"{self.left_players}"]
        # Set initial population.
        self.pop_size = 1
        self.population = [[f"policy_{i}" for i in range(self.pop_size)]]
        self.training_policy_names = [f"policy_{self.pop_size}"]
        # Set initial meta-strategies.
        self.policy_sample_probs = np.ones(self.pop_size) / self.pop_size
        self.eval_policy_sample_probs = np.ones(self.pop_size + 1) / (self.pop_size + 1)
        # Set initial payoffs.
        self.initial_payoffs = np.zeros((2, self.pop_size, self.pop_size))
        # Set stopping conditions: version >= 100000 or converged.
        self.conditions = [[
            Condition(type_=Condition.Type.SimpleBound, args=dict(field="version", lower_limit=100000)),
            Condition(type_=Condition.Type.Converged,
                      args=dict(value_field="mixed_payoff",
                                step_field="version",
                                warmup_step=1000,
                                duration=5000,
                                confidence=0.9,
                                threshold=0.1))
        ]]
        self.population_algorithm = PopulationAlgorithm(
            type_="psro",
            args=dict(
                # Set meta-solver.
                meta_solver=MetaSolver(type_=MetaSolver.Type.UNIFORM),
                population=self.population,
                training_policy_names=self.training_policy_names,
                initial_payoffs=self.initial_payoffs,
                conditions=self.conditions,
                num_iterations=10,
                symmetric=True,
            ),
        )

        self.policy = Policy(
            type_=f"football-simple115-auxiliary" if algo_name == "mappg" else "football-simple115-separate",
            args=dict(num_rnn_layers=1, chunk_len=10, rnn_type="lstm", seed=seed))
        self.inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name="inf_stream")
        self.opponent_inference_stream = InferenceStream(
            type_=InferenceStream.Type.INLINE,
            stream_name="oppo_inf_stream",
            policy=self.policy,
            population=self.population[0],
            policy_sample_probs=self.policy_sample_probs,
            policy_identifier="latest",
        )
        self.sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name="sample_stream")
        self.eval_inference_stream = InferenceStream(type_=InferenceStream.Type.NAME,
                                                     stream_name="eval_inf_stream")
        self.eval_opponent_inference_stream = InferenceStream(
            type_=InferenceStream.Type.INLINE,
            stream_name="eval_oppo_inf_stream",
            policy=self.policy,
            population=self.population[0] + [self.training_policy_names[0]],
            policy_sample_probs=self.eval_policy_sample_probs,
            policy_identifier="evaluation",
        )
        self.eval_sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name="eval_sample_stream")
        self.ma_sample_stream = SampleStream(type_=SampleStream.Type.NAME_MULTI_AGENT,
                                             stream_name="population_sample_stream")
        self.population_sample_stream = SampleStream(type_=SampleStream.Type.NAME,
                                                     stream_name="population_sample_stream")
        self.null_sample_stream = SampleStream(type_=SampleStream.Type.NULL)
        self.trainer = Trainer(type_=algo_name,
                               args=dict(discount_rate=0.99,
                                         gae_lambda=0.95,
                                         vtrace=False,
                                         clip_value=True,
                                         eps_clip=0.2,
                                         dual_clip=True,
                                         c_clip=3,
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
                                         ppg_epochs=ppg_epochs,
                                         local_cache_stack_size=None,
                                         popart=True,
                                         max_grad_norm=10.0,
                                         log_per_steps=1,
                                         decay_per_steps=1000,
                                         entropy_bonus_decay=1,
                                         bootstrap_steps=200))

    def train_env(self, env_seed):
        return Environment(type_="football",
                           args=dict(env_name=self.env_name,
                                     number_of_left_players_agent_controls=self.left_players,
                                     number_of_right_players_agent_controls=self.right_players,
                                     representation=self.representation,
                                     rewards="scoring,checkpoints",
                                     share_reward=True,
                                     seed=env_seed))

    def eval_env(self, env_seed):
        return Environment(type_="football",
                           args=dict(env_name=self.env_name,
                                     number_of_left_players_agent_controls=self.left_players,
                                     number_of_right_players_agent_controls=self.right_players,
                                     representation=self.representation,
                                     seed=env_seed))

    def initial_setup(self):
        actors = [  # Training.
            ActorWorker(
                env=self.train_env(env_seed=12023 + i),
                inference_streams=[self.inference_stream, self.opponent_inference_stream],
                sample_streams=[self.sample_stream, self.null_sample_stream],
                agent_specs=[
                    AgentSpec(  # Left players.
                        index_regex=self.index_regex[0],
                        inference_stream_idx=0,
                        sample_stream_idx=0,
                        sample_steps=600,
                        bootstrap_steps=200,
                        send_after_done=False,
                        send_full_trajectory=False,
                    ),
                    AgentSpec(  # Right players
                        index_regex=self.index_regex[1],
                        inference_stream_idx=1,
                        sample_stream_idx=1,
                        sample_steps=600,
                        bootstrap_steps=200,
                        send_after_done=False,
                        send_full_trajectory=False,
                    ),
                ],
                ring_size=self.ring_size,
                inference_splits=self.inference_splits,
                worker_info=WorkerInformation(worker_tag="training"),
            ) for i in range(self.num_actors)
        ] + [  # Evaluation.
            ActorWorker(
                env=self.eval_env(env_seed=12023 + i),
                inference_streams=[self.eval_inference_stream, self.eval_opponent_inference_stream],
                sample_streams=[self.eval_sample_stream, self.ma_sample_stream, self.null_sample_stream],
                agent_specs=[
                    AgentSpec(
                        index_regex=self.eval_index_regex[0],
                        inference_stream_idx=0,
                        sample_stream_idx=[0, 1],
                        send_full_trajectory=True,
                        deterministic_action=False,
                    ),
                    AgentSpec(
                        index_regex=self.index_regex[0],
                        inference_stream_idx=0,
                        sample_stream_idx=2,
                        send_full_trajectory=True,
                        deterministic_action=False,
                    ),
                    AgentSpec(
                        index_regex=self.eval_index_regex[1],
                        inference_stream_idx=1,
                        sample_stream_idx=1,
                        send_full_trajectory=True,
                        deterministic_action=False,
                    ),
                    AgentSpec(
                        index_regex=self.index_regex[1],
                        inference_stream_idx=1,
                        sample_stream_idx=2,
                        send_full_trajectory=True,
                        deterministic_action=False,
                    ),
                ],
                ring_size=self.ring_size,
                inference_splits=self.inference_splits,
                worker_info=WorkerInformation(worker_tag="evaluation"),
            ) for i in range(self.num_eval_actors)
        ]

        policies = [  # Training.
            PolicyWorker(
                policy_name=self.training_policy_names[0],
                inference_stream=self.inference_stream,
                policy=self.policy,
                pull_max_failures=1000,
                pull_frequency_seconds=0.5,
                scheduling=Scheduling.policy_worker_default(gpu=0.2),
            ) for j in range(self.num_policies)
        ] + [  # Evaluation.
            PolicyWorker(
                policy_name=self.training_policy_names[0],
                inference_stream=self.eval_inference_stream,
                policy=self.policy,
                pull_max_failures=1000,
                pull_frequency_seconds=2,
                policy_identifier="evaluation",
                scheduling=Scheduling.policy_worker_default(gpu=0.2),
            ) for j in range(self.num_eval_policies)
        ]

        trainers = [
            TrainerWorker(
                buffer_name='priority_queue',
                buffer_args=dict(
                    max_size=64,
                    reuses=1 if self.algo_name == "mappg" else 10,
                    batch_size=320,
                ),
                policy_name=self.training_policy_names[0],
                trainer=self.trainer,
                policy=self.policy,
                sample_stream=self.sample_stream,
                push_tag_frequency_minutes=60,
                scheduling=Scheduling.trainer_worker_default(node_type="g8"),
            ) for k in range(self.num_trainers)
        ]

        eval_managers = [
            EvaluationManager(
                eval_sample_stream=self.eval_sample_stream,
                policy_name=self.training_policy_names[0],
                eval_tag="evaluation",
                eval_games_per_version=25 * (self.pop_size + 1),
            )
        ]

        population_manager = [
            PopulationManager(
                population=self.population,
                population_algorithm=self.population_algorithm,
                population_sample_stream=self.population_sample_stream,
                actors=actors,
                policies=policies,
                trainers=trainers,
                eval_managers=eval_managers,
            )
        ]

        return ExperimentSetup(actors=actors,
                               policies=policies,
                               trainers=trainers,
                               eval_managers=eval_managers,
                               population_manager=population_manager)


class FootballPSROAsymmetricExperiment(Experiment):

    def __init__(self,
                 env_name: str = "academy_3_vs_1_with_keeper",
                 representation: str = "simple115v2",
                 algo_name: str = "mappo",
                 ppo_epochs=32,
                 ppo_iterations=1,
                 ppg_epochs: int = 6,
                 seed: int = 0):
        self.env_name = env_name
        self.left_players = map_agent_registry[env_name][0]
        self.right_players = map_agent_registry[env_name][1]
        assert representation == "simple115v2"
        self.representation = representation
        self.algo_name = algo_name
        self.ppg_epochs = ppg_epochs
        self.seed = seed
        self.num_actors = 320
        self.num_policies = 6
        self.num_trainers = 1
        self.num_eval_actors = 40
        self.num_eval_policies = 1
        self.ring_size = 1
        self.inference_splits = 1

        # PSRO configurations.
        self.num_players = 2
        self.index_regex = [
            "|".join([str(i) for i in range(self.left_players)]),
            "|".join([str(i) for i in range(self.left_players, self.left_players + self.right_players)]),
        ]
        self.eval_index_regex = ["0", f"{self.left_players}"]
        # Set initial population.
        self.pop_sizes = np.array([1, 1])
        self.population = []
        self.training_policy_names = []
        self.policy_sample_probs = []
        self.eval_policy_sample_probs = []
        for idx in range(self.num_players):
            pop_size = self.pop_sizes[idx]
            self.population.append([f"player{idx}_{j}" for j in range(pop_size)])
            self.training_policy_names.append(f"player{idx}_{pop_size}")
            # Set initial meta-strategies
            self.policy_sample_probs.append(np.ones(pop_size) / pop_size)
            self.eval_policy_sample_probs.append(np.ones(pop_size + 1) / (pop_size + 1))
        # Set initial payoffs
        self.initial_payoffs = np.zeros((self.num_players, *self.pop_sizes))
        # Set stopping conditions for each player.
        self.conditions = [
            [  # Conditions for player 0: version >= 30000 or converged.
                Condition(type_=Condition.Type.SimpleBound, args=dict(field="version", lower_limit=30000)),
                Condition(type_=Condition.Type.Converged,
                          args=dict(value_field="mixed_payoff",
                                    step_field="version",
                                    warmup_step=10000,
                                    duration=5000,
                                    confidence=0.9,
                                    threshold=0.1))
            ],
            [  # Conditions for player 1: version >= 10000 or converged.
                Condition(type_=Condition.Type.SimpleBound, args=dict(field="version", lower_limit=10000)),
                Condition(type_=Condition.Type.Converged,
                          args=dict(value_field="mixed_payoff",
                                    step_field="version",
                                    warmup_step=3000,
                                    duration=2000,
                                    confidence=0.9,
                                    threshold=0.1))
            ],
        ]
        self.population_algorithm = PopulationAlgorithm(
            type_="psro",
            args=dict(
                # Set meta-solver
                meta_solver=MetaSolver(type_=MetaSolver.Type.UNIFORM),
                population=self.population,
                training_policy_names=self.training_policy_names,
                initial_payoffs=self.initial_payoffs,
                conditions=self.conditions,
                num_iterations=10,
                symmetric=False,
            ),
        )

        self.policy = Policy(
            type_=f"football-simple115-auxiliary" if algo_name == "mappg" else "football-simple115-separate",
            args=dict(num_rnn_layers=1, chunk_len=10, rnn_type="lstm", seed=seed))
        self.ma_sample_stream = SampleStream(type_=SampleStream.Type.NAME_MULTI_AGENT,
                                             stream_name="population_sample_stream")
        self.population_sample_stream = SampleStream(type_=SampleStream.Type.NAME,
                                                     stream_name="population_sample_stream")
        self.null_sample_stream = SampleStream(type_=SampleStream.Type.NULL)
        self.trainer = Trainer(type_=algo_name,
                               args=dict(discount_rate=0.99,
                                         gae_lambda=0.95,
                                         vtrace=False,
                                         clip_value=True,
                                         eps_clip=0.2,
                                         dual_clip=True,
                                         c_clip=3,
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
                                         ppg_epochs=ppg_epochs,
                                         local_cache_stack_size=None,
                                         popart=True,
                                         max_grad_norm=10.0,
                                         log_per_steps=1,
                                         decay_per_steps=1000,
                                         entropy_bonus_decay=1,
                                         bootstrap_steps=200))

    def train_env(self, env_seed):
        return Environment(type_="football",
                           args=dict(env_name=self.env_name,
                                     number_of_left_players_agent_controls=self.left_players,
                                     number_of_right_players_agent_controls=self.right_players,
                                     representation=self.representation,
                                     rewards="scoring,checkpoints",
                                     share_reward=True,
                                     seed=env_seed))

    def eval_env(self, env_seed):
        return Environment(type_="football",
                           args=dict(env_name=self.env_name,
                                     number_of_left_players_agent_controls=self.left_players,
                                     number_of_right_players_agent_controls=self.right_players,
                                     representation=self.representation,
                                     seed=env_seed))

    def inference_streams(self, player_idx):
        inference_streams = [None] * self.num_players
        for idx in range(self.num_players):
            if idx == player_idx:
                inference_streams[idx] = InferenceStream(type_=InferenceStream.Type.NAME,
                                                         stream_name=f"{player_idx}-inf_stream")
            else:
                inference_streams[idx] = InferenceStream(
                    type_=InferenceStream.Type.INLINE,
                    stream_name=f"{player_idx}_{idx}-inf_stream",
                    policy=self.policy,
                    population=self.population[idx],
                    policy_sample_probs=self.policy_sample_probs[idx],
                    policy_identifier="latest",
                )
        return inference_streams

    def eval_inference_streams(self, player_idx):
        eval_inference_streams = [None] * self.num_players
        for idx in range(self.num_players):
            if idx == player_idx:
                eval_inference_streams[idx] = InferenceStream(type_=InferenceStream.Type.NAME,
                                                              stream_name=f"{player_idx}-eval_inf_stream")
            else:
                eval_inference_streams[idx] = InferenceStream(
                    type_=InferenceStream.Type.INLINE,
                    stream_name=f"{player_idx}_{idx}-eval_inf_stream",
                    policy=self.policy,
                    population=self.population[idx] + [self.training_policy_names[idx]],
                    policy_sample_probs=self.eval_policy_sample_probs[idx],
                    policy_identifier="evaluation",
                )
        return eval_inference_streams

    def initial_setup(self):
        actors = []
        policies = []
        trainers = []
        eval_managers = []

        for player_idx in range(self.num_players):
            pop_size = self.pop_sizes[player_idx]
            policy_name = self.training_policy_names[player_idx]
            inference_streams = self.inference_streams(player_idx)
            sample_stream = SampleStream(type_=SampleStream.Type.NAME,
                                         stream_name=f"{player_idx}-sample_stream")
            eval_inference_streams = self.eval_inference_streams(player_idx)
            eval_sample_stream = SampleStream(type_=SampleStream.Type.NAMED_EVAL,
                                              stream_name=f"{player_idx}-eval_sample_stream")

            actors.extend([  # Training.
                ActorWorker(
                    env=self.train_env(env_seed=12023 + i),
                    inference_streams=inference_streams,
                    sample_streams=[sample_stream, self.null_sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=self.index_regex[idx],
                            inference_stream_idx=idx,
                            sample_stream_idx=0 if idx == player_idx else 1,
                            sample_steps=600,
                            bootstrap_steps=200,
                            send_after_done=False,
                            send_full_trajectory=False,
                        ) for idx in range(self.num_players)
                    ],
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                    worker_info=WorkerInformation(population_index=player_idx, worker_tag="training"),
                ) for i in range(self.num_actors)
            ] + [  # Evaluation.
                ActorWorker(
                    env=self.eval_env(env_seed=12023 + i),
                    inference_streams=eval_inference_streams,
                    sample_streams=[eval_sample_stream, self.ma_sample_stream, self.null_sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=self.eval_index_regex[idx],
                            inference_stream_idx=idx,
                            sample_stream_idx=[0, 1] if idx == player_idx else 1,
                            send_full_trajectory=True,
                            deterministic_action=False,
                        ) for idx in range(self.num_players)
                    ] + [
                        AgentSpec(
                            index_regex=self.index_regex[idx],
                            inference_stream_idx=idx,
                            sample_stream_idx=2,
                            send_full_trajectory=True,
                            deterministic_action=False,
                        ) for idx in range(self.num_players)
                    ],
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                    worker_info=WorkerInformation(population_index=player_idx, worker_tag="evaluation"),
                ) for i in range(self.num_eval_actors)
            ])

            policies.extend([  # Training.
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=inference_streams[player_idx],
                    policy=self.policy,
                    pull_max_failures=1000,
                    pull_frequency_seconds=0.5,
                    scheduling=Scheduling.policy_worker_default(gpu=0.2),
                    worker_info=WorkerInformation(population_index=player_idx),
                ) for j in range(self.num_policies)
            ] + [  # Evaluation.
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=eval_inference_streams[player_idx],
                    policy=self.policy,
                    pull_max_failures=1000,
                    pull_frequency_seconds=2,
                    policy_identifier="evaluation",
                    scheduling=Scheduling.policy_worker_default(gpu=0.2),
                    worker_info=WorkerInformation(population_index=player_idx),
                ) for j in range(self.num_eval_policies)
            ])

            trainers.extend([
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(
                        max_size=64,
                        reuses=1 if self.algo_name == "mappg" else 10,
                        batch_size=320,
                    ),
                    policy_name=policy_name,
                    trainer=self.trainer,
                    policy=self.policy,
                    sample_stream=sample_stream,
                    push_tag_frequency_minutes=30,
                    scheduling=Scheduling.trainer_worker_default(node_type="g8"),
                    worker_info=WorkerInformation(population_index=player_idx),
                ) for k in range(self.num_trainers)
            ])

            eval_managers.extend([
                EvaluationManager(
                    eval_sample_stream=eval_sample_stream,
                    policy_name=policy_name,
                    eval_tag="evaluation",
                    eval_games_per_version=100 * np.prod(self.pop_sizes + 1) // (pop_size + 1),
                    worker_info=WorkerInformation(population_index=player_idx),
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
            )
        ]

        return ExperimentSetup(actors=actors,
                               policies=policies,
                               trainers=trainers,
                               eval_managers=eval_managers,
                               population_manager=population_manager)


for env_name in map_agent_registry.keys():
    for pop_size in [5, 10, 20, 40]:
        register_experiment(
            f"fb-pbt-{env_name.replace('_', '-')}-pop{pop_size}",
            functools.partial(
                FootballVanillaPBTExperiment,
                env_name=env_name,
                pop_size=pop_size,
                representation="simple115v2",
                algo_name="mappo",
            ),
        )

    if "academy" in env_name:  # Asymmetric PSRO.
        register_experiment(
            f"fb-psro-asymmetric-{env_name.replace('_', '-')}",
            functools.partial(
                FootballPSROAsymmetricExperiment,
                env_name=env_name,
                representation="simple115v2",
                algo_name="mappo",
            ),
        )

    else:  # Symmetric PSRO.
        register_experiment(
            f"fb-psro-symmetric-{env_name.replace('_', '-')}",
            functools.partial(
                FootballPSROSymmetricExperiment,
                env_name=env_name,
                representation="simple115v2",
                algo_name="mappo",
            ),
        )
