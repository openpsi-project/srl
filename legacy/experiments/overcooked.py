from api.config import *


class OvercookedMiniExperiment(Experiment):

    def __init__(self):
        self.num_actors = 4
        self.num_policies = 1
        self.num_trainers = 1

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(self.num_actors,
                              Scheduling.actor_worker_default(container_image="marl/marl-cpu-overcooked")),
            policies=TasksGroup(self.num_policies, Scheduling.policy_worker_default()),
            trainers=TasksGroup(self.num_trainers, Scheduling.trainer_worker_default()))

    def initial_setup(self):
        layout_name = 'five_by_five'
        policy_name = "default"
        inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)
        policy = Policy(type_="overcooked-separate")
        ppo_iterations = 32
        ppo_epochs = 1
        ppg_epochs = 6

        trainer = Trainer(type_='mappo',
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
                              ppo_epochs=ppo_epochs,
                              ppo_iterations=ppo_iterations,
                              ppg_epochs=ppg_epochs,
                              local_cache_stack_size=None,
                              popart=True,
                              max_grad_norm=10.0,
                              log_per_steps=1,
                              decay_per_steps=1000,
                              entropy_bonus_decay=1,
                              bootstrap_steps=10,
                          ))
        return ExperimentConfig(
            actors=[
                ActorWorker(inference_streams=[inference_stream],
                            sample_streams=[sample_stream],
                            env=Environment(type_="overcooked",
                                            args=dict(render=False, layout_name=layout_name)),
                            agent_specs=[
                                AgentSpec(
                                    index_regex=".*",
                                    inference_stream_idx=0,
                                    sample_stream_idx=0,
                                    bootstrap_steps=10,
                                )
                            ],
                            ring_size=1) for _ in range(self.num_actors)
            ],
            policies=[
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=inference_stream,
                    parameter_db=parameter_db,
                    policy=policy,
                    pull_max_failures=1000,
                    pull_frequency_seconds=0.5,
                ) for _ in range(self.num_policies)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(
                        max_size=64,
                        reuses=10,
                        batch_size=64,
                    ),
                    policy_name=policy_name,
                    trainer=trainer,
                    policy=policy,
                    sample_stream=sample_stream,
                    parameter_db=parameter_db,
                ) for _ in range(self.num_trainers)
            ],
        )


register_experiment("overcooked-mini", OvercookedMiniExperiment)


class OvercookedExperiment(Experiment):

    def __init__(self,
                 ppo_epochs=1,
                 ppo_iterations=32,
                 ppg_epochs: int = 6,
                 multiplier: int = 1,
                 layout_name: str = 'coordination_ring',
                 seed: int = 0):
        self.ppg_epochs = ppg_epochs
        self.seed = seed
        self.policy_name = policy_name = "default"
        self.num_actors = 160 * multiplier
        self.num_policies = 20 * multiplier
        self.num_trainer = 6 * multiplier
        self.num_eval_actors = 40
        self.num_eval_pw = 1
        self.ring_size = 1
        self.inference_splits = 3
        self.layout_name = layout_name

        self.inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        self.sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
        self.eval_inference_stream = InferenceStream(type_=InferenceStream.Type.NAME,
                                                     stream_name=f"eval_{policy_name}")
        self.eval_sample_stream = SampleStream(type_=SampleStream.Type.NAME,
                                               stream_name=f"eval_{policy_name}")
        self.parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)

        self.algo_name = algo_name = 'mappo'

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
                                   bootstrap_steps=200,
                               ))
        self.policy = Policy(type_="overcooked-separate",
                             args=dict(num_rnn_layers=1, chuck_len=10, rnn_type="lstm", seed=seed))

    def train_env(self):
        return Environment(type_="overcooked", args=dict(layout_name=self.layout_name))

    def eval_env(self, env_idx):
        return Environment(type_="overcooked",
                           args=dict(layout_name=self.layout_name,
                                     render=True,
                                     render_interval=100,
                                     env_idx=env_idx))

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(actors=TasksGroup(
            self.num_actors + self.num_eval_actors,
            Scheduling.actor_worker_default(container_image="marl/marl-cpu-overcooked")),
                                    policies=TasksGroup(self.num_policies + self.num_eval_pw,
                                                        Scheduling.policy_worker_default(gpu=0.12)),
                                    trainers=TasksGroup(self.num_trainer,
                                                        Scheduling.trainer_worker_default()),
                                    eval_managers=TasksGroup(1, Scheduling.eval_manager_default()))

    def initial_setup(self):
        return ExperimentConfig(
            actors=[
                ActorWorker(  # Training.
                    env=self.train_env(),
                    inference_streams=[self.inference_stream],
                    sample_streams=[self.sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            sample_steps=600,
                            bootstrap_steps=200,
                            send_after_done=False,
                            send_full_trajectory=False,
                        )
                    ],
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                ) for i in range(self.num_actors)
            ] + [
                ActorWorker(  # Evaluation
                    env=self.eval_env(env_idx=i),
                    inference_streams=[self.eval_inference_stream],
                    sample_streams=[self.eval_sample_stream,
                                    SampleStream(type_=SampleStream.Type.NULL)],
                    agent_specs=[
                        AgentSpec(
                            index_regex="0",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=True,
                            deterministic_action=True,
                            send_concise_info=True,
                        ),
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=1,
                            send_concise_info=True,
                        ),
                    ],
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                ) for i in range(self.num_eval_actors)
            ],
            policies=[
                PolicyWorker(
                    policy_name=self.policy_name,
                    inference_stream=self.inference_stream,
                    parameter_db=self.parameter_db,
                    policy=self.policy,
                ) for _ in range(self.num_policies)
            ] + [
                PolicyWorker(
                    policy_name=self.policy_name,
                    inference_stream=self.eval_inference_stream,
                    parameter_db=self.parameter_db,
                    policy=self.policy,
                    policy_identifier="evaluation",
                ) for _ in range(self.num_eval_pw)
            ],
            trainers=[
                TrainerWorker(buffer_name='priority_queue',
                              buffer_args=dict(
                                  max_size=64,
                                  reuses=1 if self.algo_name == "mappg" else 10,
                                  batch_size=320,
                              ),
                              policy_name=self.policy_name,
                              trainer=self.trainer,
                              policy=self.policy,
                              sample_stream=self.sample_stream,
                              parameter_db=self.parameter_db,
                              push_tag_frequency_minutes=20) for _ in range(self.num_trainer)
            ],
            eval_managers=[
                EvaluationManager(
                    eval_sample_stream=self.eval_sample_stream,
                    parameter_db=self.parameter_db,
                    policy_name=self.policy_name,
                    eval_tag="evaluation",
                    eval_games_per_version=100,
                )
            ])


register_experiment("overcooked", OvercookedExperiment)
