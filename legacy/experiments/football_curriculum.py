from api.config import *


class Football3v2Curriculum(Experiment):

    def __init__(self,
                 multiplier: int = 1,
                 env_name: str = "3v2_easy",
                 left_players: int = 3,
                 right_players: int = 0,
                 representation: str = "simple115v2",
                 algo_name: str = "mappo",
                 seed: int = 0):
        self.env_name = env_name
        self.left_players = left_players
        self.right_players = right_players
        self.representation = representation
        self.algo_name = algo_name
        self.seed = seed
        self.policy_name = policy_name = "default"
        self.actors = 200 * multiplier
        self.main_policies = 14 * multiplier
        self.num_trainer = 1 * multiplier
        self.num_eval_actors = 80
        self.num_eval_pw = 2
        self.ring_size = 3
        self.inference_splits = 3
        self.policy = Policy("football-simple115-separate",
                             args=dict(num_rnn_layers=1, chunk_len=10, rnn_type="lstm", seed=seed))

        self.inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        self.eval_inf_stream = InferenceStream(type_=InferenceStream.Type.NAME,
                                               stream_name=f"eval_{policy_name}")

        self.sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
        self.eval_sample_stream = SampleStream(type_=SampleStream.Type.NAME,
                                               stream_name=f"eval_{policy_name}")

        self.trainer = Trainer(type_=algo_name,
                               args=dict(
                                   discount_rate=0.999,
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
                                   popart=True,
                                   max_grad_norm=10.0,
                                   log_per_steps=1,
                                   decay_per_steps=1000,
                                   entropy_bonus_decay=1,
                                   bootstrap_steps=100,
                               ))

    def train_env(self, env_seed):
        return Environment(type_="football",
                           args=dict(
                               env_name=self.env_name,
                               number_of_left_players_agent_controls=self.left_players,
                               number_of_right_players_agent_controls=self.right_players,
                               representation=self.representation,
                               seed=env_seed,
                           ))

    def eval_env(self, env_seed, write_video=False):
        return Environment(type_="football",
                           args=dict(
                               env_name=self.env_name,
                               number_of_left_players_agent_controls=self.left_players,
                               number_of_right_players_agent_controls=0,
                               representation=self.representation,
                               seed=env_seed,
                               write_full_episode_dumps=write_video,
                               write_video=write_video,
                               dump_frequency=2000,
                           ))

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(actors=TasksGroup(
            count=self.actors + self.num_eval_actors,
            scheduling=Scheduling.actor_worker_default(container_image="marl/marl-cpu-football")),
                                    policies=TasksGroup(
                                        count=self.main_policies + self.num_eval_pw,
                                        scheduling=Scheduling.policy_worker_default(gpu=0.12)),
                                    trainers=TasksGroup(count=self.num_trainer,
                                                        scheduling=Scheduling.trainer_worker_default()),
                                    eval_managers=TasksGroup(count=1,
                                                             scheduling=Scheduling.eval_manager_default()))

    def initial_setup(self):
        sample_scheme = dict(
            sample_steps=800,
            bootstrap_steps=100,
            send_after_done=False,
            send_full_trajectory=False,
        )
        eval_scheme = dict(send_full_trajectory=True, deterministic_action=True, send_concise_info=True)

        return ExperimentConfig(
            actors=[
                ActorWorker(  # Self-play
                    env=self.train_env(env_seed=12023 + i),
                    inference_streams=[self.inference_stream],
                    sample_streams=[self.sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*", inference_stream_idx=0, sample_stream_idx=0, **sample_scheme)
                    ],
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                    worker_info=WorkerInformation(watch_keys="eval"),
                    curriculum_config=Curriculum(type_=Curriculum.Type.Linear, name="3v2"))
                for i in range(self.actors)
            ] + [
                ActorWorker(  # Evaluation-latest
                    env=self.eval_env(env_seed=12023 + i, write_video=True),
                    inference_streams=[self.eval_inf_stream],
                    sample_streams=[self.eval_sample_stream,
                                    SampleStream(type_=SampleStream.Type.NULL)],
                    agent_specs=[
                        AgentSpec(index_regex="0", inference_stream_idx=0, sample_stream_idx=0, **
                                  eval_scheme),
                        AgentSpec(
                            index_regex=".*", inference_stream_idx=0, sample_stream_idx=1, **eval_scheme)
                    ],
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                    worker_info=WorkerInformation(watch_keys="eval"),
                    curriculum_config=Curriculum(type_=Curriculum.Type.Linear, name="3v2"),
                ) for i in range(self.num_eval_actors)
            ],
            policies=[
                PolicyWorker(policy_name=self.policy_name,
                             inference_stream=self.inference_stream,
                             worker_info=WorkerInformation(watch_keys="eval"),
                             policy=self.policy) for _ in range(self.main_policies)
            ] + [
                PolicyWorker(policy_name=self.policy_name,
                             inference_stream=self.eval_inf_stream,
                             policy=self.policy,
                             worker_info=WorkerInformation(watch_keys="eval"),
                             policy_identifier="evaluation") for _ in range(self.num_eval_pw)
            ],
            trainers=[
                TrainerWorker(buffer_name='priority_queue',
                              buffer_args=dict(
                                  max_size=5,
                                  reuses=10,
                                  batch_size=100,
                              ),
                              policy_name=self.policy_name,
                              trainer=self.trainer,
                              policy=self.policy,
                              sample_stream=self.sample_stream,
                              worker_info=WorkerInformation(watch_keys="eval"),
                              push_tag_frequency_minutes=3) for _ in range(self.num_trainer)
            ],
            eval_managers=[
                EvaluationManager(eval_sample_stream=self.eval_sample_stream,
                                  policy_name=self.policy_name,
                                  eval_tag="evaluation",
                                  eval_games_per_version=1000,
                                  worker_info=WorkerInformation(host_key="eval"),
                                  curriculum_config=Curriculum(type_=Curriculum.Type.Linear,
                                                               name="3v2",
                                                               stages=[
                                                                   "3v2_easy",
                                                                   "3v2_medium",
                                                                   "3v2_hard",
                                                               ],
                                                               conditions=[
                                                                   Condition(
                                                                       type_=Condition.Type.SimpleBound,
                                                                       args=dict(field="episode_return",
                                                                                 lower_limit=0.85),
                                                                   ),
                                                                   Condition(
                                                                       type_=Condition.Type.Converged,
                                                                       args=dict(value_field="episode_return",
                                                                                 warmup_step=1000,
                                                                                 step_field="version",
                                                                                 duration=1000,
                                                                                 threshold=5e-2,
                                                                                 confidence=0.90),
                                                                   )
                                                               ]))
            ])


register_experiment(f"fb-3v2-curriculum", Football3v2Curriculum)
