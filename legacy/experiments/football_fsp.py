import functools

from api.config import *


class FootballFSPExperiment(Experiment):

    def __init__(self,
                 multiplier: int,
                 env_name: str = "5_vs_5",
                 left_players: int = 4,
                 right_players: int = 4,
                 representation: str = "simple115v2",
                 algo_name: str = "mappo",
                 ppo_epochs=1,
                 ppo_iterations=32,
                 ppg_epochs: int = 6,
                 seed: int = 0):
        self.env_name = env_name
        self.left_players = left_players
        self.right_players = right_players
        self.representation = representation
        self.algo_name = algo_name
        self.ppg_epochs = ppg_epochs
        self.seed = seed
        self.policy_name = policy_name = "default"
        self.sp_actors = 80 * multiplier
        self.random_sample_actors = 40 * multiplier
        self.strong_sample_actors = 40 * multiplier
        self.main_policies = max(self.left_players + self.right_players, 5) // 5 * 2 * multiplier
        self.strong_policies = max(self.left_players + self.right_players, 5) // 5 * 1 * multiplier
        self.num_trainer = max(self.left_players + self.right_players, 3) // 3 * 1 * multiplier
        self.num_eval_actors = 40
        self.num_eval_pw = 1
        self.ring_size = 3
        self.inference_splits = 3
        self.policy = Policy(
            "football-simple115-separate" if "simple" in self.representation else "football-smm-separate",
            args=dict(num_rnn_layers=1, chunk_len=10, rnn_type="lstm", seed=seed))

        self.inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        self.past_inf_stream = InferenceStream(type_=InferenceStream.Type.INLINE,
                                               stream_name="past-sample-random",
                                               policy_name=policy_name,
                                               policy=self.policy,
                                               policy_identifier={"$match": {}},
                                               pull_interval_seconds=180)
        self.strong_inf_stream = InferenceStream(type_=InferenceStream.Type.NAME,
                                                 stream_name=f"strong_{policy_name}")
        self.eval_inf_stream = InferenceStream(type_=InferenceStream.Type.NAME,
                                               stream_name=f"eval_{policy_name}")

        self.sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
        self.eval_sample_stream = SampleStream(type_=SampleStream.Type.NAME,
                                               stream_name=f"eval_{policy_name}")

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

    def train_env(self, env_seed):
        return Environment(type_="football",
                           args=dict(
                               env_name=self.env_name,
                               number_of_left_players_agent_controls=self.left_players,
                               number_of_right_players_agent_controls=self.right_players,
                               representation=self.representation,
                               rewards="scoring,checkpoints",
                               seed=env_seed,
                           ))

    def eval_env(self, env_seed):
        return Environment(type_="football",
                           args=dict(
                               env_name=self.env_name,
                               number_of_left_players_agent_controls=self.left_players,
                               number_of_right_players_agent_controls=0,
                               representation=self.representation,
                               seed=env_seed,
                           ))

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(actors=TasksGroup(
            count=self.sp_actors + self.random_sample_actors + self.strong_sample_actors +
            self.num_eval_actors,
            scheduling=Scheduling.actor_worker_default(container_image="marl/marl-cpu-football")),
                                    policies=TasksGroup(
                                        count=self.main_policies + self.strong_policies + self.num_eval_pw,
                                        scheduling=Scheduling.policy_worker_default(gpu=0.25)),
                                    trainers=TasksGroup(count=self.num_trainer,
                                                        scheduling=Scheduling.trainer_worker_default()),
                                    eval_managers=TasksGroup(count=1,
                                                             scheduling=Scheduling.eval_manager_default()))

    def initial_setup(self):
        sample_scheme = dict(
            sample_steps=600,
            bootstrap_steps=200,
            send_after_done=False,
            send_full_trajectory=False,
        )
        eval_scheme = dict(send_full_trajectory=True, deterministic_action=True)

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
                ) for i in range(self.sp_actors)
            ] + [
                ActorWorker(  # FSP
                    env=self.train_env(env_seed=12023 + i),
                    inference_streams=[self.inference_stream, self.strong_inf_stream],
                    sample_streams=[self.sample_stream, self.eval_sample_stream],
                    agent_specs=[
                        AgentSpec(index_regex=f"[0-{self.left_players-1}]",
                                  inference_stream_idx=0,
                                  sample_stream_idx=0,
                                  **sample_scheme),
                        AgentSpec(
                            index_regex=".*", inference_stream_idx=1, sample_stream_idx=1, **eval_scheme),
                    ],
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                    worker_info=WorkerInformation(watch_keys="eval"),
                ) for i in range(self.strong_sample_actors)
            ] + [
                ActorWorker(  # Metadata -- Past sampling
                    env=self.train_env(env_seed=12023 + i),
                    inference_streams=[self.inference_stream, self.past_inf_stream],
                    sample_streams=[self.sample_stream, self.eval_sample_stream],
                    agent_specs=[
                        AgentSpec(index_regex=f"[0-{self.left_players-1}]",
                                  inference_stream_idx=0,
                                  sample_stream_idx=0,
                                  **sample_scheme),
                        AgentSpec(
                            index_regex=".*", inference_stream_idx=1, sample_stream_idx=1, **eval_scheme)
                    ],
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                    worker_info=WorkerInformation(watch_keys="eval"),
                ) for i in range(self.random_sample_actors)
            ] + [
                ActorWorker(  # Evaluation-latest
                    env=self.eval_env(env_seed=12023 + i),
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
                ) for i in range(self.num_eval_actors)
            ],
            policies=[
                PolicyWorker(
                    policy_name=self.policy_name,
                    inference_stream=self.inference_stream,
                    worker_info=WorkerInformation(watch_keys="eval"),
                    policy=self.policy,
                ) for _ in range(self.main_policies)
            ] + [
                PolicyWorker(
                    policy_name=self.policy_name,
                    inference_stream=self.eval_inf_stream,
                    policy=self.policy,
                    worker_info=WorkerInformation(watch_keys="eval"),
                    policy_identifier="evaluation",
                ) for _ in range(self.num_eval_pw)
            ] + [
                PolicyWorker(
                    policy_name=self.policy_name,
                    inference_stream=self.strong_inf_stream,
                    pull_frequency_seconds=180,
                    worker_info=WorkerInformation(watch_keys="eval"),
                    policy=self.policy,
                    policy_identifier=[{
                        "$sort": {
                            "md.episode_return": -1
                        }
                    }, {
                        "$limit": 20
                    }],
                ) for _ in range(self.strong_policies)
            ],
            trainers=[
                TrainerWorker(buffer_name='priority_queue',
                              buffer_args=dict(
                                  max_size=64,
                                  reuses=1 if self.algo_name == "mappg" else 10,
                                  batch_size=50 if self.representation == "extracted" else 320,
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
                                  eval_games_per_version=100,
                                  worker_info=WorkerInformation(host_key="eval"),
                                  curriculum_config=Curriculum(type_=Curriculum.Type.Linear,
                                                               name="training",
                                                               stages="self-play",
                                                               conditions=[
                                                                   Condition(
                                                                       type_=Condition.Type.SimpleBound,
                                                                       args=dict(field="episode_return",
                                                                                 lower_limit=0.9),
                                                                   )
                                                               ]))
            ])


for scale in range(1, 10):
    register_experiment(f"fsp-fb-5vs5-x{scale}", functools.partial(FootballFSPExperiment, multiplier=scale))
    register_experiment(
        f"fsp-fb-5vs5-smm-x{scale}",
        functools.partial(FootballFSPExperiment, representation="extracted", multiplier=scale))
