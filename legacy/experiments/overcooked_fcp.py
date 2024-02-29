from api.config import *


class OvercookedFCPExperiment1(Experiment):

    def __init__(self, layout_name: str = 'coordination_ring', num_policies_names: int = 4, seed: int = 0):
        self.seed = seed
        self.num_policy_names = num_policies_names
        self.num_actors = 160
        self.num_policies = 16
        self.num_trainer = 1
        self.num_eval_actors = 40
        self.num_eval_pw = 1
        self.ring_size = 15
        self.inference_splits = 3
        self.layout_name = layout_name
        self.trainer = Trainer(type_="mappo",
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
                                   optimizer_config=dict(lr=1e-4),
                                   popart=True,
                                   max_grad_norm=10.0,
                                   log_per_steps=1,
                                   decay_per_steps=1000,
                                   entropy_bonus_decay=1,
                                   bootstrap_steps=50,
                               ))
        self.policy = Policy(type_="overcooked-separate",
                             args=dict(num_rnn_layers=1, chuck_len=10, rnn_type="lstm", seed=seed))

    def scheduling_setup(self) -> ExperimentScheduling:
        n = self.num_policy_names
        return ExperimentScheduling(actors=TasksGroup(
            (self.num_actors + self.num_eval_actors) * n,
            Scheduling.actor_worker_default(container_image="marl/marl-cpu-overcooked")),
                                    policies=TasksGroup((self.num_policies + self.num_eval_pw) * n,
                                                        Scheduling.policy_worker_default(gpu=0.12)),
                                    trainers=TasksGroup(self.num_trainer * n,
                                                        Scheduling.trainer_worker_default()),
                                    eval_managers=TasksGroup(1 * n, Scheduling.eval_manager_default()))

    def eval_inference_stream(self, policy_name, identifier: Union[str, Dict, List] = "evaluation"):
        return InferenceStream(type_=InferenceStream.Type.INLINE,
                               policy=self.policy,
                               policy_name=policy_name,
                               policy_identifier=identifier,
                               stream_name="")

    def train_env(self):
        return Environment(type_="overcooked", args=dict(layout_name=self.layout_name))

    def eval_env(self, env_idx):
        return Environment(type_="overcooked",
                           args=dict(layout_name=self.layout_name,
                                     render=False,
                                     render_interval=100,
                                     env_idx=env_idx))

    def initial_setup(self):
        return ExperimentConfig(
            actors=[
                ActorWorker(  # Training.
                    env=self.train_env(),
                    inference_streams=[f"policy_{j}"],
                    sample_streams=[f"policy_{j}"],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            sample_steps=400,
                            bootstrap_steps=50,
                            send_after_done=False,
                            send_full_trajectory=False,
                        )
                    ],
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                    worker_info=WorkerInformation(watch_keys=f"policy_{j}"),
                ) for i in range(self.num_actors) for j in range(self.num_policy_names)
            ] + [
                ActorWorker(  # Evaluation
                    env=self.eval_env(env_idx=i),
                    inference_streams=[self.eval_inference_stream(policy_name=f"policy_{j}")],
                    sample_streams=[f"eval_policy_{j}",
                                    SampleStream(type_=SampleStream.Type.NULL)],
                    agent_specs=[
                        AgentSpec(
                            index_regex=f"{i % 2}",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=True,
                            send_concise_info=True,
                        ),
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=1,
                            send_full_trajectory=True,
                            send_concise_info=True,
                        ),
                    ],
                    ring_size=1,
                    inference_splits=self.inference_splits,
                    worker_info=WorkerInformation(watch_keys=f"policy_{j}"),
                ) for i in range(self.num_eval_actors) for j in range(self.num_policy_names)
            ] + [
                ActorWorker(  # Evaluation
                    env=self.eval_env(env_idx=i),
                    inference_streams=[
                        self.eval_inference_stream(policy_name=f"policy_{j}",
                                                   identifier=[{
                                                       "$sort": {
                                                           "version": -1
                                                       }
                                                   }, {
                                                       "$limit": 1
                                                   }])
                    ],
                    sample_streams=[f"eval_policy_{j}",
                                    SampleStream(type_=SampleStream.Type.NULL)],
                    agent_specs=[
                        AgentSpec(
                            index_regex=f"{i % 2}",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=True,
                            send_concise_info=True,
                        ),
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=1,
                            send_full_trajectory=True,
                            send_concise_info=True,
                        ),
                    ],
                    ring_size=1,
                    inference_splits=self.inference_splits,
                    worker_info=WorkerInformation(watch_keys=f"policy_{j}"),
                ) for i in range(self.num_eval_actors) for j in range(self.num_policy_names)
            ],
            policies=[
                PolicyWorker(
                    policy_name=f"policy_{j}",
                    inference_stream=f"policy_{j}",
                    policy=self.policy,
                    worker_info=WorkerInformation(watch_keys=f"policy_{j}"),
                ) for _ in range(self.num_policies) for j in range(self.num_policy_names)
            ],
            trainers=[
                TrainerWorker(buffer_name='priority_queue',
                              buffer_args=dict(
                                  max_size=5,
                                  reuses=10,
                                  batch_size=320,
                              ),
                              policy_name=f"policy_{j}",
                              trainer=self.trainer,
                              policy=self.policy,
                              sample_stream=f"policy_{j}",
                              worker_info=WorkerInformation(wandb_name=f"trainer_policy_{j}",
                                                            log_wandb=True if _ == 0 else False,
                                                            watch_keys=f"policy_{j}"),
                              push_tag_frequency_minutes=1) for _ in range(self.num_trainer)
                for j in range(self.num_policy_names)
            ],
            eval_managers=[
                EvaluationManager(
                    eval_sample_stream=f"eval_policy_{j}",
                    policy_name=f"policy_{j}",
                    eval_tag="evaluation",
                    eval_games_per_version=100,
                    worker_info=WorkerInformation(wandb_name=f"evaluation_policy_{j}",
                                                  log_wandb=True,
                                                  host_key=f"policy_{j}"),
                    curriculum_config=Curriculum(type_=Curriculum.Type.Linear,
                                                 name="training",
                                                 stages="self-play",
                                                 conditions=[
                                                     Condition(type_=Condition.Type.SimpleBound,
                                                               args=dict(field="episode_return",
                                                                         lower_limit=150))
                                                 ]),
                ) for j in range(self.num_policy_names)
            ],
            timeout_seconds=60 * 32)


class OvercookedFCPExperiment2(Experiment):

    # Metadata DB is not working yet.

    def __init__(self, layout_name: str = 'coordination_ring', num_policies_names: int = 4, seed: int = 0):
        self.seed = seed
        self.num_policy_names = num_policies_names
        self.num_actors = 320
        self.num_policies = 16
        self.num_trainer = 1
        self.num_eval_actors = 40
        self.num_eval_pw = 1
        self.ring_size = 15
        self.inference_splits = 3
        self.layout_name = layout_name
        self.trainer = Trainer(type_="mappo",
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
                                   optimizer_config=dict(lr=1e-4),
                                   popart=True,
                                   max_grad_norm=10.0,
                                   log_per_steps=1,
                                   decay_per_steps=1000,
                                   entropy_bonus_decay=1,
                                   bootstrap_steps=50,
                               ))
        self.policy = Policy(type_="overcooked-separate",
                             args=dict(num_rnn_layers=1, chuck_len=10, rnn_type="lstm", seed=seed))

    def eval_inference_stream(self, policy_name, identifier):
        return InferenceStream(type_=InferenceStream.Type.INLINE,
                               policy=self.policy,
                               policy_name=policy_name,
                               policy_identifier=identifier,
                               stream_name="")

    def train_env(self):
        return Environment(type_="overcooked", args=dict(layout_name=self.layout_name))

    def eval_env(self, env_idx):
        return Environment(type_="overcooked",
                           args=dict(layout_name=self.layout_name,
                                     render=True,
                                     render_interval=100,
                                     env_idx=env_idx))

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(self.num_actors + self.num_eval_actors,
                              Scheduling.actor_worker_default(contaimer_image="marl/marl-cpu-overcooked")),
            policies=TasksGroup(self.num_policies, Scheduling.policy_worker_default(gpu=0.12)),
            trainers=TasksGroup(self.num_trainer, Scheduling.trainer_worker_default()),
            eval_managers=TasksGroup(1, Scheduling.eval_manager_default()),
        )

    def initial_setup(self):
        final_policy_name = "policy_x"
        q2 = [{"$sort": {"md.episode_return": -1}}, {"$limit": 1}]
        q3 = [{"$sort": {"md.episode_return": 1}}, {"$limit": 1}]
        q1 = [
            {
                "$group": {
                    "_id": "null",
                    "max_score": {
                        "$max": "$md.episode_return"
                    },
                    "data": {
                        "$push": "$$ROOT"
                    }
                }
            },
            {
                "$unwind": "$data"
            },
            {
                "$addFields": {
                    "diff": {
                        "$abs": {
                            "$subtract": ["$data.md.episode_return", {
                                "$divide": ["$max_score", 2]
                            }]
                        }
                    }
                }
            },
            {
                "$sort": {
                    "diff": 1
                }
            },
            {
                "$limit": 1
            },
            {
                "$addFields": {
                    "version": "$data.version"
                }
            },
        ]

        q_total = [{
            "$facet": {
                "min": q1,
                "max": q2,
                "avg": q3,
            }
        }, {
            "$unwind": "$min"
        }, {
            "$unwind": "$max"
        }, {
            "$unwind": "$avg"
        }, {
            "$addFields": {
                "version": ["$min.version", "$max.version", "$avg.version"]
            }
        }, {
            "$unwind": "$version"
        }]
        return ExperimentConfig(
            actors=[
                ActorWorker(  # Training.
                    env=self.train_env(),
                    inference_streams=[
                        final_policy_name,
                        self.eval_inference_stream(policy_name=None, identifier=q_total)
                    ],
                    sample_streams=[final_policy_name,
                                    SampleStream(type_=SampleStream.Type.NULL)],
                    agent_specs=[
                        AgentSpec(
                            index_regex=f"{i % 2}",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            sample_steps=400,
                            bootstrap_steps=50,
                            send_after_done=False,
                            send_full_trajectory=False,
                        ),
                        AgentSpec(index_regex=".*",
                                  inference_stream_idx=1,
                                  sample_stream_idx=1,
                                  send_concise_info=True),
                    ],
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                ) for i in range(self.num_actors)
            ] + [
                ActorWorker(  # Evaluation
                    env=self.eval_env(env_idx=i),
                    inference_streams=[
                        self.eval_inference_stream(policy_name=final_policy_name, identifier="evaluation"),
                        self.eval_inference_stream(policy_name=None, identifier=q_total)
                    ],
                    sample_streams=[f"eval_{final_policy_name}",
                                    SampleStream(type_=SampleStream.Type.NULL)],
                    agent_specs=[
                        AgentSpec(
                            index_regex=f"{i % 2}",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=True,
                            send_concise_info=True,
                        ),
                        AgentSpec(index_regex=".*",
                                  inference_stream_idx=1,
                                  sample_stream_idx=1,
                                  send_full_trajectory=True,
                                  send_concise_info=True),
                    ],
                    ring_size=1,
                    inference_splits=self.inference_splits,
                ) for i in range(self.num_eval_actors)
            ],
            policies=[
                PolicyWorker(
                    policy_name=final_policy_name,
                    inference_stream=final_policy_name,
                    policy=self.policy,
                ) for _ in range(self.num_policies)
            ],
            trainers=[
                TrainerWorker(buffer_name='priority_queue',
                              buffer_args=dict(
                                  max_size=5,
                                  reuses=10,
                                  batch_size=320,
                              ),
                              foreign_policy=ForeignPolicy(foreign_policy_name="policy_0"),
                              policy_name=final_policy_name,
                              trainer=self.trainer,
                              policy=self.policy,
                              sample_stream=final_policy_name,
                              worker_info=WorkerInformation(wandb_name=f"trainer_{final_policy_name}",
                                                            log_wandb=True if _ == 0 else False),
                              push_tag_frequency_minutes=5) for _ in range(self.num_trainer)
            ],
            eval_managers=[
                EvaluationManager(
                    eval_sample_stream=f"eval_{final_policy_name}",
                    policy_name=final_policy_name,
                    eval_tag="evaluation",
                    eval_games_per_version=500,
                    worker_info=WorkerInformation(wandb_name=f"evaluation_{final_policy_name}",
                                                  log_wandb=True),
                )
            ],
            timeout_seconds=1800)


register_experiment("overcooked-fcp", OvercookedFCPExperiment1, OvercookedFCPExperiment2)
