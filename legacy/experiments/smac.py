import copy
import dataclasses
import functools
import itertools
import math

from api.config import *
import base.timeutil

ACTOR_IMAGE = "meizy/marl-cpu-smac-blosc"


class SMACMiniExperiment(Experiment):

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(count=2,
                              scheduling=Scheduling.actor_worker_default(container_image=ACTOR_IMAGE,
                                                                         mem=4000)),
            policies=TasksGroup(count=1, scheduling=Scheduling.policy_worker_default()),
            trainers=TasksGroup(count=1, scheduling=Scheduling.trainer_worker_default()),
        )

    def initial_setup(self):
        policy_name = "default"
        inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)

        map_name = "5m_vs_6m"
        policy = Policy(type_="smac_rnn", args=dict(
            map_name=map_name,
            hidden_dim=64,
            chunk_len=10,
            seed=1,
        ))
        return ExperimentConfig(
            actors=[
                ActorWorker(
                    env=Environment(type_="smac", args=dict(map_name=map_name)),
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
                        )
                    ],
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
                        max_size=64,
                        reuses=10,
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


class SMACExperiment(Experiment):

    def __init__(self):
        self.num_actors = 32
        self.num_policies = 2
        self.num_trainers = 1

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(count=self.num_actors,
                              scheduling=Scheduling.actor_worker_default(container_image=ACTOR_IMAGE,
                                                                         mem=4000)),
            policies=TasksGroup(count=self.num_policies,
                                scheduling=Scheduling.policy_worker_default(gpu=0.5)),
            trainers=TasksGroup(count=self.num_trainers, scheduling=Scheduling.trainer_worker_default()),
        )

    def initial_setup(self):
        policy_name = "default"
        inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)

        map_name = "5m_vs_6m"
        seed = 1
        ring_size = 4
        inference_splits = 2
        trainer = Trainer(type_="mappo",
                          args=dict(
                              discount_rate=0.99,
                              gae_lambda=0.95,
                              eps_clip=0.2,
                              vtrace=False,
                              clip_value=True,
                              value_eps_clip=0.2,
                              value_loss='huber',
                              value_loss_weight=0.5,
                              value_loss_config=dict(delta=10.0,),
                              entropy_bonus_weight=0.02,
                              optimizer='adam',
                              optimizer_config=dict(lr=5e-4),
                              popart=True,
                              max_grad_norm=10.0,
                              entropy_decay_per_steps=1000,
                          ))
        return ExperimentConfig(
            actors=[
                ActorWorker(
                    env=Environment(type_="smac", args=dict(map_name=map_name)),
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
                    inference_stream=inference_stream,
                    parameter_db=parameter_db,
                    pull_frequency_seconds=3,
                    policy=Policy(type_="smac_rnn",
                                  args=dict(
                                      map_name=map_name,
                                      hidden_dim=64,
                                      chunk_len=10,
                                      seed=seed,
                                  )),
                ) for j in range(self.num_policies)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(
                        max_size=2,
                        reuses=10,
                        batch_size=240,
                    ),
                    policy_name=policy_name,
                    trainer=trainer,
                    policy=Policy(type_="smac_rnn",
                                  args=dict(
                                      map_name=map_name,
                                      hidden_dim=64,
                                      chunk_len=10,
                                      seed=seed,
                                  )),
                    sample_stream=sample_stream,
                    parameter_db=parameter_db,
                ) for _ in range(self.num_trainers)
            ],
        )


class SMACBenchmarkExperiment(Experiment):

    def __init__(self):
        self.num_actors = 32
        self.num_policies = 1
        self.num_trainers = 1

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(actors=TasksGroup(count=self.num_actors,
                                                      scheduling=Scheduling.actor_worker_default(
                                                          container_image=ACTOR_IMAGE, mem=4000)),
                                    policies=TasksGroup(count=self.num_policies,
                                                        scheduling=Scheduling.policy_worker_default()),
                                    trainers=TasksGroup(count=self.num_trainers,
                                                        scheduling=Scheduling.trainer_worker_default()))

    def initial_setup(self):
        policy_name = "default"
        inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)

        map_name = "5m_vs_6m"
        seed = 1
        ring_size = 4
        inference_splits = 2  # Actor worker will split the ring into `inference_splits` parts, and flush inference client for each part.
        return ExperimentConfig(
            actors=[
                ActorWorker(
                    env=Environment(type_="smac", args=dict(map_name=map_name)),
                    inference_streams=[inference_stream],
                    sample_streams=[sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=False,
                            send_after_done=False,
                            sample_steps=400,
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
                    inference_stream=inference_stream,
                    parameter_db=parameter_db,
                    policy=Policy(type_="smac_rnn",
                                  args=dict(
                                      map_name=map_name,
                                      hidden_dim=64,
                                      chunk_len=10,
                                      seed=seed,
                                  )),
                ) for j in range(self.num_policies)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(
                        max_size=64,
                        reuses=10,
                        batch_size=240,
                    ),
                    policy_name=policy_name,
                    trainer="mappo",
                    policy=Policy(type_="smac_rnn",
                                  args=dict(
                                      map_name=map_name,
                                      hidden_dim=64,
                                      chunk_len=10,
                                      seed=seed,
                                  )),
                    sample_stream=sample_stream,
                    parameter_db=parameter_db,
                ) for _ in range(self.num_trainers)
            ],
        )


class SMACDAggerExperiment(Experiment):

    def __init__(self):
        self.num_actors = 4
        self.num_policies = 1
        self.num_trainers = 1

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(actors=TasksGroup(count=self.num_actors,
                                                      scheduling=Scheduling.actor_worker_default(
                                                          container_image=ACTOR_IMAGE, mem=4000)),
                                    policies=TasksGroup(count=self.num_policies,
                                                        scheduling=Scheduling.policy_worker_default()),
                                    trainers=TasksGroup(count=self.num_trainers,
                                                        scheduling=Scheduling.trainer_worker_default()))

    def initial_setup(self):
        policy_name = "default"
        inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)

        map_name = "3m"
        num_actors = 4
        ring_size = 2
        inference_splits = 2
        return ExperimentConfig(
            actors=[
                ActorWorker(
                    env=Environment(type_="smac", args=dict(map_name=map_name)),
                    inference_streams=[inference_stream],
                    sample_streams=[sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=False,
                            send_after_done=False,
                            sample_steps=50,
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
                    inference_stream=inference_stream,
                    parameter_db=parameter_db,
                    foreign_policy=ForeignPolicy(
                        absolute_path="/data/marl/checkpoints/smac-paper-3m-seed0/test0/default/latest",),
                    pull_frequency_seconds=None,
                    policy=Policy(type_="smac_rnn",
                                  args=dict(
                                      map_name=map_name,
                                      hidden_dim=64,
                                      chunk_len=10,
                                      seed=seed,
                                  )),
                ) for j in range(self.num_policies)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(
                        max_size=64,
                        reuses=1,
                        batch_size=10,
                    ),
                    policy_name=policy_name,
                    trainer=Trainer(type_="dagger",
                                    args=dict(optimizer='adam',
                                              optimizer_config=dict(lr=5e-4),
                                              max_grad_norm=10.0,
                                              buffer_size_per_training_iter=4000)),
                    policy=Policy(type_="smac_actor_only",
                                  args=dict(
                                      map_name=map_name,
                                      hidden_dim=64,
                                      chunk_len=10,
                                      seed=seed,
                                  )),
                    sample_stream=sample_stream,
                    parameter_db=parameter_db,
                ) for _ in range(self.num_trainers)
            ],
        )


map_agent_registry = {
    "3m": 3,
    "8m": 8,
    "25m": 25,
    "5m_vs_6m": 5,
    "8m_vs_9m": 8,
    "10m_vs_11m": 10,
    "27m_vs_30m": 27,
    "MMM": 10,
    "MMM2": 10,
    "2s3z": 5,
    "3s5z": 8,
    "3s5z_vs_3s6z": 8,
    "3s_vs_3z": 3,
    "3s_vs_4z": 3,
    "3s_vs_5z": 3,
    "1c3s5z": 9,
    "2m_vs_1z": 2,
    "corridor": 6,
    "6h_vs_8z": 6,
    "2s_vs_1sc": 2,
    "so_many_baneling": 7,
    "bane_vs_bane": 24,
    "2c_vs_64zg": 2,

    # This is adhoc environment
    "1c2z_vs_1c1s1z": 3,
    "1c2s_vs_1c1s1z": 3,
    "2c1z_vs_1c1s1z": 3,
    "2c1s_vs_1c1s1z": 3,
    "1c1s1z_vs_1c1s1z": 3,
    "3s5z_vs_4s4z": 8,
    "4s4z_vs_4s4z": 8,
    "5s3z_vs_4s4z": 8,
    "6s2z_vs_4s4z": 8,
    "2s6z_vs_4s4z": 8,
    "6m_vs_6m_tz": 6,
    "5m_vs_6m_tz": 5,
    "3s6z_vs_3s6z": 9,
    "7h_vs_8z": 7,
    "2s2z_vs_zg": 4,
    "1s3z_vs_zg": 4,
    "3s1z_vs_zg": 4,
    "2s2z_vs_zg_easy": 4,
    "1s3z_vs_zg_easy": 4,
    "3s1z_vs_zg_easy": 4,
    "28m_vs_30m": 28,
    "29m_vs_30m": 29,
    "30m_vs_30m": 30,
    "MMM2_test": 10,
}


@dataclasses.dataclass
class SMACPaperExperiment(Experiment):
    map_name: str
    seed: int
    num_actors: int = 256
    dual_clip: bool = True
    vtrace: bool = False
    popart: bool = True
    value_loss_weight: float = 1.0
    bootstrap_steps: int = 1
    env_truncate: bool = False
    shared: bool = True
    batch_size_factor: int = 8
    data_reuse: int = 5
    compute_gae_on_aw: bool = True
    use_ps: bool = False
    num_rnn_layers: int = 1
    circular_buffer: bool = True
    burn_in_steps: int = 0
    value_clip: bool = False
    unbiased_popart: bool = False
    chunk_len: int = 10
    tw_preemption_steps: int = math.inf
    entropy_coef: float = 0.01
    popart_beta_decimal: int = 5  # popart beta = 1 - 10**(-decimal)
    num_eval_actors: int = 10

    def __post_init__(self):
        self.num_agents = num_agents = map_agent_registry[self.map_name]
        if self.burn_in_steps > 0 and self.num_rnn_layers < 1:
            self.burn_in_steps = 0
        if self.num_rnn_layers == 0:
            self.chunk_len = 10
        self.batch_size = (num_agents * self.batch_size_factor) if not self.shared else self.batch_size_factor

    @property
    def wandb_group_name(self):
        postfix = ""
        if self.dual_clip:
            postfix += "-dc"
        if self.vtrace:
            postfix += "-vtr"
        if self.popart:
            if self.unbiased_popart:
                postfix += '-upa'
            else:
                postfix += "-pa"
        if self.compute_gae_on_aw:
            postfix += "-awg"
        if self.env_truncate:
            postfix += "-t"
        if self.shared:
            postfix += "-sh"
        if self.use_ps:
            postfix += "-ps"
        if not self.circular_buffer:
            postfix += "-exh"
        if self.burn_in_steps > 0:
            postfix += f"-bi{self.burn_in_steps}"
        if not self.value_clip:
            postfix += "-nvc"
        if self.chunk_len != 10:
            postfix += f"-ch{self.chunk_len}"
        if self.tw_preemption_steps < math.inf:
            postfix += f"-e{self.tw_preemption_steps}"
        if self.popart_beta_decimal != 5:
            postfix += f"-pbd{self.popart_beta_decimal}"

        if self.bootstrap_steps > 0:
            return (
                f"aw{self.num_actors}vw{self.value_loss_weight}bo{self.bootstrap_steps}"
                f"bf{self.batch_size_factor}r{self.data_reuse}rn{self.num_rnn_layers}et{self.entropy_coef}" +
                postfix)
        else:
            return (
                f"aw{self.num_actors}vw{self.value_loss_weight}"
                f"bf{self.batch_size_factor}r{self.data_reuse}rn{self.num_rnn_layers}et{self.entropy_coef}" +
                postfix)

    def scheduling_setup(self) -> ExperimentScheduling:
        num_policies = max(1, self.num_actors // 32)
        scheduling = ExperimentScheduling(
            actors=TasksGroup(count=self.num_actors + self.num_eval_actors,
                              scheduling=Scheduling.actor_worker_default(
                                  container_image=ACTOR_IMAGE,
                                  cpu=1,
                                  exclude="frl2g005,frl2g008,frl1g[058-060]",
                                  mem=4000)),
            policies=TasksGroup(count=num_policies,
                                scheduling=Scheduling.policy_worker_default(
                                    gpu=0.12,
                                    cpu=2,
                                    exclude="frl1g085",
                                    container_image='marl/marl-gpu-blosc',
                                    mem=16000)),
            trainers=TasksGroup(count=1,
                                scheduling=Scheduling.trainer_worker_default(
                                    cpu=2,
                                    gpu=1,
                                    exclude='frl1g086',
                                    container_image='marl/marl-gpu-blosc',
                                    mem=200000)),
            eval_managers=TasksGroup(count=1,
                                     scheduling=Scheduling.eval_manager_default(
                                         cpu=1,
                                         mem=10000,
                                         container_image='marl/marl-cpu-blosc',
                                     )),
            controller_image='marl/marl-cpu-blosc',
        )
        if self.use_ps:
            scheduling.parameter_server_worker = TasksGroup(
                count=1,
                scheduling=Scheduling.parameter_server_worker_default(container_image="marl/marl-cpu-blosc"),
            )
        return scheduling

    def initial_setup(self):

        map_name = self.map_name
        seed = self.seed
        num_agents = map_agent_registry[map_name]
        num_policies = max(1, self.num_actors // 32)
        ring_size = 1
        inference_splits = 1
        trainer = Trainer(type_="mappo",
                          args=dict(
                              discount_rate=0.99,
                              gae_lambda=0.95,
                              eps_clip=0.2,
                              dual_clip=self.dual_clip,
                              vtrace=self.vtrace,
                              clip_value=self.value_clip,
                              value_eps_clip=0.2,
                              value_loss='huber',
                              value_loss_weight=self.value_loss_weight,
                              value_loss_config=dict(delta=10.0,),
                              optimizer='adam',
                              optimizer_config=dict(lr=5e-4, eps=1e-5),
                              popart=self.popart,
                              max_grad_norm=10.0,
                              entropy_bonus_weight=self.entropy_coef,
                              bootstrap_steps=self.bootstrap_steps,
                              recompute_adv_on_reuse=False,
                              recompute_adv_among_epochs=False,
                              ppo_epochs=1 if self.circular_buffer else self.data_reuse,
                              burn_in_steps=self.burn_in_steps,
                              normalize_old_value=(self.value_clip and self.compute_gae_on_aw),
                          ))
        policy = Policy(type_="smac_rnn",
                        args=dict(
                            map_name=map_name,
                            hidden_dim=64,
                            chunk_len=self.chunk_len,
                            seed=seed,
                            shared=self.shared,
                            act_init_gain=0.01 if self.map_name != 'MMM2' else 1.0,
                            num_rnn_layers=self.num_rnn_layers,
                            denormalize_value_during_rollout=self.compute_gae_on_aw,
                            popart=self.popart,
                            unbiased_popart=self.unbiased_popart,
                            popart_beta=1 - 10**(-self.popart_beta_decimal),
                        ))

        policy_name = "default"
        inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        eval_inference_stream = InferenceStream(
            type_=InferenceStream.Type.INLINE,
            stream_name=f"eval_{policy_name}",
            policy=policy,
            policy_name=policy_name,
            policy_identifier='eval',
        )
        sample_stream = SampleStream(type_=SampleStream.Type.NAME,
                                     stream_name=policy_name,
                                     serialization_method="raw_compress")
        eval_sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=f"eval_{policy_name}")

        wandb_args = dict(
            log_wandb=True,
            wandb_project=f'smac_sweep_{self.map_name}',
            wandb_group=self.wandb_group_name,
            wandb_config={
                f.name: getattr(self, f.name)
                for f in dataclasses.fields(self) if f.name != 'map_name'
            },
        )

        return ExperimentConfig(
            actors=[
                ActorWorker(
                    env=[
                        Environment(type_="smac",
                                    args=dict(
                                        map_name=map_name,
                                        use_truncate=self.env_truncate,
                                        shared=self.shared,
                                        seed=self.seed + (k * ring_size + j) * 1000,
                                        discount_rate=0.99,
                                    )) for j in range(ring_size)
                    ],
                    inference_streams=[inference_stream],
                    sample_streams=[sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=False,
                            send_after_done=self.compute_gae_on_aw,
                            sample_steps=400,
                            bootstrap_steps=self.bootstrap_steps,
                            burn_in_steps=self.burn_in_steps,
                            compute_gae_before_send=self.compute_gae_on_aw,
                            gae_args=dict(
                                gamma=0.99,
                                lmbda=0.95,
                                unmask_death_steps=True,
                            ),
                        )
                    ],
                    ring_size=ring_size,
                    inference_splits=inference_splits,
                ) for k in range(self.num_actors)
            ] + [
                ActorWorker(
                    env=Environment(type_="smac",
                                    args=dict(
                                        map_name=map_name,
                                        use_truncate=self.env_truncate,
                                        shared=self.shared,
                                        seed=self.seed * 50000 + k * 10000,
                                        discount_rate=1,
                                    )),
                    inference_streams=[eval_inference_stream],
                    sample_streams=[eval_sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=True,
                            deterministic_action=True,
                            send_concise_info=True,
                        )
                    ],
                    ring_size=1,
                ) for k in range(self.num_eval_actors)
            ],
            policies=[
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=inference_stream,
                    max_inference_delay=0.05,
                    pull_frequency_seconds=None
                    if self.use_ps else 0.01,  # do not actively pull if use parameter service
                    pull_max_failures=100,
                    policy=policy,
                    parameter_service_client=ParameterServiceClient() if self.use_ps else None,
                ) for j in range(num_policies)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(
                        max_size=6,
                        reuses=self.data_reuse if self.circular_buffer else 1,
                        batch_size=self.batch_size,
                    ),
                    policy_name=policy_name,
                    trainer=trainer,
                    push_frequency_seconds=0.01,
                    push_frequency_steps=1,
                    log_frequency_seconds=1,
                    preemption_steps=self.tw_preemption_steps,
                    policy=policy,
                    sample_stream=sample_stream,
                    worker_info=WorkerInformation(
                        wandb_name=f"tw_seed{self.seed}",
                        wandb_job_type="tw",
                        **wandb_args,
                    ),
                ) for i in range(1)
            ],
            parameter_server_worker=[ParameterServerWorker()] if self.use_ps else [],
            eval_managers=[
                EvaluationManager(
                    eval_sample_stream=f"eval_{policy_name}",
                    policy_name=policy_name,
                    eval_games_per_version=100 if self.shared else 100 * self.num_agents,
                    worker_info=WorkerInformation(
                        wandb_job_type="em",
                        wandb_name=f"em_seed{self.seed}",
                        **wandb_args,
                    ),
                ),
            ],
        )


class SMACAttentionExperiment(Experiment):

    def __init__(self, seed, map_name):
        self.seed = seed
        self.map_name = map_name
        self.num_agents = map_agent_registry[map_name]
        self.num_actors = 64
        self.num_policies = 4
        self.num_trainers = 1

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(self.num_actors,
                              Scheduling.actor_worker_default(container_image=ACTOR_IMAGE, mem=4000)),
            policies=TasksGroup(self.num_policies, Scheduling.policy_worker_default()),
            trainers=TasksGroup(self.num_trainers, Scheduling.trainer_worker_default()))

    def initial_setup(self):
        policy_name = "default"
        inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)

        shared = True
        agent_specific_obs = True
        agent_specific_state = True
        dataflow_config = dict(shared=shared,
                               agent_specific_obs=agent_specific_obs,
                               agent_specific_state=agent_specific_state)

        map_name = self.map_name
        seed = self.seed
        ring_size = 2
        inference_splits = 2

        hidden_dim = 128
        buffer_bs = self.num_actors * ring_size // 4
        if not shared:
            buffer_bs *= self.num_agents

        trainer = Trainer(type_="mappo",
                          args=dict(
                              discount_rate=0.99,
                              gae_lambda=0.95,
                              eps_clip=0.2,
                              vtrace=False,
                              clip_value=True,
                              value_eps_clip=0.2,
                              value_loss='huber',
                              value_loss_weight=0.5,
                              value_loss_config=dict(delta=10.0,),
                              entropy_bonus_weight=0.02,
                              optimizer='adam',
                              optimizer_config=dict(lr=5e-4),
                              popart=True,
                              max_grad_norm=10.0,
                              entropy_decay_per_steps=1000,
                          ))

        policy = Policy(type_="smac_rnn",
                        args=dict(map_name=map_name,
                                  hidden_dim=hidden_dim,
                                  chunk_len=10,
                                  seed=seed,
                                  **dataflow_config))

        return ExperimentConfig(
            actors=[
                ActorWorker(
                    env=Environment(type_="smac", args=dict(map_name=map_name, **dataflow_config)),
                    inference_streams=[inference_stream],
                    sample_streams=[sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=False,
                            send_after_done=False,
                            sample_steps=400,
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
                    inference_stream=inference_stream,
                    parameter_db=parameter_db,
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
                        max_size=16,
                        reuses=5,
                        batch_size=buffer_bs,
                    ),
                    policy_name=policy_name,
                    trainer=trainer,
                    push_frequency_seconds=0.5,
                    push_frequency_steps=10,
                    log_frequency_seconds=1,
                    policy=policy,
                    sample_stream=sample_stream,
                    parameter_db=parameter_db,
                ) for _ in range(self.num_trainers)
            ],
        )


@dataclasses.dataclass
class SMACQMixExperiment(Experiment):
    map_name: str
    seed: int

    td_lambda: Optional[float] = None
    popart: bool = False
    bootstrap_steps: int = 1
    env_truncate: bool = False
    batch_size: int = 32
    burn_in_steps: int = 0
    unbiased_popart: bool = False  # TODO: not implemented
    chunk_len: int = 100

    use_priority_weight: bool = True

    def __post_init__(self):
        self.num_agents = map_agent_registry[self.map_name]

        self.num_actors = 256
        self.num_policies = 8
        self.num_trainers = 1
        self.num_eval_actors = 10

        self.sample_steps = 400

    @property
    def wandb_group_name(self):
        postfix = ""
        if self.td_lambda is not None:
            postfix += f"-tdl{self.td_lambda}"
        if self.popart:
            if self.unbiased_popart:
                postfix += '-upa'
            else:
                postfix += "-pa"
        if self.env_truncate:
            postfix += "-t"
        if self.burn_in_steps > 0:
            postfix += f"-bi{self.burn_in_steps}"
        if self.chunk_len != 10:
            postfix += f"-ch{self.chunk_len}"
        if not self.use_priority_weight:
            postfix += "-npw"

        return f"bo{self.bootstrap_steps}b{self.batch_size}" + postfix

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(count=self.num_actors + self.num_eval_actors,
                              scheduling=Scheduling.actor_worker_default(
                                  container_image=ACTOR_IMAGE,
                                  cpu=1,
                                  exclude="frl2g005,frl2g008,frl1g[058-060]",
                                  mem=4000)),
            policies=TasksGroup(count=self.num_policies,
                                scheduling=Scheduling.policy_worker_default(gpu=0.12,
                                                                            cpu=2,
                                                                            exclude="frl1g085,frl2g019",
                                                                            mem=16000)),
            trainers=TasksGroup(count=1,
                                scheduling=Scheduling.trainer_worker_default(cpu=2,
                                                                             gpu=1,
                                                                             exclude='frl1g086',
                                                                             mem=200000)),
            eval_managers=TasksGroup(count=1, scheduling=Scheduling.eval_manager_default(cpu=1, mem=10000)),
        )

    def initial_setup(self):
        policy_name = "default"

        map_name = self.map_name
        seed = self.seed
        ring_size = 1  # too many smac environments can be extremely easy to crush
        inference_splits = 1
        trainer = Trainer(type_="q-learning",
                          args=dict(
                              gamma=0.99,
                              use_soft_update=False,
                              tau=0.005,
                              hard_update_interval=200,
                              max_grad_norm=10.0,
                              use_popart=self.popart,
                              optimizer="adam",
                              optimizer_config=dict(lr=5e-4, eps=1e-5, weight_decay=0.),
                              value_loss="mse",
                              value_loss_config={},
                              bootstrap_steps=self.bootstrap_steps,
                              burn_in_steps=self.burn_in_steps,
                              use_priority_weight=self.use_priority_weight,
                          ))
        policy = Policy(
            type_="smac-qmix",
            args=dict(
                map_name=map_name,
                chunk_len=self.chunk_len,  # chunk length requires modification for different map
                use_double_q=True,
                epsilon_start=1.0,
                epsilon_finish=0.05,
                epsilon_anneal_time=5000,
                q_i_config=dict(hidden_dim=128, num_dense_layers=2, rnn_type="gru", num_rnn_layers=1),
                mixer_config=dict(
                    popart=self.popart,
                    hidden_dim=64,
                    num_hypernet_layers=2,
                    hypernet_hidden_dim=64,
                ),
                state_use_all_local_obs=False,
                state_concate_all_local_obs=True,
                seed=seed,
            ))
        eval_inference_stream = InferenceStream(
            type_=InferenceStream.Type.INLINE,
            stream_name=f"eval_{policy_name}",
            policy=policy,
            policy_name=policy_name,
            policy_identifier='eval',
        )
        eval_sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=f"eval_{policy_name}")

        wandb_args = dict(
            log_wandb=True,
            wandb_project=f'smac_qmix_{self.map_name}',
            wandb_group=self.wandb_group_name,
            wandb_config={
                f.name: getattr(self, f.name)
                for f in dataclasses.fields(self) if f.name != 'map_name'
            },
        )

        return ExperimentConfig(
            actors=[
                ActorWorker(
                    env=Environment(
                        type_="smac",
                        args=dict(
                            map_name=map_name,
                            shared=True,
                            use_state_agent=False,
                            seed=k * 1000 + seed * 97,
                            use_truncate=self.env_truncate,
                        ),
                    ),
                    inference_streams=[policy_name],
                    sample_streams=[policy_name],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=False,
                            send_after_done=False,
                            sample_steps=self.sample_steps,
                            bootstrap_steps=self.bootstrap_steps,
                            burn_in_steps=self.burn_in_steps,
                        )
                    ],
                    max_num_steps=2000,
                    ring_size=ring_size,
                    inference_splits=inference_splits,
                ) for k in range(self.num_actors)
            ] + [
                ActorWorker(  # Evaluation
                    env=Environment(type_="smac",
                                    args=dict(map_name=map_name,
                                              shared=True,
                                              use_state_agent=False,
                                              seed=(self.num_actors + i) * 1000 + seed * 97)),
                    inference_streams=[eval_inference_stream],
                    sample_streams=[eval_sample_stream,
                                    SampleStream(type_=SampleStream.Type.NULL)],
                    agent_specs=[
                        AgentSpec(index_regex="0",
                                  inference_stream_idx=0,
                                  sample_stream_idx=0,
                                  send_full_trajectory=True,
                                  deterministic_action=True,
                                  send_concise_info=True),
                        AgentSpec(index_regex=".*",
                                  inference_stream_idx=0,
                                  sample_stream_idx=1,
                                  send_full_trajectory=True,
                                  deterministic_action=True,
                                  send_concise_info=True),
                    ],
                    ring_size=ring_size,
                    inference_splits=inference_splits,
                ) for i in range(self.num_eval_actors)
            ],
            policies=[
                PolicyWorker(policy_name=policy_name, inference_stream=policy_name, policy=policy)
                for j in range(self.num_policies)
            ],
            trainers=[
                TrainerWorker(buffer_name='prioritized_replay_buffer',
                              buffer_args=dict(
                                  max_size=int(1e5) // 400,
                                  batch_size=self.batch_size,
                                  sample_length=self.sample_steps + self.burn_in_steps + self.bootstrap_steps,
                                  batch_length=self.burn_in_steps + self.chunk_len,
                                  warmup_transitions=int(2e3),
                                  seed=self.seed,
                                  alpha=0.6,
                                  beta=0.4,
                                  beta_scheduler=base.timeutil.ChainedScheduler([
                                      base.timeutil.LinearScheduler(
                                          init_value=0.4,
                                          total_iters=int(20e3),
                                          end_value=1.0,
                                      ),
                                      base.timeutil.ConstantScheduler(init_value=1.0, total_iters=int(1e10)),
                                  ]),
                                  max_priority=1.0,
                              ),
                              save_buffer_on_exit=True,
                              load_buffer_on_restart=True,
                              policy_name=policy_name,
                              trainer=trainer,
                              push_frequency_seconds=0.01,
                              push_frequency_steps=1,
                              log_frequency_seconds=1,
                              policy=policy,
                              sample_stream=policy_name,
                              worker_info=WorkerInformation(
                                  wandb_name=f"tw_seed{self.seed}",
                                  wandb_job_type='tw',
                                  **wandb_args,
                              )) for _ in range(self.num_trainers)
            ],
            eval_managers=[
                EvaluationManager(
                    eval_sample_stream=eval_sample_stream,
                    policy_name=policy_name,
                    eval_tag="eval",
                    eval_games_per_version=100,
                    worker_info=WorkerInformation(
                        wandb_job_type="em",
                        wandb_name=f"em_seed{self.seed}",
                        **wandb_args,
                    ),
                )
            ])


class SMACVanillaPBTExperiment(Experiment):

    def __init__(self, seed, map_name, pop_size):
        self.seed = seed
        self.map_name = map_name
        self.num_agents = map_agent_registry[map_name]
        self.num_actors = 64
        self.num_policies = 4
        self.num_trainers = 1
        self.num_eval_actors = 10
        self.num_eval_policies = 1
        self.pop_size = pop_size
        self.ring_size = 2
        self.inference_splits = 2
        self.initial_hyperparams = None

        self.population = [f"policy_{i}" for i in range(pop_size)]
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

        self.policy = Policy(type_="smac_rnn",
                             args=dict(map_name=map_name, hidden_dim=64, chunk_len=10, seed=seed))
        self.population_sample_stream = SampleStream(type_=SampleStream.Type.NAME,
                                                     stream_name="population_stream")
        self.null_sample_stream = SampleStream(type_=SampleStream.Type.NULL)

    def make_trainer(self, lr):
        return Trainer(type_="mappo",
                       args=dict(discount_rate=0.99,
                                 gae_lambda=0.95,
                                 eps_clip=0.2,
                                 vtrace=False,
                                 clip_value=True,
                                 value_eps_clip=0.2,
                                 value_loss='huber',
                                 value_loss_weight=0.5,
                                 value_loss_config=dict(delta=10.0,),
                                 entropy_bonus_weight=0.02,
                                 optimizer='adam',
                                 optimizer_config=dict(lr=lr),
                                 popart=True,
                                 max_grad_norm=10.0,
                                 entropy_decay_per_steps=1000))

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(actors=TasksGroup(
            (self.num_actors + self.num_eval_actors) * self.pop_size,
            Scheduling.actor_worker_default(container_image=ACTOR_IMAGE, cpu=2, mem=4000)),
                                    policies=TasksGroup(
                                        (self.num_eval_policies + self.num_policies) * self.pop_size,
                                        Scheduling.policy_worker_default(gpu=0.2)),
                                    trainers=TasksGroup(self.num_trainers * self.pop_size,
                                                        Scheduling.trainer_worker_default()),
                                    eval_managers=TasksGroup(self.pop_size,
                                                             Scheduling.eval_manager_default()),
                                    population_manager=TasksGroup(1, Scheduling.population_manager_default()))

    def initial_setup(self):
        import numpy as np
        actors = []
        policies = []
        trainers = []
        eval_managers = []
        self.initial_hyperparams = [dict(lr=lr) for lr in 10**(-np.linspace(2, 5, pop_size))]

        for idx, policy_name in enumerate(self.population):
            trainer = self.make_trainer(**self.initial_hyperparams[idx])
            inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
            sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
            eval_inference_stream = InferenceStream(type_=InferenceStream.Type.NAME,
                                                    stream_name=f"eval_{policy_name}")
            eval_sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=f"eval_{policy_name}")

            actors.extend([  # Training.
                ActorWorker(
                    env=Environment(type_="smac", args=dict(map_name=self.map_name)),
                    inference_streams=[inference_stream],
                    sample_streams=[sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            sample_steps=400,
                            send_after_done=False,
                            send_full_trajectory=False,
                        )
                    ],
                    max_num_steps=2000,
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                ) for i in range(self.num_actors)
            ] + [
                ActorWorker(  # Evaluation.
                    env=Environment(type_="smac", args=dict(map_name=self.map_name)),
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
                            deterministic_action=True,
                            send_concise_info=True,
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
                ) for j in range(self.num_policies)
            ] + [  # Evaluation.
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=eval_inference_stream,
                    policy=self.policy,
                    policy_identifier="evaluation",
                ) for j in range(self.num_eval_policies)
            ])

            trainers.extend([
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(
                        max_size=16,
                        reuses=5,
                        batch_size=self.num_agents * self.num_actors * self.ring_size // 4,
                    ),
                    policy_name=policy_name,
                    trainer=trainer,
                    push_frequency_seconds=0.5,
                    log_frequency_seconds=1,
                    policy=self.policy,
                    sample_stream=sample_stream,
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


register_experiment("smac-mini", SMACMiniExperiment)
register_experiment("smac", SMACExperiment)
# register_experiment("smac-benchmark", SMACBenchmarkExperiment)
# register_experiment("smac-dagger", SMACDAggerExperiment)  # Experiment is not working.
for m in map_agent_registry:
    for seed in range(64):
        register_experiment(f"smac-attn-{m.replace('_', '-')}-seed{seed}",
                            functools.partial(SMACAttentionExperiment, seed, m))
        for pop_size in [5, 10, 20, 40]:
            # SC2 Client not stable
            register_experiment(f"smac-pbt-{m.replace('_', '-')}-pop{pop_size}-seed{seed}",
                                functools.partial(SMACVanillaPBTExperiment, seed, m, pop_size))

maps = list(map_agent_registry.keys())
seeds = list(range(32))
num_actors = [256]
dual_clip = [True, False]
vtrace = [False]
popart = [True]
value_loss_weight = [1.0]
bootstrap_steps = [1]
env_truncate = [False, True]
shared = [True, False]
batch_size_factor = [8]
data_reuse = [5]
compute_gae_on_aw = [False, True]
use_ps = [False]
num_rnn_layers = [1]
circular_buffer = [False]
burn_in_steps = [0, 10]
value_clip = [True]
unbiased_popart = [False]
chunk_len = [10]
tw_preemption_steps = [math.inf]
entropy_coef = [0.01]
popart_beta_decimal = [5]
for x in itertools.product(maps, seeds, num_actors, dual_clip, vtrace, popart, value_loss_weight,
                           bootstrap_steps, env_truncate, shared, batch_size_factor, data_reuse,
                           compute_gae_on_aw, use_ps, num_rnn_layers, circular_buffer, burn_in_steps,
                           value_clip, unbiased_popart, chunk_len, tw_preemption_steps, entropy_coef,
                           popart_beta_decimal):
    m = x[0]
    seed = x[1]
    name = SMACPaperExperiment(*x).wandb_group_name
    register_experiment(f"smp{m.replace('_vs_', 'v').replace('_', '-')}-s{seed}{name}",
                        functools.partial(SMACPaperExperiment, *x))

maps = list(map_agent_registry.keys())
seeds = list(range(32))
td_lambda = [None]
popart = [False]
bootstrap_steps = [1, 5]
env_truncate = [False, True]
batch_size = [32]
burn_in_steps = [40]
unbiased_popart = [False]
chunk_len = [100]
use_priority_weight = [True]
for x in itertools.product(maps, seeds, td_lambda, popart, bootstrap_steps, env_truncate, batch_size,
                           burn_in_steps, unbiased_popart, chunk_len, use_priority_weight):
    m = x[0]
    seed = x[1]
    name = SMACQMixExperiment(*x).wandb_group_name
    register_experiment(f"smq{m.replace('_vs_', 'v').replace('_', '-')}-s{seed}{name}",
                        functools.partial(SMACQMixExperiment, *x))