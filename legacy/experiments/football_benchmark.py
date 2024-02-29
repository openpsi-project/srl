from api.config import *
import dataclasses
import itertools
import functools

map_agent_registry = {
    # evn_name: (left, right, game_length)
    "11_vs_11_competition": (11, 11, 3000),
    "11_vs_11_easy_stochastic": (11, 11, 3000),
    "11_vs_11_hard_stochastic": (11, 11, 3000),
    "11_vs_11_kaggle": (11, 11, 3000),
    "11_vs_11_stochastic": (11, 11, 3000),
    "1_vs_1_easy": (1, 1, 500),
    "5_vs_5": (4, 4, 3000),
    "academy_3_vs_1_with_keeper": (3, 1, 400),
    "academy_corner": (11, 11, 400),
    "academy_counterattack_easy": (11, 11, 400),
    "academy_counterattack_hard": (11, 11, 400),
}

ACTOR_IMAGE = "marl/marl-cpu-football"
TRAINER_IMAGE = "marl/marl-gpu-blosc"
LOCAL_IMAGE = "meizy/marl-gpu-football"


@dataclasses.dataclass
class FootballFPSBenchmark(Experiment):
    # General options
    mode: str  # "local" "distributed" "local_inline" "distributed_inline"
    scale: int
    seed: int = 1

    hidden_dim: int = 128
    entropy_coef: float = 0.01
    rnn_type: str = 'lstm'

    # Env specific options
    def __post_init__(self):
        assert self.mode == "distributed" or self.mode == "distributed-inline", self.mode
        self.aws = 400 * self.scale
        self.pws = 4 * self.scale if "inline" not in self.mode else 0
        self.tws = 1 * self.scale
        self.inference_splits = 4
        self.ring_size = 12

    def scheduling_setup(self) -> ExperimentScheduling:
        actors = TasksGroup(count=self.aws,
                            scheduling=Scheduling.actor_worker_default(
                                cpu=2 if 'inline' in self.mode else 1,
                                mem=5000,
                                container_image=ACTOR_IMAGE,
                                exclude="frl2g[004-034],frl8g[134-137],frl8a[138-141]"))
        trainers = TasksGroup(count=self.tws,
                              scheduling=Scheduling.trainer_worker_default(container_image=TRAINER_IMAGE,
                                                                           cpu=4,
                                                                           gpu=1,
                                                                           mem=60 * 1024,
                                                                           gpu_type="geforce"))
        if "inline" in self.mode:
            policies = []
        else:
            policies = TasksGroup(count=self.pws,
                                  scheduling=Scheduling.policy_worker_default(
                                      cpu=2,
                                      gpu=0.25,
                                      mem=20 * 1024,
                                      container_image=TRAINER_IMAGE,
                                      exclude="frl2g[004-034],frl8g[134-137],frl8a[138-141]"))
        return ExperimentScheduling(
            actors=actors,
            trainers=trainers,
            policies=policies,
            parameter_server_worker=TasksGroup(
                count=1,
                scheduling=Scheduling.parameter_server_worker_default(),
            ),
        )

    def _make_envs(self, aw_rank):
        return [
            Environment(type_="football",
                        args=dict(
                            env_name="11_vs_11_hard_stochastic",
                            number_of_left_players_agent_controls=1,
                            number_of_right_players_agent_controls=0,
                            representation="simple115v2",
                            seed=aw_rank * self.ring_size + j,
                            share_reward=True,
                            rewards="scoring,checkpoints",
                        )) for j in range(self.ring_size)
        ]

    def initial_setup(self):
        burn_in_steps = 0
        bootstrap_steps = 20
        policy = Policy(
            type_="football-simple115-separate",
            args=dict(
                num_rnn_layers=1,
                chunk_len=10,
                rnn_type=self.rnn_type,
                hidden_dim=self.hidden_dim,
                seed=self.seed,
            ),
        )
        agent_specs = [
            AgentSpec(
                index_regex=".*",
                inference_stream_idx=0,
                sample_stream_idx=0,
                sample_steps=400,
                bootstrap_steps=bootstrap_steps,
                burn_in_steps=burn_in_steps,
            )
        ]
        trainer = Trainer(
            type_="mappo",
            args=dict(
                discount_rate=0.99,
                gae_lambda=0.95,
                clip_value=True,
                eps_clip=0.2,
                value_loss='huber',
                value_loss_weight=1,
                value_loss_config=dict(delta=10.0,),
                entropy_bonus_weight=self.entropy_coef,
                optimizer='adam',
                optimizer_config=dict(lr=5e-4),
                ppo_epochs=1,
                popart=True,
                max_grad_norm=10.0,
                bootstrap_steps=bootstrap_steps,
                burn_in_steps=burn_in_steps,
            ),
        )
        self.policy_name = "policy"
        self.buffer_maxsize = 8
        self.batch_size = 1000
        self.parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=self.policy_name)
        if 'inline' in self.mode:
            inference_stream = InferenceStream(
                type_=InferenceStream.Type.INLINE,
                stream_name=self.policy_name,
                policy=policy,
                policy_name=self.policy_name,
                pull_interval_seconds=None,
                parameter_service_client=ParameterServiceClient(),
            )
        else:
            inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=self.policy_name)

        sample_stream = SampleStream(type_=SampleStream.Type.NAME,
                                     stream_name=self.policy_name,
                                     serialization_method="pickle_compress")

        wandb_group = f"x{self.scale}"
        if self.hidden_dim != 128:
            wandb_group += f"h{self.hidden_dim}"
        if self.entropy_coef != 0.01:
            wandb_group += f"ent{self.entropy_coef}"
        if self.rnn_type != 'lstm':
            assert self.rnn_type == 'gru'
            wandb_group += "gru"

        actors = [
            ActorWorker(env=self._make_envs(i),
                        inference_streams=[inference_stream],
                        sample_streams=[sample_stream],
                        agent_specs=agent_specs,
                        max_num_steps=20000,
                        decorrelate_seconds=(i % (self.aws // self.tws)) * 0.1 * self.ring_size,
                        inference_splits=self.inference_splits,
                        ring_size=self.ring_size) for i in range(self.aws)
        ]
        policies = [
            PolicyWorker(
                policy_name=self.policy_name,
                inference_stream=inference_stream,
                parameter_db=self.parameter_db,
                policy=policy,
                pull_frequency_seconds=0.01,
                max_inference_delay=0.05,
                parameter_service_client=ParameterServiceClient(),
            ) for i in range(self.pws)
        ]
        trainers = [
            TrainerWorker(
                buffer_name='priority_queue',
                buffer_args=dict(
                    max_size=self.buffer_maxsize,
                    reuses=4,
                    batch_size=self.batch_size,
                ),
                cudnn_benchmark=True,
                policy_name=self.policy_name,
                trainer=trainer,
                policy=policy,
                log_frequency_seconds=10,
                push_frequency_seconds=1,
                preemption_steps=2500,
                push_tag_frequency_minutes=30,
                sample_stream=sample_stream,
                parameter_db=self.parameter_db,
                worker_info=WorkerInformation(
                    wandb_project="fb11v11",
                    wandb_name=f"seed{self.seed}",
                    wandb_group=wandb_group,
                    wandb_job_type="tw",
                    log_wandb=(i == 0),
                ),
            ) for i in range(self.tws)
        ]

        return ExperimentConfig(
            actors=actors,
            policies=policies,
            trainers=trainers,
            parameter_server_worker=[ParameterServerWorker()],
            timeout_seconds=24 * 3600,
        )


# distributed multi-machine
mode = ['distributed', 'distributed-inline']
scales = [1, 4, 8, 16, 32]  # 1 for testing
seeds = [1, 2, 3]

for x in itertools.product(mode, scales, seeds):
    m, s, seed = x
    expr_name = f"fb11v11-s{seed}x{s}"
    if 'inline' in m:
        expr_name += "-inline"
    register_experiment(expr_name, functools.partial(FootballFPSBenchmark, *x))
    register_experiment(
        expr_name + "h256ent5e-3gru",
        functools.partial(FootballFPSBenchmark, *x, hidden_dim=256, entropy_coef=5e-3, rnn_type='gru'))
