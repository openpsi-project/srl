from api.config import *
import dataclasses
import itertools
import functools

ACTOR_IMAGE = "fw/marl-cpu-vizdoom-dmlab"
GPU_IMAGE = "fw/marl-gpu-vizdoom-dmlab"


@dataclasses.dataclass
class DMLabFPSBenchmarkExperiment(Experiment):
    aws: int = 8
    pws: int = 4
    tws: int = 1

    inline_inference: bool = False
    shared_memory: bool = True
    shared_memory_inference: bool = True
    inference_splits: int = 4
    ring_size: int = 40

    sample_stream_qsize: int = 2560
    inference_stream_qsize: int = 100

    buffer_zero_copy: bool = True

    # @property
    # def wandb_group(self):
    #     expr_name = f"dmlab-{self.aws}-{self.pws}-{self.tws}"
    #     if shared_memory:
    #         expr_name += "-shm"
    #     elif inline_inference:
    #         expr_name += "-inline"
    #     else:
    #         expr_name += "-dist"
    #     expr_name += f"-{self.inference_splits}-{self.ring_size}"
    #     return expr_name

    def scheduling_setup(self) -> ExperimentScheduling:
        # single machine setup
        return ExperimentScheduling(
            actors=TasksGroup(count=self.aws,
                              scheduling=Scheduling.actor_worker_default(cpu=1,
                                                                         container_image=ACTOR_IMAGE,
                                                                         mem=2048)),
            policies=TasksGroup(count=self.pws,
                                scheduling=Scheduling.policy_worker_default(cpu=2,
                                                                            gpu=0.25,
                                                                            container_image=GPU_IMAGE,
                                                                            mem=1024 * 20)),
            trainers=TasksGroup(count=self.tws,
                                scheduling=Scheduling.trainer_worker_default(cpu=4,
                                                                             gpu=1,
                                                                             container_image=GPU_IMAGE,
                                                                             mem=1024 * 50)),
            shared_memory_worker=TasksGroup(count=1 + self.pws,
                                            scheduling=Scheduling.shared_memory_worker_default()))

    def initial_setup(self):
        ppo_epochs = 1
        bootstrap_steps = 1
        burn_in_steps = 1

        self.policy_name = "default"

        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=self.policy_name)
        policy = Policy(
            type_="dmlab",
            args=dict(
                obs_shapes={
                    "obs": (3, 72, 96),
                    'INSTR': (16,)
                },
                action_dim=15,  # 9 for normal action set and 15 for extended action set
                num_dense_layers=0,
                hidden_dim=512,
                popart=True,
                layernorm=False,
                rnn_type='lstm',
                num_rnn_layers=1,
                seed=10000,
                chunk_len=10,
                popart_beta=0.99999,
            ))
        trainer = Trainer(type_="mappo",
                          args=dict(
                              discount_rate=0.99,
                              gae_lambda=0.95,
                              eps_clip=0.2,
                              clip_value=True,
                              dual_clip=True,
                              burn_in_steps=burn_in_steps,
                              recompute_adv_on_reuse=False,
                              recompute_adv_among_epochs=False,
                              popart=True,
                              value_loss='huber',
                              value_loss_weight=1.0,
                              value_loss_config=dict(delta=10.0,),
                              entropy_bonus_weight=0.003,
                              optimizer='adam',
                              optimizer_config=dict(lr=1e-4, eps=1e-6),
                              max_grad_norm=1.0,
                              bootstrap_steps=bootstrap_steps,
                              ppo_epochs=ppo_epochs,
                          ))

        sample_stream = SampleStream(type_=SampleStream.Type.SHARED_MEMORY,
                                     stream_name=self.policy_name,
                                     qsize=2560)
        inference_streams = [
            InferenceStream(type_=InferenceStream.Type.SHARED_MEMORY,
                            stream_name=self.policy_name + f"_{i}",
                            qsize=100,
                            lock_index=i) for i in range(self.pws)
        ]

        self.agent_specs = [
            AgentSpec(
                index_regex=".*",
                inference_stream_idx=0,
                sample_stream_idx=0,
                send_full_trajectory=False,
                send_after_done=False,
                sample_steps=200,
                bootstrap_steps=bootstrap_steps,
                burn_in_steps=burn_in_steps,
            )
        ]

        actors = []
        for i in range(self.pws):
            actors += [
                ActorWorker(
                    env=[
                        Environment(
                            type_="dmlab",
                            args=dict(
                                # NOTE: change to dmlab_benchmark when benchmarking FPS
                                # spec_name="dmlab_benchmark",
                                spec_name="dmlab_watermaze",
                                # NOTE: rank determines which task/level to run
                                rank=(j + i * self.aws // self.pws) * self.ring_size + x,
                                seed=12345 + 678900 * x + 1000,
                            ),
                        ) for x in range(self.ring_size)
                    ],
                    inference_streams=[
                        inference_streams[i],
                        # inline_inference_stream
                    ],
                    sample_streams=[sample_stream],
                    agent_specs=self.agent_specs,
                    max_num_steps=20000,
                    inference_splits=self.inference_splits,
                    ring_size=self.ring_size) for j in range(self.aws // self.pws)
            ]

        # wandb_args = dict(
        #     wandb_project="dmlab-benchmark",
        #     wandb_group=self.wandb_group,
        #     log_wandb=True,
        # )

        return ExperimentConfig(
            actors=actors,
            policies=[
                PolicyWorker(
                    policy_name=self.policy_name,
                    inference_stream=inference_streams[i],
                    parameter_db=parameter_db,
                    policy=policy,
                ) for i in range(self.pws)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_zero_copy=False,
                    buffer_args=dict(
                        max_size=6,
                        reuses=1,
                        batch_size=16,
                    ),
                    policy_name=self.policy_name,
                    trainer=trainer,
                    policy=policy,
                    sample_stream=sample_stream,
                    parameter_db=parameter_db,
                    cudnn_benchmark=True,
                    worker_info=WorkerInformation(
                        #   wandb_job_type='tw',
                        #   wandb_name='tw',
                        #   **wandb_args,
                    )) for _ in range(self.tws)
            ],
            shared_memory_worker=[
                SharedMemoryWorker(type_=SharedMemoryWorker.Type.SAMPLE,
                                   stream_name=self.policy_name,
                                   qsize=self.sample_stream_qsize,
                                   reuses=1)
            ] + [
                SharedMemoryWorker(
                    type_=SharedMemoryWorker.Type.INFERENCE,
                    stream_name=self.policy_name + f"_{i}",
                    qsize=self.inference_stream_qsize,
                ) for i in range(self.pws)
            ])


aws = [1, 2, 4, 8, 16, 32, 64, 128]
pws = [0, 1, 2, 4, 8]
tws = [1]
inline_inference = [True, False]
shared_memory = [True, False]
shared_memory_inference = [True, False]
inference_split = [1, 2, 3, 4, 5, 6, 7, 8, 20, 40]
ring_size = [10, 20, 30, 40, 50, 60, 70, 80, 100, 200]

for x in itertools.product(aws, pws, tws, inline_inference, shared_memory, shared_memory_inference,
                           inference_split, ring_size):
    aws, pws, tws, inline_inference, shared_memory, shared_memory_inference, inference_split, ring_size = x
    expr_name = f"dmlab-benchmark"
    if inline_inference:
        expr_name += "-il"
    if shared_memory:
        expr_name += "-sm"
        if not shared_memory_inference:
            expr_name += "-ri"
    else:
        expr_name += "-d"
    expr_name += f"-a{aws}-p{pws}-t{tws}-i{inference_split}-r{ring_size}"
    try:
        register_experiment(expr_name, functools.partial(DMLabFPSBenchmarkExperiment, *x))
    except AssertionError:
        pass
