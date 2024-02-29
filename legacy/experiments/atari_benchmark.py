from api.config import *
import dataclasses
import itertools
import functools

ACTOR_IMAGE = "marl/marl-cpu"
TRAINER_IMAGE = "meizy/marl-gpu-atari"


@dataclasses.dataclass
class AtariFPSBenchmarkExperiment(Experiment):
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

    def scheduling_setup(self) -> ExperimentScheduling:
        if self.shared_memory:
            return ExperimentScheduling(
                actors=TasksGroup(
                    count=self.aws,
                    scheduling=Scheduling.actor_worker_default(cpu=1, mem=1024, container_image=ACTOR_IMAGE)),
                policies=TasksGroup(count=self.pws, scheduling=Scheduling.policy_worker_default(container_image=TRAINER_IMAGE, \
                                                                                            cpu=2, gpu=0.25, mem=20*1024)),
                trainers=TasksGroup(count=self.tws, scheduling=Scheduling.trainer_worker_default(container_image=TRAINER_IMAGE, \
                                                                                            cpu=4, gpu=1, mem=50*1024,
                                                                                            )),
                shared_memory_worker=TasksGroup(count=1+self.pws if (self.shared_memory_inference and not self.inline_inference) else 1,
                                                scheduling=Scheduling.shared_memory_worker_default()),
            )
        else:
            return ExperimentScheduling(
                actors=TasksGroup(
                    count=self.aws,
                    scheduling=Scheduling.actor_worker_default(cpu=1, mem=1024, container_image=ACTOR_IMAGE)),
                policies=TasksGroup(count=self.pws, scheduling=Scheduling.policy_worker_default(container_image=TRAINER_IMAGE, \
                                                                                            cpu=2, gpu=0.25, mem=20*1024)),
                trainers=TasksGroup(count=self.tws, scheduling=Scheduling.trainer_worker_default(container_image=TRAINER_IMAGE, \
                                                                                            cpu=4, gpu=1, mem=50*1024, node_list="frl8a139", gpu_type="tesla"
                                                                                            ))
            )

    def initial_setup(self):
        self.policy_name = "default"
        self.parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=self.policy_name)
        self.policy = Policy(
            type_="actor-critic",
            args=dict(
                obs_dim={"obs": (4, 84, 84)},
                action_dim=6,
                num_dense_layers=0,
                hidden_dim=512,
                popart=True,
                layernorm=False,
                shared_backbone=True,
                rnn_type='lstm',
                num_rnn_layers=1,
                seed=10000,
                # cnn_layers=dict(obs=[(32, 8, 4, 0, 'zeros'), (64, 4, 2, 0, 'zeros')]),
                cnn_layers=dict(obs=[(16, 8, 4, 0, 'zeros'), (32, 4, 2, 0, 'zeros')]),
                chunk_len=40,
            ))
        self.trainer = Trainer(type_="mappo",
                               args=dict(
                                   discount_rate=0.98,
                                   gae_lambda=0.97,
                                   eps_clip=0.2,
                                   popart=True,
                                   value_loss='huber',
                                   value_loss_weight=0.5,
                                   value_loss_config=dict(delta=10.0,),
                                   entropy_bonus_weight=0.02,
                                   optimizer='adam',
                                   optimizer_config=dict(lr=1e-4),
                                   max_grad_norm=10.0,
                                   entropy_decay_per_steps=1000,
                                   entropy_bonus_decay=0.99,
                                   bootstrap_steps=1,
                               ))
        self.envs = [
            Environment(type_="atari",
                        args=dict(game_name="PongNoFrameskip-v4",
                                  seed=10000 + x,
                                  render=False,
                                  pause=False,
                                  noop_max=30,
                                  frame_skip=4,
                                  stacked_observations=4,
                                  max_episode_steps=108000,
                                  gray_scale=True,
                                  obs_shape=(84, 84))) for x in range(self.ring_size)
        ]
        self.agent_specs = [
            AgentSpec(
                index_regex=".*",
                inference_stream_idx=0,
                sample_stream_idx=0,
                send_full_trajectory=False,
                send_after_done=False,
                sample_steps=200,
                bootstrap_steps=1,
            )
        ]

        print(
            f"setup: inline {self.inline_inference}, shared {self.shared_memory}, shared inference {self.shared_memory_inference}"
        )
        if self.inline_inference:
            assert self.pws == 0
            return self.__inline_inference_setup()
        else:
            return self.__remote_inference_setup()

    def __inline_inference_setup(self):
        if self.shared_memory:
            sample_stream = SampleStream(type_=SampleStream.Type.SHARED_MEMORY,
                                         stream_name=self.policy_name,
                                         qsize=self.sample_stream_qsize)
        else:
            sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=self.policy_name)

        inference_stream = InferenceStream(type_=InferenceStream.Type.INLINE,
                                           stream_name="",
                                           policy=self.policy,
                                           policy_name=self.policy_name)
        return ExperimentConfig(
            actors=[
                ActorWorker(env=self.envs,
                            inference_streams=[inference_stream],
                            sample_streams=[sample_stream],
                            agent_specs=self.agent_specs,
                            max_num_steps=20000,
                            inference_splits=self.inference_splits,
                            ring_size=self.ring_size) for _ in range(self.aws)
            ],
            policies=[],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_zero_copy=self.buffer_zero_copy,
                    buffer_args=dict(
                        max_size=128,
                        reuses=1,
                        batch_size=16,
                    ),
                    policy_name=self.policy_name,
                    trainer=self.trainer,
                    policy=self.policy,
                    sample_stream=sample_stream,
                    parameter_db=self.parameter_db,
                ) for _ in range(self.tws)
            ],
            shared_memory_worker=[
                SharedMemoryWorker(type_=SharedMemoryWorker.Type.SAMPLE,
                                   stream_name=self.policy_name,
                                   qsize=self.sample_stream_qsize,
                                   reuses=1)
            ] if self.shared_memory else [],
        )

    def __remote_inference_setup(self):
        if self.shared_memory:
            sample_stream = SampleStream(type_=SampleStream.Type.SHARED_MEMORY,
                                         stream_name=self.policy_name,
                                         qsize=self.sample_stream_qsize)
            # sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=self.policy_name, qsize=self.sample_stream_qsize)
            if self.shared_memory_inference:
                inference_streams = [
                    InferenceStream(type_=InferenceStream.Type.SHARED_MEMORY,
                                    stream_name=self.policy_name + f"_{i}",
                                    qsize=self.inference_stream_qsize,
                                    lock_index=i) for i in range(self.pws)
                ]
            else:
                inference_streams = [
                    InferenceStream(type_=InferenceStream.Type.NAME, stream_name=self.policy_name)
                    for i in range(self.pws)
                ]

            actors = []
            for i in range(self.pws):
                actors += [
                    ActorWorker(
                        env=self.envs,
                        inference_streams=[
                            inference_streams[i],
                            # inline_inference_stream
                        ],
                        sample_streams=[sample_stream],
                        agent_specs=self.agent_specs,
                        max_num_steps=20000,
                        inference_splits=self.inference_splits,
                        ring_size=self.ring_size) for _ in range(self.aws // self.pws)
                ]
            policies = [
                PolicyWorker(
                    policy_name=self.policy_name,
                    inference_stream=inference_streams[i],
                    parameter_db=self.parameter_db,
                    policy=self.policy,
                ) for i in range(self.pws)
            ]
            shared_memory_worker = [
                SharedMemoryWorker(type_=SharedMemoryWorker.Type.SAMPLE,
                                   stream_name=self.policy_name,
                                   qsize=self.sample_stream_qsize,
                                   reuses=1)
            ] + [
                SharedMemoryWorker(
                    type_=SharedMemoryWorker.Type.INFERENCE,
                    stream_name=self.policy_name + f"_{i}",
                    qsize=self.inference_stream_qsize,
                ) for i in range(self.pws if self.shared_memory_inference else 0)
            ]
        else:
            sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=self.policy_name)
            inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=self.policy_name)
            actors = [
                ActorWorker(env=self.envs,
                            inference_streams=[inference_stream],
                            sample_streams=[sample_stream],
                            agent_specs=self.agent_specs,
                            max_num_steps=20000,
                            inference_splits=self.inference_splits,
                            ring_size=self.ring_size) for _ in range(self.aws)
            ]
            policies = [
                PolicyWorker(
                    policy_name=self.policy_name,
                    inference_stream=inference_stream,
                    parameter_db=self.parameter_db,
                    policy=self.policy,
                ) for i in range(self.pws)
            ]
            self.buffer_zero_copy = False
            shared_memory_worker = []

        return ExperimentConfig(actors=actors,
                                policies=policies,
                                trainers=[
                                    TrainerWorker(
                                        buffer_name='priority_queue',
                                        buffer_zero_copy=self.buffer_zero_copy,
                                        buffer_args=dict(
                                            max_size=2560,
                                            reuses=1,
                                            batch_size=64,
                                        ),
                                        policy_name=self.policy_name,
                                        trainer=self.trainer,
                                        policy=self.policy,
                                        sample_stream=sample_stream,
                                        parameter_db=self.parameter_db,
                                    ) for _ in range(self.tws)
                                ],
                                shared_memory_worker=shared_memory_worker)


# debug this
aws = [1, 2, 4, 8, 16, 32, 64, 128, 224, 256, 384, 512, 1024]
# pws=[0, 1, 2, 4, 8]
pws = [0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
tws = [1, 2, 4, 8, 16, 32]
inline_inference = [True, False]
shared_memory = [True, False]
shared_memory_inference = [True, False]
inference_split = [1, 2, 3, 4, 5, 6, 7, 8, 20, 40]
ring_size = [10, 20, 30, 40, 50, 60, 70, 80, 100, 200]

for x in itertools.product(aws, pws, tws, inline_inference, shared_memory, shared_memory_inference,
                           inference_split, ring_size):
    aws, pws, tws, inline_inference, shared_memory, shared_memory_inference, inference_split, ring_size = x
    expr_name = f"atari-benchmark"
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
        register_experiment(expr_name, functools.partial(AtariFPSBenchmarkExperiment, *x))
    except AssertionError:
        pass


class AtariInlineBenchmarkExperiment(Experiment):

    def __init__(self):
        self.aws = 32
        self.tws = 1
        self.inference_splits = 4
        self.ring_size = 40

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(count=self.aws,
                              scheduling=Scheduling.actor_worker_default(cpu=1,
                                                                         container_image="marl/marl-cpu",
                                                                         mem=2048)),
            trainers=TasksGroup(count=self.tws,
                                scheduling=Scheduling.trainer_worker_default(
                                    cpu=4, gpu=1, container_image="meizy/marl-gpu-atari", mem=1024 * 50)),
            shared_memory_worker=TasksGroup(count=1, scheduling=Scheduling.shared_memory_worker_default()),
        )

    def initial_setup(self):
        self.policy_name = "default"
        self.parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=self.policy_name)
        self.policy = Policy(
            type_="actor-critic",
            args=dict(
                obs_dim={"obs": (4, 84, 84)},
                action_dim=6,
                num_dense_layers=0,
                hidden_dim=512,
                popart=True,
                layernorm=False,
                shared_backbone=True,
                rnn_type='lstm',
                num_rnn_layers=1,
                seed=10000,
                # cnn_layers=dict(obs=[(32, 8, 4, 0, 'zeros'), (64, 4, 2, 0, 'zeros')]),
                # cnn_layers=dict(obs=[(16, 8, 4, 0, 'zeros'), (32, 4, 2, 0, 'zeros')]),
                cnn_layers=dict(obs=[(16, 8, 4, 0, 'zeros'), (32, 4, 2, 0, 'zeros')]),
                chunk_len=40,
            ))
        self.trainer = Trainer(type_="mappo",
                               args=dict(
                                   discount_rate=0.98,
                                   gae_lambda=0.97,
                                   eps_clip=0.2,
                                   popart=True,
                                   value_loss='huber',
                                   value_loss_weight=0.5,
                                   value_loss_config=dict(delta=10.0,),
                                   entropy_bonus_weight=0.02,
                                   optimizer='adam',
                                   optimizer_config=dict(lr=1e-4),
                                   max_grad_norm=10.0,
                                   entropy_decay_per_steps=1000,
                                   entropy_bonus_decay=0.99,
                                   bootstrap_steps=1,
                               ))
        self.envs = [
            Environment(type_="atari",
                        args=dict(game_name="PongNoFrameskip-v4",
                                  seed=10000 + x,
                                  render=False,
                                  pause=False,
                                  noop_max=30,
                                  frame_skip=4,
                                  stacked_observations=4,
                                  max_episode_steps=108000,
                                  gray_scale=True,
                                  obs_shape=(84, 84))) for x in range(self.ring_size)
        ]
        self.agent_specs = [
            AgentSpec(
                index_regex=".*",
                inference_stream_idx=0,
                sample_stream_idx=0,
                send_full_trajectory=False,
                send_after_done=False,
                sample_steps=200,
                bootstrap_steps=1,
            )
        ]

        sample_stream = SampleStream(type_=SampleStream.Type.SHARED_MEMORY,
                                     stream_name=self.policy_name,
                                     qsize=2560)
        inference_stream = InferenceStream(type_=InferenceStream.Type.INLINE,
                                           stream_name=self.policy_name,
                                           policy=self.policy,
                                           policy_name=self.policy_name)

        return ExperimentConfig(
            actors=[
                ActorWorker(
                    env=self.envs,
                    inference_streams=[
                        inference_stream,
                        # inline_inference_stream
                    ],
                    sample_streams=[sample_stream],
                    agent_specs=self.agent_specs,
                    max_num_steps=20000,
                    inference_splits=self.inference_splits,
                    ring_size=self.ring_size) for i in range(self.aws)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_zero_copy=False,
                    buffer_args=dict(
                        max_size=20,
                        reuses=1,
                        batch_size=5,
                    ),
                    policy_name=self.policy_name,
                    trainer=self.trainer,
                    policy=self.policy,
                    sample_stream=sample_stream,
                    parameter_db=self.parameter_db,
                ) for _ in range(self.tws)
            ],
            shared_memory_worker=[
                SharedMemoryWorker(type_=SharedMemoryWorker.Type.SAMPLE,
                                   stream_name=self.policy_name,
                                   qsize=2560,
                                   reuses=1)
            ])


register_experiment("atari-inline", AtariInlineBenchmarkExperiment)
