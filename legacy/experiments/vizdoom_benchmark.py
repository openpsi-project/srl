from api.config import *
import dataclasses
import itertools
import functools

ACTOR_IMAGE = "fw/marl-cpu-vizdoom-dmlab"
GPU_IMAGE = "fw/marl-gpu-vizdoom-dmlab"

from api.config import *
import dataclasses
import itertools
import functools

ACTOR_IMAGE = "fw/marl-cpu-vizdoom-dmlab"
GPU_IMAGE = "fw/marl-gpu-vizdoom-dmlab"


@dataclasses.dataclass
class VizdoomFPSBenchmarkExperiment(Experiment):
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
                                scheduling=Scheduling.trainer_worker_default(
                                    cpu=4,
                                    gpu=1,
                                    container_image=GPU_IMAGE,
                                    mem=1024 * 50,
                                    exclude="frl2g[004-034],frl8g[134-137],frl8a[138-141]")),
            shared_memory_worker=TasksGroup(count=1 + self.pws,
                                            scheduling=Scheduling.shared_memory_worker_default()))

    def initial_setup(self):
        ppo_epochs = 1
        bootstrap_steps = 1
        burn_in_steps = 1

        self.policy_name = "default"

        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=self.policy_name)

        # scenario_name = 'doom_battle'
        scenario_name = 'doom_my_way_home_flat_actions'
        # action_dims = [3, 3, 2, 2, 11]
        action_dims = [5]
        obs_shapes = {
            "obs": (3, 72, 128),
            # 'measurements': (23,),
        }

        policy = Policy(type_="vizdoom",
                        args=dict(
                            obs_shapes=obs_shapes,
                            action_dims=action_dims,
                            num_dense_layers=0,
                            hidden_dim=512,
                            popart=True,
                            layernorm=False,
                            rnn_type='gru',
                            num_rnn_layers=0,
                            seed=1,
                            chunk_len=32,
                            popart_beta=0.999999,
                            use_symmetric_kl=True,
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
                              entropy_bonus_weight=0.001,
                              optimizer='adam',
                              optimizer_config=dict(lr=1e-4, eps=1e-6),
                              max_grad_norm=1.0,
                              bootstrap_steps=bootstrap_steps,
                              ppo_epochs=ppo_epochs,
                          ))

        sample_stream = SampleStream(type_=SampleStream.Type.SHARED_MEMORY,
                                     stream_name=self.policy_name,
                                     qsize=2560)
        # sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=self.policy_name)
        inference_streams = [
            InferenceStream(type_=InferenceStream.Type.SHARED_MEMORY,
                            stream_name=self.policy_name + f"_{i}",
                            qsize=100,
                            lock_index=i) for i in range(self.pws)
        ]

        self.env = [
            Environment(
                type_="vizdoom",
                args=dict(
                    seed=12345 + 678900 * x + 1000,
                    scenario_name=scenario_name,
                ),
            ) for x in range(self.ring_size)
        ]

        self.agent_specs = [
            AgentSpec(
                index_regex=".*",
                inference_stream_idx=0,
                sample_stream_idx=0,
                send_full_trajectory=False,
                send_after_done=False,
                sample_steps=32,
                bootstrap_steps=bootstrap_steps,
                burn_in_steps=burn_in_steps,
            )
        ]

        actors = []
        for i in range(self.pws):
            actors += [
                ActorWorker(
                    env=self.env,
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

        # wandb_args = dict(
        #     wandb_project="dmlab-benchmark",
        #     wandb_group=self.wandb_group,
        #     log_wandb=True,
        # )

        return ExperimentConfig(
            actors=actors,
            policies=[
                PolicyWorker(policy_name=self.policy_name,
                             inference_stream=inference_streams[i],
                             parameter_db=parameter_db,
                             policy=policy,
                             batch_size=1000) for i in range(self.pws)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_zero_copy=True,
                    buffer_args=dict(
                        max_size=2560,
                        reuses=1,
                        batch_size=128,
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
    expr_name = f"vizdoom-benchmark"
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
        register_experiment(expr_name, functools.partial(VizdoomFPSBenchmarkExperiment, *x))
    except AssertionError:
        pass

# @dataclasses.dataclass
# class VizDoomBenchmarkExperiment(Experiment):
#     aws: int = 8
#     pws: int = 2
#     tws: int = 1

#     shared_memory: bool = False
#     inline_inference: bool = False
#     inference_splits: int = 4
#     ring_size: int = 8

#     @property
#     def wandb_group(self):
#         expr_name = f"a{self.aws}-p{self.pws}-t{self.tws}"
#         if self.shared_memory:
#             expr_name += "-shm"
#         elif self.inline_inference:
#             expr_name += "-inline"
#         else:
#             expr_name += "-dist"
#         expr_name += f"-i{self.inference_splits}-r{self.ring_size}"
#         return expr_name

#     def scheduling_setup(self) -> ExperimentScheduling:
#         # single machine setup
#         return ExperimentScheduling(
#             actors=TasksGroup(count=self.aws,
#                               scheduling=Scheduling.actor_worker_default(
#                                   cpu=1,
#                                   container_image=ACTOR_IMAGE if not self.shared_memory else GPU_IMAGE,
#                                   mem=50 * self.ring_size)),
#             policies=TasksGroup(count=self.pws,
#                                 scheduling=Scheduling.policy_worker_default(
#                                     cpu=4,
#                                     gpu=0.25,
#                                     container_image="marl/marl-gpu",
#                                     mem=1024 * 20,
#                                     node_list='frl1g065',
#                                     exclude="frl1g105,frl2g032,frl1g059,frl2g021,frl8g136,frl1g104")),
#             trainers=TasksGroup(count=self.tws,
#                                 scheduling=Scheduling.trainer_worker_default(
#                                     cpu=8,
#                                     gpu=1,
#                                     node_list='frl1g066',
#                                     container_image=GPU_IMAGE if self.shared_memory else "marl/marl-gpu",
#                                     mem=1024 * 150)),
#             shared_memory_worker=TasksGroup(count=1, scheduling=Scheduling.shared_memory_worker_default()),
#         )

#     def initial_setup(self):
#         bootstrap_steps = 10
#         burn_in_steps = 32
#         ppo_epochs = 1

#         buffer_qsize = 16
#         batch_size = 64
#         total_slots = buffer_qsize * batch_size

#         scenario_name = 'doom_battle'
#         action_dims = [3, 3, 2, 2, 11]
#         obs_shapes = {
#             "obs": (3, 72, 128),
#             'measurements': (23,),
#         }

#         # scenario_name = 'doom_my_way_home_flat_actions'
#         # action_dims = [5]
#         # obs_shapes = {
#         #     "obs": (3, 72, 128),
#         # }

#         policy_name = "default"
#         if self.shared_memory:
#             sample_stream = SampleStream(
#                 type_=SampleStream.Type.SHARED_MEMORY,
#                 stream_name=policy_name,
#                 qsize=total_slots,
#                 batch_size=batch_size,
#                 reuses=1,
#             )
#         else:
#             sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
#         parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)
#         policy = Policy(type_="vizdoom",
#                         args=dict(
#                             obs_shapes=obs_shapes,
#                             action_dims=action_dims,
#                             num_dense_layers=0,
#                             hidden_dim=512,
#                             popart=True,
#                             layernorm=False,
#                             rnn_type='gru',
#                             num_rnn_layers=1,
#                             seed=1,
#                             chunk_len=32,
#                             popart_beta=0.999999,
#                             use_symmetric_kl=True,
#                         ))

#         if not self.inline_inference:
#             inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
#         else:
#             inference_stream = InferenceStream(type_=InferenceStream.Type.INLINE,
#                                                stream_name=policy_name,
#                                                policy=policy)

#         trainer = Trainer(type_="mappo",
#                           args=dict(
#                               discount_rate=0.99,
#                               gae_lambda=0.95,
#                               eps_clip=0.2,
#                               clip_value=True,
#                               dual_clip=True,
#                               burn_in_steps=burn_in_steps,
#                               recompute_adv_on_reuse=False,
#                               recompute_adv_among_epochs=False,
#                               popart=True,
#                               value_loss='huber',
#                               value_loss_weight=1.0,
#                               value_loss_config=dict(delta=10.0,),
#                               entropy_bonus_weight=0.001,
#                               optimizer='adam',
#                               optimizer_config=dict(lr=1e-4, eps=1e-6),
#                               max_grad_norm=1.0,
#                               bootstrap_steps=bootstrap_steps,
#                               ppo_epochs=ppo_epochs,
#                           ))

#         wandb_args = dict(
#             wandb_project="vizdoom-battle",
#             wandb_group=self.wandb_group,
#         )

#         return ExperimentConfig(
#             actors=[
#                 ActorWorker(
#                     env=[
#                         Environment(
#                             type_="vizdoom",
#                             args=dict(
#                                 seed=12345 * i + 678900 * x + 1000,
#                                 scenario_name=scenario_name,
#                             ),
#                         ) for x in range(self.ring_size)
#                     ],
#                     inference_streams=[
#                         inference_stream,
#                         # inline_inference_stream
#                     ],
#                     sample_streams=[sample_stream],
#                     agent_specs=[
#                         AgentSpec(
#                             index_regex=".*",
#                             inference_stream_idx=0,
#                             sample_stream_idx=0,
#                             send_full_trajectory=False,
#                             send_after_done=False,
#                             sample_steps=32,
#                             bootstrap_steps=bootstrap_steps,
#                             burn_in_steps=burn_in_steps,
#                         )
#                     ],
#                     max_num_steps=20000,
#                     inference_splits=self.inference_splits,
#                     ring_size=self.ring_size) for i in range(self.aws)
#             ],
#             policies=[
#                 PolicyWorker(
#                     policy_name=policy_name,
#                     inference_stream=inference_stream,
#                     max_inference_delay=0.05,
#                     pull_frequency_seconds=0.01,
#                     parameter_db=parameter_db,
#                     policy=policy,
#                 ) for _ in range(self.pws)
#             ],
#             trainers=[
#                 TrainerWorker(
#                     buffer_name='simple_queue',
#                     buffer_zero_copy=True,
#                     cudnn_benchmark=True,
#                     policy_name=policy_name,
#                     trainer=trainer,
#                     policy=policy,
#                     push_frequency_seconds=None,
#                     push_frequency_steps=1,
#                     sample_stream=sample_stream,
#                     preemption_steps=200 // ppo_epochs,
#                     parameter_db=parameter_db,
#                     worker_info=WorkerInformation(
#                         wandb_job_type='tw',
#                         **wandb_args,
#                     ),
#                 ) for _ in range(self.tws)
#             ],
#             shared_memory_worker=[
#                 SharedMemoryWorker(
#                     type_=SharedMemoryWorker.Type.SAMPLE,
#                     stream_name=policy_name,
#                     qsize=total_slots,
#                     reuses=1,
#                 )
#             ])

# aws = [1, 2, 4, 8, 16, 32, 64, 128]
# pws = [2, 4, 8]
# tws = [1]
# shared_memory = [True, False]
# inline_inference = [False]
# inference_split = [1, 2, 4]
# ring_size = [8, 16]

# for x in itertools.product(aws, pws, tws, shared_memory, inline_inference, inference_split, ring_size):
#     aws, pws, tws, shared_memory, inline_inference, inference_split, ring_size = x
#     expr_name = f"vizdoom-a{aws}-p{pws}-t{tws}"
#     if shared_memory:
#         expr_name += "-shm"
#     elif inline_inference:
#         expr_name += "-inline"
#     else:
#         expr_name += "-dist"
#     expr_name += f"-i{inference_split}-r{ring_size}"
#     register_experiment(expr_name, functools.partial(VizDoomBenchmarkExperiment, *x))
