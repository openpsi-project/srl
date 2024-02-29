from api.config import *
import functools
import itertools
import dataclasses

ACTOR_IMAGE = "marl/marl-cpu-gym_mujoco"
TRAINER_IMAGE = "meizy/marl-gpu-mujoco"

gym_mujoco_registry = {
    # env_name: (obs_dim, act_dim)
    "Humanoid-v4": (376, 17),
    "Humanoid-v3": (376, 17),
    "HumanoidStandup-v2": (376, 17),
    "HalfCheetah-v3": (17, 6),
    "Ant-v3": (111, 8),
    "Walker2d-v3": (17, 6),
    "Hopper-v3": (11, 3),
    "Swimmer-v3": (8, 2),
    "InvertedPendulum-v2": (4, 1),
    "InvertedDoublePendulum-v2": (11, 1),
    "Reacher-v2": (11, 2),
}


@dataclasses.dataclass
class GymMuJoCoBenchmarkExperiment(Experiment):
    aws: int = 8
    pws: int = 4
    tws: int = 1

    inline_inference: bool = False
    inference_splits: int = 4
    ring_size: int = 40

    sample_stream_qsize: int = 2560
    inference_stream_qsize: int = 100000

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(
                count=self.aws,
                scheduling=Scheduling.actor_worker_default(cpu=1, mem=1024, container_image=ACTOR_IMAGE)),
            policies=TasksGroup(count=self.pws, scheduling=Scheduling.policy_worker_default(container_image=TRAINER_IMAGE, \
                                                                                        cpu=2, gpu=0.25, mem=20*1024)),
            trainers=TasksGroup(count=self.tws, scheduling=Scheduling.trainer_worker_default(container_image=TRAINER_IMAGE, \
                                                                                        cpu=4, gpu=1, mem=150*1024,
                                                                                        )),
            shared_memory_worker=TasksGroup(count=5, scheduling=Scheduling.shared_memory_worker_default()),
        )

    def initial_setup(self):
        scenario = 'Humanoid-v4'
        policy_name = "default"
        sample_stream = SampleStream(type_=SampleStream.Type.SHARED_MEMORY,
                                     stream_name=policy_name,
                                     qsize=self.sample_stream_qsize)
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)
        policy = Policy(type_="gym_mujoco",
                        args=dict(obs_dim=gym_mujoco_registry[scenario][0],
                                  action_dim=gym_mujoco_registry[scenario][1]))
        # inference_stream = InferenceStream(type_=InferenceStream.Type.SHARED_MEMORY,
        #                                     stream_name=policy_name,
        #                                     qsize=self.inference_stream_qsize)

        inference_streams = [
            InferenceStream(type_=InferenceStream.Type.SHARED_MEMORY,
                            stream_name=policy_name + f"_{i}",
                            qsize=self.inference_stream_qsize,
                            lock_index=i) for i in range(4)
        ]
        # inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        actors = []
        for i in range(4):
            actors += [
                ActorWorker(
                    inference_streams=[inference_streams[i]],
                    sample_streams=[sample_stream],
                    env=Environment(type_="gym_mujoco", args=dict(scenario=scenario)),
                    agent_specs=[AgentSpec(
                        index_regex=".*",
                        inference_stream_idx=0,
                        sample_stream_idx=0,
                    )],
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                ) for _ in range(self.aws // 4)
            ]

        return ExperimentConfig(
            # actors=[
            #     ActorWorker(
            #         inference_streams=[inference_stream],
            #         sample_streams=[sample_stream],
            #         env=Environment(type_="gym_mujoco", args=dict(scenario=scenario)),
            #         agent_specs=[AgentSpec(
            #             index_regex=".*",
            #             inference_stream_idx=0,
            #             sample_stream_idx=0,
            #         )],
            #         ring_size=self.ring_size,
            #         inference_splits=self.inference_splits,
            #     ) for _ in range(self.aws)
            # ],
            actors=actors,
            policies=[
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=inference_streams[i],
                    parameter_db=parameter_db,
                    policy=policy,
                ) for i in range(4)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_zero_copy=True,
                    buffer_args=dict(
                        max_size=2560,
                        reuses=1,
                        batch_size=64,
                    ),
                    policy_name=policy_name,
                    trainer="mappo",
                    policy=policy,
                    sample_stream=sample_stream,
                    parameter_db=parameter_db,
                ) for _ in range(self.tws)
            ],
            shared_memory_worker=[
                SharedMemoryWorker(type_=SharedMemoryWorker.Type.SAMPLE,
                                   stream_name=policy_name,
                                   qsize=self.sample_stream_qsize,
                                   reuses=1)
            ] + [
                SharedMemoryWorker(
                    type_=SharedMemoryWorker.Type.INFERENCE,
                    stream_name=policy_name + f"_{i}",
                    qsize=self.inference_stream_qsize,
                ) for i in range(4)
            ])


aws = [1, 2, 4, 8, 16, 32, 64, 128, 224]
# pws=[0, 1, 2, 4, 8]
pws = [4]
tws = [1]
inline_inference = [False]
inference_split = [1, 2, 3, 4, 5, 6, 7, 8, 20, 40]
ring_size = [10, 20, 30, 40, 50, 60, 70, 80, 100, 200]

for x in itertools.product(aws, pws, tws, inline_inference, inference_split, ring_size):
    aws, pws, tws, inline_inference, inference_split, ring_size = x
    expr_name = f"mujoco-benchmark-a{aws}-p{pws}-t{tws}-i{inference_split}-r{ring_size}"
    register_experiment(expr_name, functools.partial(GymMuJoCoBenchmarkExperiment, *x))

# register_experiment("gym-mujoco-benchmark", GymMuJoCoBenchmarkExperiment)
