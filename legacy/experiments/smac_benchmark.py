from api.config import *

ACTOR_IMAGE = "marl/marl-cpu-smac"
TRAINER_IMAGE = "marl/marl-gpu"


class SMACBenchmarkExperiment(Experiment):

    def __init__(self):
        self.aws = 32
        self.pws = 0
        self.tws = 1

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(count=self.aws,
                              scheduling=Scheduling.actor_worker_default(cpu=2,
                                                                         partition="cpu",
                                                                         container_image=ACTOR_IMAGE)),
            trainers=TasksGroup(count=self.tws,
                                scheduling=Scheduling.trainer_worker_default(cpu=4,
                                                                             container_image=TRAINER_IMAGE,
                                                                             mem=10 * 1024)),
        )

    def initial_setup(self):
        policy_name = "default"
        parameter_db = ParameterDB(type_=ParameterDB.Type.FILESYSTEM, policy_name=policy_name)

        map_name = "27m_vs_30m"
        seed = 1
        ring_size = 1
        # inference_splits = 2  # Actor worker will split the ring into `inference_splits` parts, and flush inference client for each part.
        inference_splits = 1

        def actor_workers_on_node(num, sample_stream_name="default"):
            # inline inference
            inference_stream = InferenceStream(type_=InferenceStream.Type.INLINE,
                                               policy=Policy(type_="smac_rnn",
                                                             args=dict(
                                                                 map_name=map_name,
                                                                 hidden_dim=32,
                                                                 chunk_len=5,
                                                                 seed=seed,
                                                             )),
                                               policy_name=policy_name,
                                               stream_name="")
            # one node sample stream
            sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=sample_stream_name)
            return [
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
                ) for _ in range(num)
            ]

        # since running on g1, each node 1 trainer worker
        def trainer_worker_on_node(sample_stream_name="default"):
            sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=sample_stream_name)
            return TrainerWorker(
                buffer_name='priority_queue',
                buffer_args=dict(
                    max_size=128,
                    reuses=1,
                    batch_size=100,
                ),
                policy_name=policy_name,
                trainer="mappo",
                policy=Policy(type_="smac_rnn",
                              args=dict(
                                  map_name=map_name,
                                  hidden_dim=32,
                                  chunk_len=5,
                                  seed=seed,
                              )),
                sample_stream=sample_stream,
                parameter_db=parameter_db,
            )

        actor_list = actor_workers_on_node(self.aws)
        trainer_list = [trainer_worker_on_node() for _ in range(self.tws)]

        return ExperimentConfig(actors=actor_list, policies=[], trainers=trainer_list, timeout_seconds=1200)


register_experiment("smac-benchmark", SMACBenchmarkExperiment)
