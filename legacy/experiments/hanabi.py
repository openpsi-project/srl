import functools

from api.config import *

HANABI_PLAYER_DIM_MAPPING = {
    ('Hanabi-Full', 2, True): {
        'obs_dim': 660,
        'state_dim': 1318,
        'action_dim': 20
    },
    ('Hanabi-Full-Minimal', 2, True): {
        'obs_dim': 310,
        'state_dim': 618,
        'action_dim': 20
    },
    ('Hanabi-Small', 2, True): {
        'obs_dim': 173,
        'state_dim': 344,
        'action_dim': 11
    },
    ('Hanabi-Very-Small', 2, True): {
        'obs_dim': 108,
        'state_dim': 214,
        'action_dim': 10
    },
    ('Hanabi-Full', 2, False): {
        'obs_dim': 660,
        'state_dim': 785,
        'action_dim': 20
    },
    ('Hanabi-Full-Minimal', 2, False): {
        'obs_dim': 310,
        'state_dim': 435,
        'action_dim': 20
    },
    ('Hanabi-Small', 2, False): {
        'obs_dim': 173,
        'state_dim': 193,
        'action_dim': 11
    },
    ('Hanabi-Very-Small', 2, False): {
        'obs_dim': 108,
        'state_dim': 118,
        'action_dim': 10
    },
    ('Hanabi-Full', 3, True): {
        'obs_dim': 959,
        'state_dim': 2871,
        'action_dim': 30
    },
    ('Hanabi-Full-Minimal', 3, True): {
        'obs_dim': 434,
        'state_dim': 1296,
        'action_dim': 30
    },
    ('Hanabi-Small', 3, True): {
        'obs_dim': 229,
        'state_dim': 681,
        'action_dim': 18
    },
    ('Hanabi-Very-Small', 3, True): {
        'obs_dim': 142,
        'state_dim': 420,
        'action_dim': 16
    },
    ('Hanabi-Full', 3, False): {
        'obs_dim': 959,
        'state_dim': 1084,
        'action_dim': 30
    },
    ('Hanabi-Full-Minimal', 3, False): {
        'obs_dim': 434,
        'state_dim': 559,
        'action_dim': 30
    },
    ('Hanabi-Small', 3, False): {
        'obs_dim': 229,
        'state_dim': 249,
        'action_dim': 18
    },
    ('Hanabi-Very-Small', 3, False): {
        'obs_dim': 142,
        'state_dim': 152,
        'action_dim': 16
    },
    ('Hanabi-Full', 4, True): {
        'obs_dim': 1045,
        'state_dim': 4168,
        'action_dim': 38
    },
    ('Hanabi-Full-Minimal', 4, True): {
        'obs_dim': 485,
        'state_dim': 1928,
        'action_dim': 38
    },
    ('Hanabi-Small', 4, True): {
        'obs_dim': 285,
        'state_dim': 1128,
        'action_dim': 25
    },
    ('Hanabi-Very-Small', 4, True): {
        'obs_dim': 176,
        'state_dim': 692,
        'action_dim': 22
    },
    ('Hanabi-Full', 4, False): {
        'obs_dim': 1045,
        'state_dim': 1145,
        'action_dim': 38
    },
    ('Hanabi-Full-Minimal', 4, False): {
        'obs_dim': 485,
        'state_dim': 585,
        'action_dim': 38
    },
    ('Hanabi-Small', 4, False): {
        'obs_dim': 285,
        'state_dim': 305,
        'action_dim': 25
    },
    ('Hanabi-Very-Small', 4, False): {
        'obs_dim': 176,
        'state_dim': 186,
        'action_dim': 22
    },
    ('Hanabi-Full', 5, True): {
        'obs_dim': 1285,
        'state_dim': 6405,
        'action_dim': 48
    },
    ('Hanabi-Full-Minimal', 5, True): {
        'obs_dim': 585,
        'state_dim': 2905,
        'action_dim': 48
    },
    ('Hanabi-Small', 5, True): {
        'obs_dim': 341,
        'state_dim': 1685,
        'action_dim': 32
    },
    ('Hanabi-Very-Small', 5, True): {
        'obs_dim': 210,
        'state_dim': 1030,
        'action_dim': 28
    },
    ('Hanabi-Full', 5, False): {
        'obs_dim': 1285,
        'state_dim': 1385,
        'action_dim': 48
    },
    ('Hanabi-Full-Minimal', 5, False): {
        'obs_dim': 585,
        'state_dim': 685,
        'action_dim': 48
    },
    ('Hanabi-Small', 5, False): {
        'obs_dim': 341,
        'state_dim': 361,
        'action_dim': 32
    },
    ('Hanabi-Very-Small', 5, False): {
        'obs_dim': 210,
        'state_dim': 220,
        'action_dim': 28
    },
}


class HanabiExperiment(Experiment):

    def __init__(self, hanabi_name, num_agents, use_obs):
        self.env_cfg = Environment(type_="hanabi",
                                   args=dict(hanabi_name=hanabi_name,
                                             num_agents=num_agents,
                                             use_obs_instead_of_state=use_obs))
        self.__hanabi_name = hanabi_name
        self.__players = num_agents
        self.__use_obs = use_obs
        self.aws = 256
        self.eaws = 10
        self.pws = 16
        self.tws = 1

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            actors=TasksGroup(
                count=self.aws + self.eaws,
                scheduling=Scheduling.actor_worker_default(container_image="marl/marl-cpu-hanabi")),
            policies=TasksGroup(count=self.pws,
                                scheduling=Scheduling.policy_worker_default(gpu=0.12, mem=12000)),
            trainers=TasksGroup(count=self.tws, scheduling=Scheduling.trainer_worker_default(mem=50000)),
            eval_managers=TasksGroup(count=1, scheduling=Scheduling.eval_manager_default()))

    def initial_setup(self):
        policy_name = "default"
        players = self.__players
        policy = Policy(type_="actor-critic-separate",
                        args=dict(**HANABI_PLAYER_DIM_MAPPING[(self.__hanabi_name, self.__players,
                                                               self.__use_obs)],
                                  hidden_dim=64 * players,
                                  num_dense_layers=4,
                                  num_rnn_layers=0))
        # inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        eval_inference_stream = InferenceStream(type_=InferenceStream.Type.INLINE,
                                                stream_name=f"eval_{policy_name}",
                                                policy=policy,
                                                policy_name=policy_name,
                                                policy_identifier="evaluation",
                                                pull_interval_seconds=5)
        # sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
        # eval_sample_stream = SampleStream(type_=SampleStream.Type.NAMED_EVAL,
        #                                   stream_name=f"eval_{policy_name}")
        trainer = Trainer(type_="mappo",
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
                              bootstrap_steps=50,
                          ))
        return ExperimentConfig(
            actors=[
                ActorWorker(env=self.env_cfg,
                            inference_streams=[policy_name],
                            sample_streams=[policy_name],
                            agent_specs=[
                                AgentSpec(
                                    index_regex=".*",
                                    inference_stream_idx=0,
                                    sample_stream_idx=0,
                                    send_full_trajectory=False,
                                    send_after_done=False,
                                    sample_steps=400,
                                    bootstrap_steps=50,
                                )
                            ],
                            inference_splits=4,
                            max_num_steps=10000,
                            ring_size=8) for _ in range(self.aws)
            ] + [
                ActorWorker(
                    env=self.env_cfg,
                    inference_streams=[eval_inference_stream],
                    sample_streams=[f"eval_{policy_name}",
                                    SampleStream(type_=SampleStream.Type.NULL)],
                    agent_specs=[
                        AgentSpec(index_regex="0",
                                  inference_stream_idx=0,
                                  sample_stream_idx=0,
                                  send_full_trajectory=True,
                                  deterministic_action=True),
                        AgentSpec(index_regex=".*",
                                  inference_stream_idx=0,
                                  sample_stream_idx=1,
                                  send_full_trajectory=True,
                                  deterministic_action=True)
                    ],
                    max_num_steps=10000,
                    ring_size=1) for _ in range(self.eaws)
            ],
            policies=[
                PolicyWorker(policy_name=policy_name, inference_stream=policy_name, policy=policy)
                for _ in range(self.pws)
            ],
            trainers=[
                TrainerWorker(buffer_name='priority_queue',
                              buffer_args=dict(
                                  max_size=2,
                                  reuses=10,
                                  batch_size=500,
                              ),
                              policy_name=policy_name,
                              trainer=trainer,
                              policy=policy,
                              sample_stream=policy_name,
                              push_frequency_steps=1,
                              push_tag_frequency_minutes=10) for _ in range(self.tws)
            ],
            eval_managers=[
                EvaluationManager(eval_sample_stream=f"eval_{policy_name}",
                                  policy_name=policy_name,
                                  eval_tag="evaluation",
                                  eval_games_per_version=500)
            ],
        )


for k, v in HANABI_PLAYER_DIM_MAPPING.items():
    register_experiment(f"{k[0]}-{k[1]}{'-obs' if k[2] else ''}",
                        functools.partial(
                            HanabiExperiment,
                            hanabi_name=k[0],
                            num_agents=k[1],
                            use_obs=k[2],
                        ))
