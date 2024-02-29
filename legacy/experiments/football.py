import functools
import math
import itertools

from api.config import *
from api.config import ExperimentConfig, ExperimentScheduling

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


class FootballMiniExperiment(Experiment):

    def __init__(self):
        self.aws, self.pws, self.tws = 10, 1, 1

    def scheduling_setup(self) -> ExperimentScheduling:
        ExperimentScheduling(
            actors=TasksGroup(
                count=10,
                scheduling=Scheduling.actor_worker_default(container_image="marl/marl-cpu-football")),
            policies=TasksGroup(count=10, scheduling=Scheduling.policy_worker_default()),
            trainers=TasksGroup(count=10, scheduling=Scheduling.trainer_worker_default()),
        )

    def initial_setup(self):
        policy_name = "default"
        policy = Policy(type_="football-simple115-separate")
        return ExperimentConfig(
            actors=[
                ActorWorker(
                    inference_streams=[policy_name],
                    sample_streams=[policy_name],
                    env=Environment(type_="football",
                                    args=dict(
                                        env_name="academy_3_vs_1_with_keeper",
                                        number_of_left_players_agent_controls=3,
                                        number_of_right_players_agent_controls=0,
                                        representation="simple115v2",
                                    )),
                    agent_specs=[AgentSpec(
                        index_regex=".*",
                        inference_stream_idx=0,
                        sample_stream_idx=0,
                    )],
                ) for _ in range(self.aws)
            ],
            policies=[
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=policy_name,
                    policy=policy,
                ) for _ in range(self.pws)
            ],
            trainers=[
                TrainerWorker(
                    policy_name=policy_name,
                    trainer="null_trainer",
                    policy=policy,
                    sample_stream=policy_name,
                ) for _ in range(self.tws)
            ],
        )


class FootballExperiment(Experiment):

    def __init__(
            self,
            env_name: str,
            left_players: int,
            right_players: int,
            representation: str,
            multiplier: int,  # aka scale
            algo_name: str,
            ppo_epochs=32,
            ppo_iterations=1,
            ppg_epochs: int = 6,
            seed: int = 0):
        self.env_name = env_name
        self.left_players = left_players
        self.right_players = right_players
        self.representation = representation
        self.algo_name = algo_name
        self.ppg_epochs = ppg_epochs
        self.seed = seed
        self.scale = multiplier
        self.policy_name = policy_name = "default"
        self.num_actors = int(40 * multiplier if self.algo_name == "mappg" else 160 * multiplier)
        self.num_policies = max(1, int(max(self.left_players + self.right_players, 5) // 5 * multiplier))
        self.num_trainer = int(
            math.ceil(max(self.left_players + self.right_players, 3) // 3 * multiplier / 16))
        # self.num_trainer = max(self.left_players + self.right_players, 3) // 3 * 1 * multiplier
        self.num_eval_actors = 40
        self.ring_size = 8
        self.inference_splits = 4

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
                                   bootstrap_steps=50,
                               ))
        _policy_type = "football-smm-separate" if self.representation == "extracted" else "football-simple115-separate"
        self.policy = Policy(type_=_policy_type,
                             args=dict(num_rnn_layers=1,
                                       chunk_len=10,
                                       rnn_type="gru",
                                       auxiliary_head=bool(algo_name == "mappg"),
                                       seed=seed))

        self.inference_stream = InferenceStream(type_=InferenceStream.Type.NAME, stream_name=policy_name)
        self.sample_stream = SampleStream(type_=SampleStream.Type.NAME, stream_name=policy_name)
        self.eval_inference_stream = InferenceStream(
            type_=InferenceStream.Type.INLINE,
            stream_name=f"eval_{policy_name}",
            policy=self.policy,
            policy_name=policy_name,
            policy_identifier='eval',
        )
        self.eval_sample_stream = SampleStream(type_=SampleStream.Type.NAME,
                                               stream_name=f"eval_{policy_name}")

        if self.algo_name == "mappg":
            if self.representation == "extracted":
                self.train_batch_size = int(25 * multiplier)
            else:
                self.train_batch_size = int(50 * multiplier)
        else:
            self.train_batch_size = int(160 * multiplier)

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
        gpu = 0.5 if self.representation == "extracted" else 0.12
        return ExperimentScheduling(actors=TasksGroup(count=self.num_actors + self.num_eval_actors,
                                                      scheduling=Scheduling.actor_worker_default(
                                                          container_image="marl/marl-cpu-football",
                                                          cpu=1,
                                                          mem=1024,
                                                      )),
                                    policies=TasksGroup(count=self.num_policies,
                                                        scheduling=Scheduling.policy_worker_default(
                                                            gpu=gpu,
                                                            cpu=4,
                                                            mem=10 * 1024,
                                                            node_type=['g1', 'g2'],
                                                            container_image='marl/marl-gpu-blosc',
                                                        )),
                                    trainers=TasksGroup(
                                        count=self.num_trainer,
                                        scheduling=Scheduling.trainer_worker_default(
                                            cpu=4,
                                            mem=60 * 1024,
                                            node_type=['g1', 'g2', 'g8'] if self.scale < 16 else ['g8'],
                                            container_image='marl/marl-gpu-blosc',
                                        )),
                                    eval_managers=TasksGroup(count=1,
                                                             scheduling=Scheduling.eval_manager_default()))

    def initial_setup(self):
        return ExperimentConfig(
            actors=[
                ActorWorker(  # Training.
                    env=[
                        self.train_env(env_seed=self.seed * 12023 + i * 97 + j) for j in range(self.ring_size)
                    ],
                    inference_streams=[self.inference_stream],
                    sample_streams=[self.sample_stream],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            sample_steps=200,
                            bootstrap_steps=50,
                            send_after_done=False,
                            send_full_trajectory=False,
                        )
                    ],
                    ring_size=self.ring_size,
                    inference_splits=self.inference_splits,
                ) for i in range(self.num_actors)
            ] + [
                ActorWorker(  # Evaluation
                    env=[
                        self.eval_env(env_seed=self.seed * 97 + 12023 + i * 101 + j)
                        for j in range(self.ring_size)
                    ],
                    inference_streams=[self.eval_inference_stream],
                    sample_streams=[self.eval_sample_stream,
                                    SampleStream(type_=SampleStream.Type.NULL)],
                    agent_specs=[
                        AgentSpec(
                            index_regex="0",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                            send_full_trajectory=True,
                            send_concise_info=True,
                            deterministic_action=True,
                        ),
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=1,
                        ),
                    ],
                    ring_size=1,
                    inference_splits=1,
                ) for i in range(self.num_eval_actors)
            ],
            policies=[
                PolicyWorker(
                    policy_name=self.policy_name,
                    inference_stream=self.inference_stream,
                    pull_max_failures=1000,
                    pull_frequency_seconds=0.01,
                    max_inference_delay=0.05,
                    policy=self.policy,
                ) for _ in range(self.num_policies)
            ],
            trainers=[
                TrainerWorker(
                    buffer_name='priority_queue',
                    buffer_args=dict(max_size=6,
                                     reuses=1 if self.algo_name == "mappg" else 5,
                                     batch_size=self.train_batch_size // self.num_trainer),
                    policy_name=self.policy_name,
                    trainer=self.trainer,
                    policy=self.policy,
                    sample_stream=self.sample_stream,
                    train_for_seconds=8 * 3600,
                ) for _ in range(self.num_trainer)
            ],
            eval_managers=[
                EvaluationManager(
                    eval_sample_stream=self.eval_sample_stream,
                    policy_name=self.policy_name,
                    eval_games_per_version=100,
                )
            ])


@dataclasses.dataclass
class FootballQmixExperiment(Experiment):
    env_name: str
    seed: int
    scale: int

    mixer_type: str = 'qmix'
    bootstrap_steps: int = 5

    burn_in_steps: int = 40
    chunk_len: int = 100
    use_priority_weight: bool = False

    def __post_init__(self):
        self.num_agents = map_agent_registry[self.env_name][0]

        multiplier = self.scale
        self.num_actors = int(160 * multiplier)
        self.num_policies = max(1, int(max(self.num_agents, 5) // 5 * multiplier))
        self.num_trainer = int(math.ceil(max(self.num_agents, 3) // 3 * multiplier / 16))
        self.num_eval_actors = 40
        self.ring_size = 8
        self.inference_splits = 4

        self.sample_steps = 200

        self.batch_size = int(160 * multiplier) // self.num_agents

    def scheduling_setup(self) -> ExperimentScheduling:
        gpu = 0.12
        return ExperimentScheduling(actors=TasksGroup(count=self.num_actors + self.num_eval_actors,
                                                      scheduling=Scheduling.actor_worker_default(
                                                          container_image="marl/marl-cpu-football",
                                                          cpu=1,
                                                          mem=1024,
                                                      )),
                                    policies=TasksGroup(count=self.num_policies,
                                                        scheduling=Scheduling.policy_worker_default(
                                                            gpu=gpu,
                                                            cpu=4,
                                                            mem=10 * 1024,
                                                            node_type=['g1', 'g2'],
                                                            container_image='marl/marl-gpu-blosc',
                                                        )),
                                    trainers=TasksGroup(count=self.num_trainer,
                                                        scheduling=Scheduling.trainer_worker_default(
                                                            cpu=4,
                                                            mem=240 * 1024,
                                                            node_type=['g1', 'g2'],
                                                            container_image='marl/marl-gpu-blosc',
                                                        )),
                                    eval_managers=TasksGroup(count=1,
                                                             scheduling=Scheduling.eval_manager_default()))

    def make_env(self, seed, is_eval):
        return Environment(type_="football",
                           args=dict(
                               env_name=self.env_name,
                               number_of_left_players_agent_controls=self.num_agents,
                               number_of_right_players_agent_controls=0,
                               representation="simple115v2",
                               rewards=("scoring,checkpoints" if not is_eval else "scoring"),
                               seed=seed,
                               shared=True,
                           ))

    def initial_setup(self) -> ExperimentConfig:
        policy_name = "default"

        seed = self.seed
        ring_size = self.ring_size
        inference_splits = self.inference_splits
        trainer = Trainer(
            type_="q-learning",
            args=dict(
                gamma=0.99,
                bootstrap_steps=self.bootstrap_steps,
                burn_in_steps=self.burn_in_steps,
                use_soft_update=False,
                hard_update_interval=200,
                max_grad_norm=40.0,
                value_loss='smoothl1',
                optimizer='adam',
                optimizer_config=dict(lr=5e-4, eps=1e-5),
                use_popart=False,
                use_priority_weight=self.use_priority_weight,
                priority_interpolation_eta=0.9,
            ),
        )
        policy = Policy(
            type_="football-qmix",
            args=dict(
                env_name=self.env_name,
                chunk_len=self.chunk_len,  # chunk length requires modification for different map
                mixer_type=self.mixer_type,
                use_double_q=True,
                epsilon_start=1.0,
                epsilon_finish=0.05,
                epsilon_anneal_time=5000,
                q_i_config=dict(hidden_dim=128, num_dense_layers=2, rnn_type="gru", num_rnn_layers=1),
                mixer_config=dict(
                    popart=False,
                    hidden_dim=64,
                    num_hypernet_layers=2,
                    hypernet_hidden_dim=64,
                ),
                state_use_all_local_obs=False,
                state_concate_all_local_obs=False,
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
        # buffer_config = dict(
        #     buffer_name='prioritized_replay_buffer',
        #     buffer_args=dict(
        #         max_size=int(2e6) // self.sample_steps,
        #         batch_size=self.batch_size // self.num_trainer,
        #         sample_length=self.burn_in_steps + self.sample_steps,
        #         burn_in_steps=self.burn_in_steps,
        #         warmup_transitions=int(5e4),
        #         seed=self.seed,
        #         alpha=0.6,
        #         beta=0.4,
        #         beta_scheduler=None,
        #         max_priority=1.0,
        #         priority_interpolation_eta=0.9,
        #     ),
        # )
        buffer_config = dict(
            buffer_name='simple_replay_buffer',
            buffer_args=dict(
                max_size=int(4e6) // self.sample_steps,
                batch_size=self.batch_size,
                sample_length=self.burn_in_steps + self.sample_steps,
                batch_length=self.burn_in_steps + self.sample_steps,
                warmup_transitions=max(self.batch_size * (self.burn_in_steps + self.sample_steps), int(5e4)),
                seed=self.seed,
            ),
        )

        return ExperimentConfig(
            actors=[
                ActorWorker(
                    env=[
                        self.make_env(seed=k * 1000 + seed * 97 + j, is_eval=False)
                        for j in range(self.ring_size)
                    ],
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
                            bootstrap_steps=0,
                            burn_in_steps=self.burn_in_steps,
                        )
                    ],
                    max_num_steps=2000,
                    ring_size=ring_size,
                    inference_splits=inference_splits,
                ) for k in range(self.num_actors)
            ] + [
                ActorWorker(  # Evaluation
                    env=[
                        self.make_env(is_eval=True, seed=(self.num_actors + i) * 1000 + seed * 97 + j)
                        for j in range(self.ring_size)
                    ],
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
                    ring_size=1,
                    inference_splits=1,
                ) for i in range(self.num_eval_actors)
            ],
            policies=[
                PolicyWorker(
                    policy_name=policy_name,
                    inference_stream=policy_name,
                    policy=policy,
                    max_inference_delay=0.05,
                    pull_max_failures=1000,
                    pull_frequency_seconds=0.01,
                ) for j in range(self.num_policies)
            ],
            trainers=[
                TrainerWorker(
                    **buffer_config,
                    policy_name=policy_name,
                    trainer=trainer,
                    push_frequency_seconds=0.01,
                    push_frequency_steps=5,
                    log_frequency_seconds=50,
                    train_for_seconds=8 * 3600,
                    policy=policy,
                    sample_stream=policy_name,
                ) for _ in range(self.num_trainer)
            ],
            eval_managers=[
                EvaluationManager(
                    eval_sample_stream=eval_sample_stream,
                    policy_name=policy_name,
                    eval_tag="eval",
                    eval_games_per_version=100,
                )
            ])


register_experiment("football-mini", FootballMiniExperiment)

for map, player_spec in map_agent_registry.items():
    left, right, _ = player_spec
    for i in range(1, left + 1):
        for j in range(right + 1):
            for m in range(1, 3):
                for representation in ["simple115v2", "extracted"]:
                    rep = "smm-" if representation == "extracted" else ""
                    seed = 1
                    ppo_exp_name = f"fb-{rep}{map.replace('_', '-')}-l{i}-r{j}-x{m}"
                    register_experiment(
                        ppo_exp_name,
                        functools.partial(FootballExperiment,
                                          env_name=map,
                                          left_players=i,
                                          right_players=j,
                                          representation=representation,
                                          multiplier=m,
                                          algo_name="mappo",
                                          seed=seed))

                    ppo_epochs_ = 1
                    ppo_iterations_ = 32
                    ppg_epochs_ = 5

                    ppg_exp_name = f"fb-ppg-{rep}{map.replace('_', '-')}-l{i}-r{j}-x{m}"
                    register_experiment(
                        ppg_exp_name,
                        functools.partial(FootballExperiment,
                                          env_name=map,
                                          left_players=i,
                                          right_players=j,
                                          representation=representation,
                                          multiplier=m,
                                          algo_name="mappg",
                                          ppo_epochs=ppo_epochs_,
                                          ppo_iterations=ppo_iterations_,
                                          ppg_epochs=ppg_epochs_,
                                          seed=seed))

seeds = list(range(1, 33))
scales = [0.5, 1, 2, 4, 8, 16, 32]
for map_, player_spec in map_agent_registry.items():
    left, _, _ = player_spec
    if map_ == "academy_corner":
        game = 'Corner'
    elif map_ == 'academy_counterattack_easy':
        game = 'CAeasy'
    elif map_ == 'academy_counterattack_hard':
        game = 'CAhard'
    elif map_ == "academy_3_vs_1_with_keeper":
        game = "3v1"
    else:
        continue
    for seed, scale in itertools.product(seeds, scales):
        register_experiment(
            f"{game}-ppo-s{seed}-x{scale}",
            functools.partial(FootballExperiment,
                              env_name=map_,
                              left_players=left,
                              right_players=0,
                              representation='simple115v2',
                              multiplier=scale,
                              algo_name='mappo',
                              seed=seed))
        register_experiment(f"{game}-qmix-s{seed}-x{scale}",
                            functools.partial(FootballQmixExperiment, env_name=map_, seed=seed, scale=scale))
        register_experiment(
            f"{game}-vdn-s{seed}-x{scale}",
            functools.partial(FootballQmixExperiment, env_name=map_, seed=seed, scale=scale,
                              mixer_type='vdn'))
