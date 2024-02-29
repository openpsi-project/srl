from collections import defaultdict
import copy
import logging
import numpy as np
import queue

from .meta_solver import make_solver
from base.conditions import make as make_condition
from api.pbt import PopulationAlgorithm, register
from api.trainer import SampleBatch

logger = logging.getLogger("psro")


class PSRO(PopulationAlgorithm):

    def __init__(self,
                 meta_solver,
                 population,
                 training_policy_names,
                 initial_payoffs,
                 conditions,
                 num_iterations=10,
                 symmetric=False):
        # Population.
        self.num_players = len(population)
        self.symmetric = symmetric
        if self.symmetric:
            assert self.num_players == 1, (
                f"Number of players is {self.num_players}, this must be 1 for symmetric PSRO.")
            self.pop_sizes = np.array([len(population[0]), len(population[0])])
        else:
            assert self.num_players >= 2, (
                f"Number of players is {len(population)}, this must be at least 2 for asymmetric PSRO.")
            self.pop_sizes = np.array([len(pop) for pop in population])
        self.population = copy.deepcopy(population)
        self.training_policy_names = training_policy_names

        # PSRO.
        self.num_iteration = num_iterations
        self.payoffs = np.zeros((max(2, self.num_players), *(self.pop_sizes + num_iterations)))
        self.payoffs[tuple(np.indices(initial_payoffs.shape))] = initial_payoffs
        self.__meta_solver = make_solver(meta_solver)
        self.__meta_strategies = self.__meta_solver.solve(initial_payoffs)
        self.__conditions = [[make_condition(cfg) for cfg in conditions[i]] for i in range(self.num_players)]

        self.current_iteration = 0
        self.current_versions = [-1] * self.num_players
        self.__last_versions = [np.inf] * self.num_players
        self.__warm_up = [True] * self.num_players
        self.__condition_is_met = [False] * self.num_players
        self.__payoff_caches = [self._new_cache(i) for i in range(self.num_players)]
        self.__payoff_queues = [queue.Queue(10) for _ in range(self.num_players)]
        self.__num_eval_episodes = None

        # Workers.
        self.__actor_workers = None
        self.__eval_actor_workers = None
        self.__policy_workers = None
        self.__trainer_workers = None
        self.__eval_managers = None

    def configure(self, actors, policies, trainers, eval_managers):
        self.__actor_workers = [[] for _ in range(self.num_players)]
        self.__eval_actor_workers = [[] for _ in range(self.num_players)]
        self.__policy_workers = [[] for _ in range(self.num_players)]
        self.__trainer_workers = [[] for _ in range(self.num_players)]
        self.__eval_managers = [[] for _ in range(self.num_players)]

        for cfg in actors:
            player_idx = 0 if self.symmetric else cfg.worker_info.population_index
            if cfg.worker_info.worker_tag == "training":
                self.__actor_workers[player_idx].append(cfg)
            elif cfg.worker_info.worker_tag == "evaluation":
                self.__eval_actor_workers[player_idx].append(cfg)
            else:
                raise ValueError(f"Invalid actor worker tag: {cfg.worker_info.worker_tag}.")
        for cfg in policies:
            player_idx = 0 if self.symmetric else cfg.worker_info.population_index
            self.__policy_workers[player_idx].append(cfg)
        for cfg in trainers:
            player_idx = 0 if self.symmetric else cfg.worker_info.population_index
            self.__trainer_workers[player_idx].append(cfg)
        for cfg in eval_managers:
            player_idx = 0 if self.symmetric else cfg.worker_info.population_index
            self.__eval_managers[player_idx].append(cfg)

        self.__num_eval_episodes = eval_managers[0].eval_games_per_version * (
            self.pop_sizes[0] + 1) // np.prod(self.pop_sizes + 1)

    def step(self, sample: SampleBatch):
        self._evaluate(sample)
        if self._complete():
            self._solve()
            self._expand()
            self._log_info()
            return self._make_requests()
        else:
            return None

    def _log_info(self):
        logger.info(f"current iteration: {self.current_iteration}")
        logger.info(f"population: {self.population}")
        logger.info(f"training policy names: {self.training_policy_names}")
        sub_payoffs = self.payoffs[tuple(np.indices((max(2, self.num_players), *(self.pop_sizes))))]
        logger.info(f"current payoffs: {sub_payoffs}")
        logger.info(f"meta-strategies: {self.__meta_strategies}")

    def _evaluate(self, sample: SampleBatch):
        """Evaluate payoffs use samples.
        """
        player_idx, version, others_indices = self._parse(sample)
        if player_idx is None:
            return
        if version > self.current_versions[player_idx]:
            self._update_queue(player_idx)
            data = dict(version=version, mixed_payoff=self._mixed_payoff(player_idx))
            self.__condition_is_met[player_idx] = any(
                [condition.is_met_with(data) for condition in self.__conditions[player_idx]])
            self.current_versions[player_idx] = version
            self.__payoff_caches[player_idx] = self._new_cache(player_idx)
        self.__payoff_caches[player_idx][others_indices].append(sample.info["episode_return"][-1, :, 0])

    def _complete(self):
        """Check if the current PSRO iteration is completed and update payoff tensor.
           For each player: is completed if ANY condition of this player is met.
           For all players: is completed if ALL players complete.
        """
        if all(self.__condition_is_met):
            if self.symmetric:
                player_payoffs = np.mean(self.__payoff_queues[0].queue, axis=0)
                self.payoffs[:, self.pop_sizes[0], :self.pop_sizes[0]] = player_payoffs[:, 0, :-1]
                self.payoffs[:, :self.pop_sizes[0], self.pop_sizes[0]] = np.flip(player_payoffs[:, 0, :-1], 0)
                self.payoffs[:, self.pop_sizes[0], self.pop_sizes[0]] = np.mean(player_payoffs[:, 0, -1])
            else:
                for player_idx in range(self.num_players):
                    player_payoffs = np.mean(self.__payoff_queues[player_idx].queue, axis=0)
                    grid = [range(self.num_players)]
                    grid.extend(
                        [n if i == player_idx else range(n + 1) for i, n in enumerate(self.pop_sizes)])
                    indices = tuple(np.meshgrid(*grid, indexing="ij"))
                    self.payoffs[indices] = player_payoffs
            return True
        else:
            return False

    def _solve(self):
        """Calculate the meta-strategy for each player.
        """
        sub_payoffs = self.payoffs[tuple(np.indices((max(2, self.num_players), *(self.pop_sizes + 1))))]
        self.__meta_strategies = self.__meta_solver.solve(sub_payoffs)

    def _expand(self):
        """Add a new policy to each player's population and start a new PSRO iteration.
        """
        self.current_iteration += 1
        self.__last_versions = self.current_versions

        self.pop_sizes += 1
        for idx in range(self.num_players):
            self.population[idx].append(self.training_policy_names[idx])
            if self.symmetric:
                self.training_policy_names[idx] = f"policy_{self.pop_sizes[idx]}"
            else:
                self.training_policy_names[idx] = f"player{idx}_{self.pop_sizes[idx]}"
            for condition in self.__conditions[idx]:
                condition.reset()

        self.current_versions = [-1] * self.num_players
        self.__warm_up = [True] * self.num_players
        self.__condition_is_met = [False] * self.num_players
        self.__payoff_caches = [self._new_cache(i) for i in range(self.num_players)]
        self.__payoff_queues = [queue.Queue(10) for _ in range(self.num_players)]

    def _make_requests(self):
        if self.current_iteration < self.num_iteration:
            requests = {command: defaultdict(list) for command in ["pause", "reconfigure", "start"]}
            for idx in range(self.num_players):
                for cfg in self.__actor_workers[idx]:
                    worker_name = f"{cfg.worker_info.worker_type}/{cfg.worker_info.worker_index}"
                    if self.symmetric:
                        worker_kwargs = dict(
                            inference_stream_idx=1,
                            inference_stream_kwargs=dict(
                                population=self.population[0],
                                policy_sample_probs=self.__meta_strategies[1],
                            ),
                        )
                    else:
                        worker_kwargs = dict(
                            inference_stream_idx=[i for i in range(self.num_players) if i != idx],
                            inference_stream_kwargs=[
                                dict(
                                    population=self.population[i],
                                    policy_sample_probs=self.__meta_strategies[i],
                                ) for i in range(self.num_players) if i != idx
                            ],
                        )
                    requests["pause"]["worker_names"].append(worker_name)
                    requests["reconfigure"]["worker_names"].append(worker_name)
                    requests["reconfigure"]["worker_kwargs"].append(worker_kwargs)
                    requests["start"]["worker_names"].append(worker_name)
                for cfg in self.__eval_actor_workers[idx]:
                    worker_name = f"{cfg.worker_info.worker_type}/{cfg.worker_info.worker_index}"
                    if self.symmetric:
                        worker_kwargs = dict(
                            inference_stream_idx=1,
                            inference_stream_kwargs=dict(
                                population=self.population[0] + [self.training_policy_names[0]],
                                policy_sample_probs=np.ones(self.pop_sizes[0] + 1) / (self.pop_sizes[0] + 1),
                            ),
                        )
                    else:
                        worker_kwargs = dict(
                            inference_stream_idx=[i for i in range(self.num_players) if i != idx],
                            inference_stream_kwargs=[
                                dict(
                                    population=self.population[i] + [self.training_policy_names[i]],
                                    policy_sample_probs=np.ones(self.pop_sizes[i] + 1) /
                                    (self.pop_sizes[i] + 1),
                                ) for i in range(self.num_players) if i != idx
                            ],
                        )
                    requests["pause"]["worker_names"].append(worker_name)
                    requests["reconfigure"]["worker_names"].append(worker_name)
                    requests["reconfigure"]["worker_kwargs"].append(worker_kwargs)
                    requests["start"]["worker_names"].append(worker_name)
                for cfg in self.__policy_workers[idx]:
                    worker_name = f"{cfg.worker_info.worker_type}/{cfg.worker_info.worker_index}"
                    worker_kwargs = dict(policy_name=self.training_policy_names[idx])
                    requests["pause"]["worker_names"].append(worker_name)
                    requests["reconfigure"]["worker_names"].append(worker_name)
                    requests["reconfigure"]["worker_kwargs"].append(worker_kwargs)
                    requests["start"]["worker_names"].append(worker_name)
                for cfg in self.__trainer_workers[idx]:
                    worker_name = f"{cfg.worker_info.worker_type}/{cfg.worker_info.worker_index}"
                    worker_kwargs = dict(
                        policy_name=self.training_policy_names[idx],
                        trainer_config=cfg.trainer,
                    )
                    requests["pause"]["worker_names"].append(worker_name)
                    requests["reconfigure"]["worker_names"].append(worker_name)
                    requests["reconfigure"]["worker_kwargs"].append(worker_kwargs)
                    requests["start"]["worker_names"].append(worker_name)
                for cfg in self.__eval_managers[idx]:
                    worker_name = f"{cfg.worker_info.worker_type}/{cfg.worker_info.worker_index}"
                    worker_kwargs = dict(
                        policy_name=self.training_policy_names[idx],
                        eval_games_per_version=self.__num_eval_episodes * np.prod(self.pop_sizes + 1) //
                        (self.pop_sizes[idx] + 1),
                    )
                    requests["pause"]["worker_names"].append(worker_name)
                    requests["reconfigure"]["worker_names"].append(worker_name)
                    requests["reconfigure"]["worker_kwargs"].append(worker_kwargs)
                    requests["start"]["worker_names"].append(worker_name)
        else:
            requests = {command: defaultdict(list) for command in ["exit", "self_exit"]}
            for idx in range(self.num_players):
                for cfg in (self.__actor_workers[idx] + self.__eval_actor_workers[idx] +
                            self.__policy_workers[idx] + self.__trainer_workers[idx] +
                            self.__eval_managers[idx]):
                    worker_name = f"{cfg.worker_info.worker_type}/{cfg.worker_info.worker_index}"
                    requests["exit"]["worker_names"].append(worker_name)
            requests["self_exit"] = dict()

        return requests

    def _parse(self, sample: SampleBatch):
        # sample: num_steps x num_players x ...
        policy_names = [sample[:, i].unique_of("policy_name") for i in range(max(2, self.num_players))]
        versions = [sample[:, i].unique_of("policy_version_steps") for i in range(max(2, self.num_players))]
        if None in policy_names or None in versions:
            logger.debug(f"Non-unique policy name {policy_names} or version {versions}, samples are ignored.")
            return None, None, None
        if len(np.intersect1d(self.training_policy_names, policy_names)) == 0:
            logger.debug(f"No training policy name found. This is possible during warmup. Sample policy "
                         f"names: {policy_names}, training policy names: {self.training_policy_names}. "
                         f"samples are ignored.")
            return None, None, None

        policy_name = np.random.choice(np.intersect1d(self.training_policy_names, policy_names))
        player_idx = 0 if self.symmetric else self.training_policy_names.index(policy_name)
        version = versions[player_idx]
        if self.symmetric:
            others_indices = (self.population[0] + [self.training_policy_names[0]]).index(policy_names[1])
        else:
            others_indices = tuple(
                (self.population[i] + [self.training_policy_names[i]]).index(policy_names[i])
                for i in range(self.num_players) if i != player_idx)
        if self.__warm_up[player_idx]:
            if version >= self.__last_versions[player_idx]:
                logger.debug(f"Received samples of player {player_idx}'s previous policy during warmup, "
                             f"samples are ignored.")
                return None, None, None
            else:
                self.current_versions[player_idx] = version
                self.__warm_up[player_idx] = False

        return player_idx, version, others_indices

    def _new_cache(self, player_idx):
        shape = tuple(n for i, n in enumerate(self.pop_sizes + 1) if i != player_idx)
        cache = np.empty(shape, dtype=object)
        for indices in np.ndindex(shape):
            cache[indices] = []
        return cache

    def _update_queue(self, player_idx):
        shape = self.__payoff_caches[player_idx].shape
        current_payoffs = np.zeros((*shape, max(2, self.num_players)))
        for indices in np.ndindex(shape):
            cache = self.__payoff_caches[player_idx][indices]
            if len(cache) == 0:
                logger.warning(f"No samples for player {player_idx} with {indices} when updating queue.")
                continue
            current_payoffs[indices] = np.mean(cache, axis=0)
        current_payoffs = np.expand_dims(current_payoffs, axis=player_idx)
        current_payoffs = np.moveaxis(current_payoffs, -1, 0)

        if self.__payoff_queues[player_idx].full():
            self.__payoff_queues[player_idx].get()
        self.__payoff_queues[player_idx].put(current_payoffs)

    def _mixed_payoff(self, player_idx):
        mixed_payoff = self.__payoff_queues[player_idx].queue[-1][player_idx]
        for i, meta_strategy in enumerate(self.__meta_strategies):
            if i == player_idx:
                mixed_payoff = mixed_payoff[0]
            else:
                mixed_payoff = np.tensordot(meta_strategy, mixed_payoff[:-1], axes=1)
        return mixed_payoff


register("psro", PSRO)
