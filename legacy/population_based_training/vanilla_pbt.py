from collections import defaultdict
import copy
import logging
import numpy as np

from api.pbt import PopulationAlgorithm, register
from api.trainer import SampleBatch

logger = logging.getLogger("vanilla_pbt")


class VanillaPBT(PopulationAlgorithm):

    def __init__(self, population, ready_interval, truncation_ratio, explore_configs):
        self.population = np.array(population)
        self.__ready_interval = ready_interval
        self.__num_truncation = int(truncation_ratio * len(population))
        self.__explore_configs = explore_configs
        self.__last_versions = np.zeros(len(population))
        self.__current_versions = np.zeros(len(population))
        self.__return_caches = [[] for _ in range(len(population))]
        self.__average_returns = np.zeros(len(population))

        self.ready_policies = None
        self.src_policies = None
        self.dst_policies = None

        self.__policy_workers = None
        self.__trainer_workers = None
        self.__eval_managers = None

    def configure(self, actors, policies, trainers, eval_managers):
        self.__policy_workers = defaultdict(list)
        self.__trainer_workers = defaultdict(list)
        self.__eval_managers = defaultdict(list)
        for cfg in policies:
            self.__policy_workers[cfg.policy_name].append(cfg)
        for cfg in trainers:
            self.__trainer_workers[cfg.policy_name].append(cfg)
        for cfg in eval_managers:
            self.__eval_managers[cfg.policy_name].append(cfg)

    def step(self, sample: SampleBatch):
        self._evaluate(sample)
        if self._ready() and self._exploit() != 0:
            self._explore()
            self._log_info()
            return self._make_requests()
        else:
            return None

    def _log_info(self):
        logger.info(f"current versions: {self.__current_versions}")
        logger.info(f"average returns: {self.__average_returns}")
        logger.info(f"dst policies: {self.dst_policies}")
        logger.info(f"src policies: {self.src_policies}")

    def _evaluate(self, sample: SampleBatch):
        policy_name = sample.unique_of("policy_name")
        version = sample.unique_of("policy_version_steps")
        if policy_name is None or version is None:
            logger.debug(f"Non-unique policy name {policy_name} or version {version}, samples are ignored.")
            return

        policy_idx = np.where(self.population == policy_name)[0][0]
        if version < self.__current_versions[policy_idx]:
            logger.debug(f"Received samples of policy {policy_idx}'s previous version: {version} < "
                         f"{self.__current_versions[policy_idx]}, samples are ignored.")
            return
        if version > self.__current_versions[policy_idx]:
            if self.__return_caches[policy_idx]:
                self.__average_returns[policy_idx] = np.mean(self.__return_caches[policy_idx])
            else:
                logger.info(f"No samples for policy {policy_idx} of version "
                            f"{self.__current_versions[policy_idx]}, this is normal during warmup.")
            self.__current_versions[policy_idx] = version
            self.__return_caches[policy_idx] = []

        self.__return_caches[policy_idx].append(sample.info["episode_return"][-1, 0])

    def _ready(self):
        is_ready = (self.__current_versions - self.__last_versions >= self.__ready_interval)
        self.__last_versions[is_ready] = self.__current_versions[is_ready]
        self.ready_policies = self.population[is_ready]
        return any(is_ready)

    def _exploit(self):
        sort_indices = np.argsort(self.__average_returns)
        top_policies = self.population[sort_indices][-self.__num_truncation:]
        bottom_policies = self.population[sort_indices][:self.__num_truncation]
        self.dst_policies = np.intersect1d(self.ready_policies, bottom_policies)
        self.src_policies = np.random.choice(top_policies, len(self.dst_policies))
        return len(self.dst_policies)

    def _explore(self):
        for src_name, dst_name in zip(self.src_policies, self.dst_policies):
            src_trainer = self.__trainer_workers[src_name][0].trainer
            dst_trainer = copy.deepcopy(src_trainer)
            for kwargs in self.__explore_configs:
                if kwargs["method"] == "perturb":
                    dst_trainer = self._perturb(dst_trainer, **kwargs)
                elif kwargs["method"] == "resample":
                    dst_trainer = self._resample(dst_trainer, **kwargs)
                else:
                    raise ValueError(f"Invalid explore method {kwargs['method']}. Use perturb or resample.")
            for cfg in self.__trainer_workers[dst_name]:
                cfg.trainer = dst_trainer

    def _make_requests(self):
        requests = {command: defaultdict(list) for command in ["pause", "reconfigure", "start"]}
        for src_name, dst_name in zip(self.src_policies, self.dst_policies):
            for cfg in self.__policy_workers[dst_name]:
                worker_name = f"{cfg.worker_info.worker_type}/{cfg.worker_info.worker_index}"
                requests["pause"]["worker_names"].append(worker_name)
                requests["start"]["worker_names"].append(worker_name)
            for cfg in self.__trainer_workers[dst_name]:
                worker_name = f"{cfg.worker_info.worker_type}/{cfg.worker_info.worker_index}"
                requests["pause"]["worker_names"].append(worker_name)
                requests["reconfigure"]["worker_names"].append(worker_name)
                requests["reconfigure"]["worker_kwargs"].append(
                    dict(trainer_config=cfg.trainer, src_policy_name=src_name))
                requests["start"]["worker_names"].append(worker_name)
            for cfg in self.__eval_managers[dst_name]:
                worker_name = f"{cfg.worker_info.worker_type}/{cfg.worker_info.worker_index}"
                requests["pause"]["worker_names"].append(worker_name)
                requests["start"]["worker_names"].append(worker_name)
        return requests

    def _perturb(self, trainer, keys, factors, min_value=None, max_value=None, **kwargs):
        if type(keys) is str:
            value = trainer.args[keys] * np.random.choice(factors)
            if min_value is not None:
                value = max(value, min_value)
            if max_value is not None:
                value = min(value, max_value)
            trainer.args[keys] = value
        elif type(keys) is list and len(keys) == 2:
            value = trainer.args[keys[0]][keys[1]] * np.random.choice(factors)
            if min_value is not None:
                value = max(value, min_value)
            if max_value is not None:
                value = min(value, max_value)
            trainer.args[keys[0]][keys[1]] = value
        else:
            raise ValueError(f"Invalid keys {keys}. keys must be str or list of length 2.")
        return trainer

    def _resample(self, trainer, keys, distribution, values=None, value_range=None, **kwargs):
        if distribution == "categorical":
            assert values is not None, "values must be set for categorical distribution."
            value = np.random.choice(values)
        elif distribution == "uniform":
            assert value_range is not None, "value_range must be set for uniform distribution."
            value = np.random.uniform(value_range[0], value_range[1])
        elif distribution == "log_uniform":
            assert value_range is not None, "value_range must be set for log_uniform distribution."
            value = np.exp(np.random.uniform(np.log(value_range[0]), np.log(value_range[1])))
        else:
            raise ValueError(f"Invalid distribution {distribution}. Use categorical, uniform, or "
                             f"log_uniform.")

        if type(keys) is str:
            trainer.args[keys] = value
        elif type(keys) is list and len(keys) == 2:
            trainer.args[keys[0]][keys[1]] = value
        else:
            raise ValueError(f"Invalid keys {keys}. keys must be str or list of length 2.")
        return trainer


register("vanilla_pbt", VanillaPBT)
