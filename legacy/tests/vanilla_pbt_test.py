import numpy as np
import unittest

from legacy.population_based_training.vanilla_pbt import VanillaPBT
from api.trainer import SampleBatch
from base.namedarray import NamedArray
import api.config


class TestEpisodeInfo(NamedArray):

    def __init__(self, episode_return: np.ndarray = np.zeros(1, dtype=np.float32)):
        super(TestEpisodeInfo, self).__init__(episode_return=episode_return)


def make_sample(policy_name, version, episode_return):
    num_steps = 10
    return SampleBatch(
        obs=np.random.random(size=(num_steps, 10)),
        reward=np.random.random((num_steps, 1)),
        policy_name=np.full(shape=(num_steps, 1), fill_value=policy_name),
        policy_version_steps=np.full(shape=(num_steps, 1), fill_value=version),
        info=TestEpisodeInfo(episode_return=np.full(shape=(num_steps, 1), fill_value=episode_return)),
        on_reset=None,
        info_mask=None,
    )


def make_vanilla_pbt(population, ready_interval, truncation_ratio):
    return VanillaPBT(population=population,
                      ready_interval=ready_interval,
                      truncation_ratio=truncation_ratio,
                      explore_configs=[
                          dict(keys="eps_clip",
                               method="resample",
                               distribution="categorical",
                               values=[0.1, 0.2, 0.3])
                      ])


def make_workers(pop_size):
    actors = []
    policies = []
    trainers = []
    eval_managers = []
    for i in range(pop_size):
        actors.extend([
            api.config.ActorWorker(env=None,
                                   sample_streams=None,
                                   inference_streams=None,
                                   agent_specs=None,
                                   worker_info=api.config.WorkerInformation()),
        ])
        policies.extend([
            api.config.PolicyWorker(policy_name=f"policy_{i}",
                                    inference_stream=None,
                                    policy=None,
                                    worker_info=api.config.WorkerInformation()),
        ])
        trainers.extend([
            api.config.TrainerWorker(policy_name=f"policy_{i}",
                                     trainer=api.config.Trainer(type_="mappo", args=dict(eps_clip=0.2)),
                                     policy=None,
                                     sample_stream=None,
                                     worker_info=api.config.WorkerInformation()),
        ])
        eval_managers.extend([
            api.config.EvaluationManager(policy_name=f"policy_{i}",
                                         eval_sample_stream=None,
                                         worker_info=api.config.WorkerInformation()),
        ])
    return actors, policies, trainers, eval_managers


class VanillaPBTTest(unittest.TestCase):

    def test_vanilla_pbt(self):
        pop_size = 2
        population = [f"policy_{i}" for i in range(pop_size)]
        ready_interval = 100
        truncation_ratio = 0.5

        vanilla_pbt = make_vanilla_pbt(population, ready_interval, truncation_ratio)
        vanilla_pbt.configure(*make_workers(pop_size))

        # Before version is larger than last_ready_version + ready_interval, make no requests.
        episode_return = [1, 0]
        for version in range(ready_interval):
            for i in range(pop_size):
                requests = vanilla_pbt.step(make_sample(population[i], version, episode_return[i]))
                self.assertIsNone(requests)

        # policy_0 is ready but has better performance, make no requests.
        version = ready_interval
        requests = vanilla_pbt.step(make_sample(population[0], version, episode_return[0]))
        self.assertIsNone(requests)
        self.assertEqual(vanilla_pbt.ready_policies, ["policy_0"])

        # policy_1 is ready and has worse performance, make requests.
        requests = vanilla_pbt.step(make_sample(population[1], version, episode_return[1]))
        self.assertIsNotNone(requests)
        self.assertEqual(vanilla_pbt.ready_policies, ["policy_1"])
        self.assertEqual(vanilla_pbt.src_policies, ["policy_0"])
        self.assertEqual(vanilla_pbt.dst_policies, ["policy_1"])


if __name__ == '__main__':
    unittest.main()
