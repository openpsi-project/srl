import numpy as np
import unittest

from legacy.population_based_training.psro import PSRO
from api.trainer import SampleBatch
from base.namedarray import NamedArray
import api.config


class TestEpisodeInfo(NamedArray):

    def __init__(self, episode_return: np.ndarray = np.zeros(1, dtype=np.float32)):
        super(TestEpisodeInfo, self).__init__(episode_return=episode_return)


def make_ma_sample(policy_names, versions, episode_returns):
    num_players = 2
    num_steps = 10
    return SampleBatch(
        obs=np.random.random(size=(num_steps, num_players, 10)),
        reward=np.random.random((num_steps, num_players, 1)),
        policy_name=np.stack(
            [np.full(shape=(num_steps, 1), fill_value=policy_names[i]) for i in range(num_players)], axis=1),
        policy_version_steps=np.stack(
            [np.full(shape=(num_steps, 1), fill_value=versions[i]) for i in range(num_players)], axis=1),
        info=TestEpisodeInfo(episode_return=np.stack(
            [np.full(shape=(num_steps, 1), fill_value=episode_returns[i]) for i in range(num_players)],
            axis=1)),
        on_reset=None,
        info_mask=None,
    )


def make_psro(pop_sizes, max_versions, symmetric):
    if symmetric:
        return PSRO(meta_solver=api.config.MetaSolver(type_=api.config.MetaSolver.Type.UNIFORM),
                    population=[[f"policy_{i}" for i in range(pop_sizes)]],
                    training_policy_names=[f"policy_{pop_sizes}"],
                    initial_payoffs=np.zeros((2, pop_sizes, pop_sizes)),
                    conditions=[[
                        api.config.Condition(type_=api.config.Condition.Type.SimpleBound,
                                             args=dict(field="version", lower_limit=max_versions))
                    ]],
                    num_iterations=10,
                    symmetric=True)
    else:
        return PSRO(meta_solver=api.config.MetaSolver(type_=api.config.MetaSolver.Type.UNIFORM),
                    population=[[f"player0_{i}" for i in range(pop_sizes[0])],
                                [f"player1_{j}" for j in range(pop_sizes[1])]],
                    training_policy_names=[f"player0_{pop_sizes[0]}", f"player1_{pop_sizes[1]}"],
                    initial_payoffs=np.zeros((2, *pop_sizes)),
                    conditions=[
                        [
                            api.config.Condition(type_=api.config.Condition.Type.SimpleBound,
                                                 args=dict(field="version", lower_limit=max_versions[0]))
                        ],
                        [
                            api.config.Condition(type_=api.config.Condition.Type.SimpleBound,
                                                 args=dict(field="version", lower_limit=max_versions[1]))
                        ],
                    ],
                    num_iterations=10,
                    symmetric=False)


def make_workers(num_players):
    actors = []
    policies = []
    trainers = []
    eval_managers = []
    for i in range(num_players):
        actors.extend([
            api.config.ActorWorker(env=None,
                                   sample_streams=None,
                                   inference_streams=None,
                                   agent_specs=None,
                                   worker_info=api.config.WorkerInformation(population_index=i,
                                                                            worker_tag="training")),
            api.config.ActorWorker(env=None,
                                   sample_streams=None,
                                   inference_streams=None,
                                   agent_specs=None,
                                   worker_info=api.config.WorkerInformation(population_index=i,
                                                                            worker_tag="evaluation")),
        ])
        policies.extend([
            api.config.PolicyWorker(policy_name=None,
                                    inference_stream=None,
                                    policy=None,
                                    worker_info=api.config.WorkerInformation(population_index=i)),
        ])
        trainers.extend([
            api.config.TrainerWorker(policy_name=None,
                                     trainer=None,
                                     policy=None,
                                     sample_stream=None,
                                     worker_info=api.config.WorkerInformation(population_index=i)),
        ])
        eval_managers.extend([
            api.config.EvaluationManager(policy_name=None,
                                         eval_sample_stream=None,
                                         eval_games_per_version=20,
                                         worker_info=api.config.WorkerInformation(population_index=i)),
        ])
    return actors, policies, trainers, eval_managers


class PSROTest(unittest.TestCase):

    def test_symmetric(self):
        num_players = 1
        pop_size = 2
        max_version = 100
        training_policy_name = f"policy_{pop_size}"
        population = [f"policy_{i}" for i in range(pop_size)]

        # Check intialization.
        psro_alg = make_psro(pop_size, max_version, symmetric=True)
        psro_alg.configure(*make_workers(num_players))
        self.assertEqual(psro_alg.num_players, num_players)
        self.assertEqual(psro_alg.current_iteration, 0)
        self.assertEqual(psro_alg.pop_sizes[0], pop_size)
        self.assertEqual(psro_alg.pop_sizes[1], pop_size)
        self.assertEqual(psro_alg.population[0], population)
        self.assertEqual(psro_alg.training_policy_names[0], training_policy_name)

        # Before condition is met, make no requests.
        episode_returns = np.array([[0.5, -0.5], [0.9, -0.9]])
        for version in range(max_version + 1):
            for opponent_idx in range(pop_size):
                requests = psro_alg.step(
                    make_ma_sample(
                        policy_names=[training_policy_name, population[opponent_idx]],
                        versions=[version, max_version],
                        episode_returns=episode_returns[opponent_idx],
                    ))
                self.assertIsNone(requests)
            requests = psro_alg.step(
                make_ma_sample(
                    policy_names=[training_policy_name, training_policy_name],
                    versions=[version, version],
                    episode_returns=[0, 0],
                ))
            self.assertIsNone(requests)

        # Condition is met, make requests and start a new PSRO iteration.
        version = max_version + 1
        opponent_idx = np.random.choice(pop_size)
        requests = psro_alg.step(
            make_ma_sample(
                policy_names=[training_policy_name, population[opponent_idx]],
                versions=[version, max_version],
                episode_returns=episode_returns[opponent_idx],
            ))
        self.assertIsNotNone(requests)
        self.assertEqual(psro_alg.current_iteration, 1)
        self.assertEqual(psro_alg.pop_sizes[0], pop_size + 1)
        self.assertEqual(psro_alg.pop_sizes[1], pop_size + 1)
        self.assertEqual(psro_alg.population[0], population + [training_policy_name])
        self.assertAlmostEqual(psro_alg.payoffs[0, pop_size, 0], episode_returns[0, 0])
        self.assertAlmostEqual(psro_alg.payoffs[0, pop_size, 1], episode_returns[1, 0])
        self.assertAlmostEqual(psro_alg.payoffs[1, pop_size, 0], episode_returns[0, 1])
        self.assertAlmostEqual(psro_alg.payoffs[1, pop_size, 1], episode_returns[1, 1])
        self.assertAlmostEqual(psro_alg.payoffs[0, 0, pop_size], episode_returns[0, 1])
        self.assertAlmostEqual(psro_alg.payoffs[0, 1, pop_size], episode_returns[1, 1])
        self.assertAlmostEqual(psro_alg.payoffs[1, 0, pop_size], episode_returns[0, 0])
        self.assertAlmostEqual(psro_alg.payoffs[1, 1, pop_size], episode_returns[1, 0])

    def test_asymmetric(self):
        num_players = 2
        pop_sizes = [2, 2]
        max_versions = [300, 100]
        training_policy_names = [f"player{i}_{n}" for i, n in enumerate(pop_sizes)]
        populations = [[f"player{i}_{n}" for n in range(pop_sizes[i])] for i in range(num_players)]

        # Check intialization.
        psro_alg = make_psro(pop_sizes, max_versions, symmetric=False)
        psro_alg.configure(*make_workers(num_players))
        self.assertEqual(psro_alg.num_players, num_players)
        self.assertEqual(psro_alg.current_iteration, 0)
        self.assertEqual(list(psro_alg.pop_sizes), pop_sizes)
        self.assertEqual(psro_alg.population[0], populations[0])
        self.assertEqual(psro_alg.population[1], populations[1])
        self.assertEqual(psro_alg.training_policy_names[0], training_policy_names[0])
        self.assertEqual(psro_alg.training_policy_names[1], training_policy_names[1])

        # Before all conditions are met, make no requests.
        episode_returns_0 = np.array([[0.5, -0.5], [0.9, -0.9]])
        episode_returns_1 = np.array([[0.3, -0.3], [0.1, -0.1]])
        for version in range(max(max_versions) + 1):
            for opponent_idx in range(pop_sizes[1]):
                requests = psro_alg.step(
                    make_ma_sample(
                        policy_names=[training_policy_names[0], populations[1][opponent_idx]],
                        versions=[version, max_versions[1]],
                        episode_returns=episode_returns_0[opponent_idx],
                    ))
                self.assertIsNone(requests)
            for opponent_idx in range(pop_sizes[0]):
                requests = psro_alg.step(
                    make_ma_sample(policy_names=[populations[0][opponent_idx], training_policy_names[1]],
                                   versions=[max_versions[0], version],
                                   episode_returns=episode_returns_1[opponent_idx]))
                self.assertIsNone(requests)

        # All conditions are met, make requests and start a new PSRO iteration.
        version = max(max_versions) + 1
        opponent_idx = np.random.choice(pop_sizes[1])
        requests = psro_alg.step(
            make_ma_sample(
                policy_names=[training_policy_names[0], populations[1][opponent_idx]],
                versions=[version, max_versions[1]],
                episode_returns=episode_returns_0[opponent_idx],
            ))
        self.assertIsNotNone(requests)
        self.assertEqual(psro_alg.current_iteration, 1)
        self.assertEqual(psro_alg.pop_sizes[0], pop_sizes[0] + 1)
        self.assertEqual(psro_alg.pop_sizes[1], pop_sizes[1] + 1)
        self.assertEqual(psro_alg.population[0], populations[0] + [training_policy_names[0]])
        self.assertEqual(psro_alg.population[1], populations[1] + [training_policy_names[1]])
        self.assertAlmostEqual(psro_alg.payoffs[0, pop_sizes[0], 0], episode_returns_0[0, 0])
        self.assertAlmostEqual(psro_alg.payoffs[0, pop_sizes[0], 1], episode_returns_0[1, 0])
        self.assertAlmostEqual(psro_alg.payoffs[1, pop_sizes[0], 0], episode_returns_0[0, 1])
        self.assertAlmostEqual(psro_alg.payoffs[1, pop_sizes[0], 1], episode_returns_0[1, 1])
        self.assertAlmostEqual(psro_alg.payoffs[0, 0, pop_sizes[1]], episode_returns_1[0, 0])
        self.assertAlmostEqual(psro_alg.payoffs[0, 1, pop_sizes[1]], episode_returns_1[1, 0])
        self.assertAlmostEqual(psro_alg.payoffs[1, 0, pop_sizes[1]], episode_returns_1[0, 1])
        self.assertAlmostEqual(psro_alg.payoffs[1, 1, pop_sizes[1]], episode_returns_1[1, 1])


if __name__ == '__main__':
    unittest.main()
