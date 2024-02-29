import numpy as np
import unittest

from legacy.population_based_training.meta_solver import make_solver
import api.config


def make_test_solver(type_="uniform"):
    if type_ == "uniform":
        return make_solver(api.config.MetaSolver(type_=api.config.MetaSolver.Type.UNIFORM))
    elif type_ == "nash":
        return make_solver(api.config.MetaSolver(type_=api.config.MetaSolver.Type.NASH))
    else:
        raise NotImplementedError


class MetaSolverTest(unittest.TestCase):

    def test_uniform_solver(self):
        solver = make_test_solver("uniform")

        # Two-player game.
        payoffs = np.ones((2, 3, 4))
        meta_strategies = solver.solve(payoffs)
        self.assertEqual(len(meta_strategies), 2)
        for p in meta_strategies[0]:
            self.assertAlmostEqual(p, 1 / 3)
        for p in meta_strategies[1]:
            self.assertAlmostEqual(p, 1 / 4)

        # Multi-player game.
        num_player = 4
        pop_sizes = [5, 6, 7, 8]
        payoffs = np.ones((num_player, *pop_sizes))
        meta_strategies = solver.solve(payoffs)
        self.assertEqual(len(meta_strategies), num_player)
        for i, n in enumerate(pop_sizes):
            for p in meta_strategies[i]:
                self.assertAlmostEqual(p, 1 / n)

    def test_nash_solver(self):
        solver = make_test_solver("nash")

        # Rock-Paper-Scissors (RPS).
        payoffs = np.array([[[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]],
                            [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]]])
        meta_strategies = solver.solve(payoffs)
        self.assertEqual(len(meta_strategies), 2)
        self.assertAlmostEqual(meta_strategies[0][0], 1 / 3)
        self.assertAlmostEqual(meta_strategies[0][1], 1 / 3)
        self.assertAlmostEqual(meta_strategies[0][2], 1 / 3)
        self.assertAlmostEqual(meta_strategies[1][0], 1 / 3)
        self.assertAlmostEqual(meta_strategies[1][1], 1 / 3)
        self.assertAlmostEqual(meta_strategies[1][2], 1 / 3)

        # Biased RPS.
        payoffs = np.array([[[0.0, -0.25, 0.5], [0.25, 0.0, -0.05], [-0.5, 0.05, 0.0]],
                            [[0.0, 0.25, -0.5], [-0.25, 0.0, 0.05], [0.5, -0.05, 0.0]]])
        meta_strategies = solver.solve(payoffs)
        self.assertEqual(len(meta_strategies), 2)
        self.assertAlmostEqual(meta_strategies[0][0], 1 / 16)
        self.assertAlmostEqual(meta_strategies[0][1], 10 / 16)
        self.assertAlmostEqual(meta_strategies[0][2], 5 / 16)
        self.assertAlmostEqual(meta_strategies[1][0], 1 / 16)
        self.assertAlmostEqual(meta_strategies[1][1], 10 / 16)
        self.assertAlmostEqual(meta_strategies[1][2], 5 / 16)

        # Asymmetric game.
        payoffs = np.array([[[2.0, 1.0, 5.0], [-3.0, -4.0, -2.0]], [[-2.0, -1.0, -5.0], [3.0, 4.0, 2.0]]])
        meta_strategies = solver.solve(payoffs)
        self.assertEqual(len(meta_strategies), 2)
        self.assertAlmostEqual(meta_strategies[0][0], 1)
        self.assertAlmostEqual(meta_strategies[0][1], 0)
        self.assertAlmostEqual(meta_strategies[1][0], 0)
        self.assertAlmostEqual(meta_strategies[1][1], 1)
        self.assertAlmostEqual(meta_strategies[1][2], 0)


if __name__ == '__main__':
    unittest.main()
