from scipy.optimize import linprog
import numpy as np

import api.config


class MetaSolver:

    def solve(self, payoffs):
        """Given a payoff tensor, calculate the meta-strategy for each player.
        Args:
            payoffs (np.ndarray): payoffs[i, j, k], i is the index of player, j is the index of player 0's 
              policy, k is the index of player 1's policy.
        Returns:
            List[np.ndarray]: list of meta-strategies, i.e. policy sample probabilities.
        """
        raise NotImplementedError()


class UniformSolver(MetaSolver):
    """Use uniform sample probability as the meta-strategy for each player.
    """

    def solve(self, payoffs):
        return [np.ones(pop_size) / pop_size for pop_size in payoffs.shape[1:]]


class NashSolver(MetaSolver):
    """Use Nash equilibrium as the meta-strategy for each player. A meta-game is built using the empirical
       payoff tensor, and each policy in the population is a strategy for the player. Then Nash equilibrium
       gives the mixed strategy that no player can benefit from changing his meta-strategy unilaterally.

       NOTE: Currently Nash solver only supports 2-player zero-sum game. For other settings like multi-player
       or general-sum game, the complexity to compute Nash equilibrium is too high, use approximate solvers
       instead.
    """

    def solve(self, payoffs):
        assert payoffs.shape[0] == 2 and (np.sum(payoffs, axis=0) == 0).all(), (
            f"Currently Nash solver only supports 2-player zero-sum game, but got {payoffs.shape[0]} players "
            f"or not zero-sum payoffs: {np.sum(payoffs, axis=0)}")
        num_player, num_row, num_col = payoffs.shape
        # Add a constant to row player's payoff matrix to make all elements positive, this will not affect
        # the Nash equilibium. Same thing is done to the column player's payoff matrix
        payoffs_row = payoffs[0] - np.min(payoffs[0])
        result_row = linprog(c=np.ones(num_row),
                             A_ub=-payoffs_row.T,
                             b_ub=-np.ones(num_col),
                             bounds=[(0, None)] * num_row)
        meta_strategy_row = result_row.x / np.sum(result_row.x)

        payoffs_col = payoffs[1] - np.min(payoffs[1])
        result_col = linprog(c=np.ones(num_col),
                             A_ub=-payoffs_col,
                             b_ub=-np.ones(num_row),
                             bounds=[(0, None)] * num_col)
        meta_strategy_col = result_col.x / np.sum(result_col.x)

        return [meta_strategy_row, meta_strategy_col]


def make_solver(cfg: api.config.MetaSolver):
    if cfg.type_ == api.config.MetaSolver.Type.UNIFORM:
        return UniformSolver()
    elif cfg.type_ == api.config.MetaSolver.Type.NASH:
        return NashSolver()
    else:
        raise NotImplementedError()
