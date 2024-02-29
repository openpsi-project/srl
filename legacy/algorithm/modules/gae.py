from typing import Union, Optional
import numpy as np
import torch

import api.trainer


@torch.no_grad()
def gae_trace(
    reward: torch.FloatTensor,
    value: torch.FloatTensor,
    truncated: torch.FloatTensor,
    done: torch.FloatTensor,
    on_reset: torch.FloatTensor,
    gamma: Union[float, torch.FloatTensor],
    lmbda: Union[float, torch.FloatTensor],
    vtrace: Optional[bool] = False,
    imp_ratio: Optional[torch.FloatTensor] = None,
    rho: Optional[float] = 1.0,
    c: Optional[float] = 1.0,
    high_precision: Optional[bool] = True,
) -> torch.FloatTensor:
    """Compute the Generalized Advantage Estimation.

    Args:
        reward (torch.FloatTensor): rewards of shape [T, bs, Nc]
        value (torch.FloatTensor): values of shape [T+1, bs, Nc]
        truncated (torch.FloatTensor): truncated indicator of shape [T+1, bs, 1]
        done (torch.FloatTensor): done (aka terminated) indicator of shape [T+1, bs, 1]
        on_reset (torch.FloatTensor): whether on the reset step, shape [T+1, bs, 1]
        gamma (Union[float, torch.FloatTensor]): discount factor.
            If input is a tensor, it should have shape [T, bs, 1]
        lmbda (Union[float, torch.FloatTensor]): GAE lambda.
            If input is a tensor, it should have shape [T, bs, 1]
        vtrace (Optional[bool]): whether to use V-trace correction. Defaults to False.
        imp_ratio (Optional[torch.FloatTensor]): importance sampling ratio of shape [T, bs, 1]
        rho (Optional[float]):
            Clipping hyperparameter rho as described in the paper. Defaults to 1.0.
        c (Optional[float]):
            Clipping hyperparameter c as described in the paper. Defaults to 1.0.
        high_precision (Optional[bool]): whether to use float64. Defaults to True.
    Returns:
        torch.FloatTensor: GAE of shape [T, bs, Nc]
    """

    if high_precision:
        reward, value, truncated, done, on_reset = map(lambda x: x.to(torch.float64),
                                                       [reward, value, truncated, done, on_reset])
        if vtrace:
            imp_ratio = imp_ratio.to(torch.float64)
    if not isinstance(gamma, float):
        assert isinstance(gamma, torch.FloatTensor), type(gamma)
        assert gamma.shape == on_reset[:-1].shape, gamma.shape
        if high_precision:
            gamma = gamma.to(torch.float64)
    if not isinstance(lmbda, float):
        assert isinstance(lmbda, torch.FloatTensor), type(lmbda)
        assert lmbda.shape == on_reset[:-1].shape, lmbda.shape
        if high_precision:
            lmbda = lmbda.to(torch.float64)

    episode_length = int(reward.shape[0])
    delta = reward + gamma * value[1:] * (1 - on_reset[1:]) - value[:-1]
    if vtrace:
        delta *= imp_ratio.clip(max=rho)

    ################## ASSERTIONS START ##################
    ###### disable assertions with `python -O xxx` #######
    assert (truncated * done == 0).all()
    assert ((truncated + done)[:-1] == on_reset[1:]).all()
    # the reward should not be amended at the final step
    assert (reward * on_reset[1:] == 0).all()
    # when an episode is done (not truncated), reward, value, and bootstrapped value should be zero
    # hence delta is also zero
    assert (delta * on_reset[1:] * (1 - truncated[:-1]) == 0).all()
    # when an episode is truncated, reward and bootstrapped value are zero
    assert (delta * truncated[:-1] == -value[:-1] * truncated[:-1]).all()
    ################### ASSERTIONS END ###################

    gae = torch.zeros_like(reward[0])
    adv = torch.zeros_like(reward)

    # 1. If the next step is a new episode, GAE doesn't propagate back
    # 2. If the next step is a truncated final step, the backpropagated GAE is -V(t),
    #    which is not correct. We ignore it such that the current GAE is r(t-1)+ɣV(t)-V(t-1)
    # 3. If the next step is a done final step, the backpropagated GAE is zero.
    m = gamma * lmbda * (1 - on_reset[1:]) * (1 - truncated[1:])
    if vtrace:
        m *= imp_ratio.clip(max=c)

    step = episode_length - 1
    while step >= 0:
        gae = delta[step] + m[step] * gae
        adv[step] = gae
        step -= 1

    return adv.float()


class TrajGAE(api.trainer.TrajPostprocessor):
    """Compute GAE along a trajectory.
    
    We don't care about cross-episode data, so the code is much simplified.
    """

    def __init__(self, gamma, lmbda):
        self.gamma = gamma
        self.lmbda = lmbda

    def process(self, memory):
        assert np.logical_or(memory[-1].done, memory[-1].truncated).all()
        gamma = self.gamma
        lmbda = self.lmbda

        ep_len = len(memory)
        gae = np.zeros_like(memory[0].reward)
        step = ep_len - 2  # computing GAE except for the last step
        while step >= 0:
            reward = memory[step].reward
            value = memory[step].analyzed_result.value
            if step == ep_len - 2:
                if memory[step + 1].analyzed_result is None:
                    # the episode is done, no final value
                    bootstrap_value = 0
                else:
                    # at the final step, we remain bootstrap value only when the episode is truncated
                    bootstrap_value = memory[step + 1].analyzed_result.value * memory[step + 1].truncated
            else:
                bootstrap_value = memory[step + 1].analyzed_result.value

            delta = reward + gamma * bootstrap_value - value
            gae = gamma * lmbda * gae + delta

            memory[step].analyzed_result.adv = gae
            memory[step].analyzed_result.ret = gae + value

            step -= 1

        return memory


api.trainer.register_traj_postprocessor('gae', TrajGAE)