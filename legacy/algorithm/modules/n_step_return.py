from typing import Optional
import numpy as np
import scipy.signal
import torch

from .utils import scalar_transform, inverse_scalar_transform
import api.trainer
import base.namedarray


@torch.no_grad()
def n_step_return(n: int,
                  reward: torch.FloatTensor,
                  nex_value: torch.FloatTensor,
                  nex_done: torch.FloatTensor,
                  nex_truncated: torch.FloatTensor,
                  gamma: float,
                  high_precision: Optional[bool] = True):
    """Compute the n-step return given reward and the bootstrap Q value.

    Denote the chunk length to compute return as T.

    Args:
        n (int): The "n" in "n"-step return.
        reward (torch.FloatTensor): Reward with shape [n+T-1, B, Nc].
        value (torch.FloatTensor): Bootstrapped Q value with shape [n+T-1, B, Nc].
        nex_done (torch.FloatTensor): Indicator of whether an episode is 
            finished because agents are not alive. Shape [n+T-1, B, 1].
        nex_truncated (torch.FloatTensor): Indicator of whether an episode is 
            finished because of exceeding time limit. Shape [n+T-1, B, 1].
        gamma (float): Discount factor.
        high_precision (bool): Whether to use float64.
    Returns:
        torch.FloatTensor: N-step return of shape [T, B, Nc].
    """
    if high_precision:
        reward, nex_value, nex_done, nex_truncated = map(lambda x: x.to(torch.float64),
                                                         [reward, nex_value, nex_done, nex_truncated])

    T = nex_value.shape[0] - n + 1
    assert T >= 1
    ret = torch.zeros_like(reward[:T])
    discount = torch.ones_like(reward[:T])
    for i in range(n):
        ret += reward[i:i + T] * discount
        # If the next step is truncated, bootstrap value in advance.
        # In following iterations, `discount` will be 0 and truncated returns will not be updated.
        ret += discount * gamma * nex_truncated[i:i + T] * nex_value[i:i + T]
        discount *= gamma * (1 - nex_done[i:i + T]) * (1 - nex_truncated[i:i + T])
    return (ret + discount * nex_value[n - 1:n - 1 + T]).float()


class TrajNstepReturn(api.trainer.TrajPostprocessor):

    def __init__(self,
                 n: int,
                 gamma: float,
                 apply_scalar_transform: bool,
                 scalar_transform_eps: float = 1e-3):
        self.n = n
        self.gamma = gamma
        self.apply_scalar_transform = apply_scalar_transform
        self.scalar_transform_eps = scalar_transform_eps

    def process(self, memory):
        assert memory[-1].done.all() or memory[-1].truncated.all()

        gamma = self.gamma
        n = self.n
        ep_len = len(memory)
        if memory[-1].analyzed_result is None:
            memory[-1].analyzed_result = base.namedarray.array_like(memory[-2].analyzed_result, value=0)

        if memory[-1].done.all():
            bootstrap_value = np.array(
                [x.analyzed_result.target_value for x in memory[n:]] +
                [np.zeros_like(memory[0].analyzed_result.target_value) for _ in range(n - 1)],
                dtype=np.float64)
        else:
            bootstrap_value = np.array(
                [x.analyzed_result.target_value for x in memory[n:]] +
                [memory[-1].analyzed_result.target_value / gamma**(i + 1) for i in range(n - 1)],
                dtype=np.float64)

        reward = np.array([x.reward for x in memory[:-1]], dtype=np.float64)
        shape = reward.shape
        assert reward.shape == bootstrap_value.shape, (n, ep_len, reward.shape, bootstrap_value.shape)
        reward = reward.reshape(ep_len - 1, -1)
        bootstrap_value = bootstrap_value.reshape(ep_len - 1, -1)

        ret = np.zeros_like(reward)
        for i in range(reward.shape[-1]):
            ret[:, i] = scipy.signal.lfilter([gamma**i for i in range(n)], [1],
                                             reward[:, i].flatten()[::-1])[::-1]
            if not self.apply_scalar_transform:
                ret[:, i] += gamma**n * bootstrap_value[:, i].flatten()
            else:
                ret[:, i] += gamma**n * inverse_scalar_transform(
                    torch.from_numpy(bootstrap_value[:, i].flatten()), eps=self.scalar_transform_eps).numpy()

        ret = ret.reshape(*shape)

        for i, m in enumerate(memory[:-1]):
            if not self.apply_scalar_transform:
                m.analyzed_result.ret = ret[i]
            else:
                m.analyzed_result.ret = scalar_transform(torch.from_numpy(ret[i]),
                                                         eps=self.scalar_transform_eps).numpy()
        return memory


api.trainer.register_traj_postprocessor("n-step-return", TrajNstepReturn)