from legacy.algorithm.modules.utils import RunningMeanStd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PopArtValueHead(nn.Module):

    def __init__(
        self,
        input_dim,
        critic_dim,
        beta=0.99999,
        epsilon=1e-5,
        burn_in_updates=torch.inf,
        high_precision=True,
    ):
        super().__init__()
        self.__rms = RunningMeanStd((critic_dim,), beta=beta, epsilon=epsilon, high_precision=high_precision)

        self.__weight = nn.Parameter(torch.zeros(critic_dim, input_dim))
        self.__bias = nn.Parameter(torch.zeros(critic_dim))
        # The same initialization as `nn.Linear`.
        torch.nn.init.kaiming_uniform_(self.__weight, a=math.sqrt(5))
        torch.nn.init.uniform_(self.__bias, -1 / math.sqrt(input_dim), 1 / math.sqrt(input_dim))

        self.__burn_in_updates = burn_in_updates
        self.__update_cnt = 0

    @property
    def weight(self):
        return self.__weight

    @property
    def bias(self):
        return self.__bias

    def forward(self, feature):
        return F.linear(feature, self.__weight, self.__bias)

    @torch.no_grad()
    def update(self, x, mask):
        old_mean, old_std = self.__rms.mean_std()
        self.__rms.update(x, mask)
        new_mean, new_std = self.__rms.mean_std()
        self.__update_cnt += 1

        if self.__update_cnt > self.__burn_in_updates:
            self.__weight.data[:] = self.__weight * (old_std / new_std).unsqueeze(-1)
            self.__bias.data[:] = (old_std * self.__bias + old_mean - new_mean) / new_std

    @torch.no_grad()
    def normalize(self, x):
        return self.__rms.normalize(x)

    @torch.no_grad()
    def denormalize(self, x):
        return self.__rms.denormalize(x)
