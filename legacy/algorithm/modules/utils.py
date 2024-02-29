from typing import Union, Any, Optional

from torch.distributions import Categorical
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn


@torch.no_grad()
def masked_normalization(
    x,
    mask=None,
    dim=None,
    inplace=False,
    unbiased=False,
    eps=1e-5,
    high_precision=True,
):
    """Normalize x with a mask. Typically used in advantage normalization.

    Args:
        x (torch.Tensor):
            Tensor to be normalized.
        mask (torch.Tensor, optional):
            A mask with the same shape as x. Defaults to None.
        dim (int or tuple of ints, optional):
            Dimensions to be normalized. Defaults to None.
        inplace (bool, optional):
            Whether to perform in-place operation. Defaults to False.
        eps (torch.Tensor, optional):
            Minimal denominator. Defaults to 1e-5.

    Returns:
        torch.Tensor:
            Normalized x, with the same shape as x.
    """
    dtype = torch.float64 if high_precision else torch.float32
    x = x.to(dtype)
    if not inplace:
        x = x.clone()
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        factor = torch.tensor(np.prod([x.shape[d] for d in dim]), dtype=dtype)
    else:
        mask = mask.to(dtype)
        assert len(mask.shape) == len(x.shape), (mask.shape, x.shape, dim)
        for i in range(len(x.shape)):
            if i in dim:
                assert mask.shape[i] == x.shape[i], (mask.shape, x.shape, dim)
            else:
                assert mask.shape[i] == 1, (mask.shape, x.shape, dim)
        x = x * mask
        factor = mask.sum(dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)
    if dist.is_initialized():
        dist.all_reduce(factor, op=dist.ReduceOp.SUM)
        dist.all_reduce(x_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(x_sum_sq, op=dist.ReduceOp.SUM)
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return ((x - mean) / (var.sqrt() + eps)).float()


class RunningMeanStd(nn.Module):

    def __init__(self, input_shape, beta=0.999, epsilon=1e-5, high_precision=True):
        super().__init__()
        self.__beta = beta
        self.__eps = epsilon
        self.__input_shape = input_shape

        self.__dtype = torch.float64 if high_precision else torch.float32

        self.__mean = nn.Parameter(torch.zeros(input_shape, dtype=self.__dtype), requires_grad=False)
        self.__mean_sq = nn.Parameter(torch.zeros(input_shape, dtype=self.__dtype), requires_grad=False)
        self.__debiasing_term = nn.Parameter(torch.zeros(1, dtype=self.__dtype), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.__mean.zero_()
        self.__mean_sq.zero_()
        self.__debiasing_term.zero_()

    def forward(self, *args, **kwargs):
        # we don't implement the forward function because its meaning
        # is somewhat ambiguous
        raise NotImplementedError()

    def __check(self, x, mask):
        assert isinstance(x, torch.Tensor)
        trailing_shape = x.shape[-len(self.__input_shape):]
        assert trailing_shape == self.__input_shape, (
            'Trailing shape of input tensor'
            f'{x.shape} does not equal to configured input shape {self.__input_shape}')
        if mask is not None:
            assert mask.shape == (*x.shape[:-len(self.__input_shape)],
                                  *((1,) * len(self.__input_shape))), (mask.shape, x.shape)

    @torch.no_grad()
    def update(self, x, mask=None):
        self.__check(x, mask)
        x = x.to(self.__dtype)
        if mask is not None:
            mask = mask.to(self.__dtype)
        norm_dims = tuple(range(len(x.shape) - len(self.__input_shape)))
        if mask is None:
            factor = torch.tensor(np.prod(x.shape[:-len(self.__input_shape)])).to(x)
        else:
            x = x * mask
            factor = mask.sum()

        x_sum = x.sum(dim=norm_dims)
        x_sum_sq = x.square().sum(dim=norm_dims)
        if dist.is_initialized():
            dist.all_reduce(factor, op=dist.ReduceOp.SUM)
            dist.all_reduce(x_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(x_sum_sq, op=dist.ReduceOp.SUM)
        batch_mean = x_sum / factor
        batch_sq_mean = x_sum_sq / factor

        self.__mean.data[:] = self.__beta * self.__mean.data[:] + batch_mean * (1.0 - self.__beta)
        self.__mean_sq.data[:] = self.__beta * self.__mean_sq.data[:] + batch_sq_mean * (1.0 - self.__beta)
        self.__debiasing_term.data[:] = self.__beta * self.__debiasing_term.data[:] + 1.0 - self.__beta

    @torch.no_grad()
    def mean_std(self):
        debiased_mean = self.__mean / self.__debiasing_term.clamp(min=self.__eps)
        debiased_mean_sq = self.__mean_sq / self.__debiasing_term.clamp(min=self.__eps)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var.sqrt()

    @torch.no_grad()
    def normalize(self, x):
        self.__check(x, None)
        x = x.to(self.__dtype)
        mean, std = self.mean_std()
        return ((x - mean) / std).clip(-5, 5).float()  # clipping is a trick from hide and seek

    @torch.no_grad()
    def denormalize(self, x):
        self.__check(x, None)
        x = x.to(self.__dtype)
        mean, std = self.mean_std()
        return (x * std + mean).float()


def mlp(sizes, activation=nn.ReLU, layernorm=True):
    # refer to https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py#L15
    layers = []
    for j in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[j], sizes[j + 1]), activation()]
        if layernorm:
            layers += [nn.LayerNorm([sizes[j + 1]])]
    return nn.Sequential(*layers)


def to_chunk(x, num_chunks):
    """Split sample batch into chunks along the time dimension.
    Typically with input shape [T, B, *D].

    Args:
        x (torch.Tensor): The array to be chunked.
        num_chunks (int): Number of chunks. Note as C.

    Returns:
        torch.Tensor: The chunked array.
            Typically with shape [T//C, B*C, *D].
    """
    if x.shape[0] % num_chunks != 0:
        raise IndexError(f"The first dimension(usually the step/time) {x.shape[0]} must be a multiple of "
                         f"num_chunks {num_chunks}. This usually means the sample_steps(config:AgentSpec) "
                         f"is not dividable by chunk_len(config:Policy).")
    return torch.cat(torch.split(x, x.shape[0] // num_chunks, dim=0), dim=1)


def back_to_trajectory(x, num_chunks):
    """Inverse operation of to_chunk.
    Typically with input shape [T//C, B*C, *D].

    Args:
        x (torch.Tensor): The array to be inverted.
        num_chunks (int): Number of chunks. Note as C.

    Returns:
        torch.Tensor: Inverted array.
            Typically with shape [T, B, *D]
    """
    return torch.cat(torch.split(x, x.shape[1] // num_chunks, dim=1), dim=0)


def distribution_back_to_trajctory(distribution: torch.distributions.Distribution, num_chunks):
    if isinstance(distribution, Categorical):
        return Categorical(logits=back_to_trajectory(distribution.logits, num_chunks))
    elif isinstance(distribution, torch.distributions.Normal):
        return torch.distributions.Normal(loc=back_to_trajectory(distribution.loc, num_chunks),
                                          scale=back_to_trajectory(distribution.scale, num_chunks))
    else:
        raise NotImplementedError(f"Don't know how to process {distribution.__class__.__name__}")


def distribution_detach_to_cpu(distribution: torch.distributions.Distribution):
    if isinstance(distribution, Categorical):
        return Categorical(logits=distribution.logits.cpu().detach())
    elif isinstance(distribution, torch.distributions.Normal):
        return torch.distributions.Normal(loc=distribution.loc.cpu().detach(),
                                          scale=distribution.scale.cpu.detach())
    else:
        raise NotImplementedError(f"Don't know how to process {distribution.__class__.__name__}")


def distribution_to(distribution: torch.distributions.Distribution, to_able: Any):
    if isinstance(distribution, Categorical):
        return Categorical(logits=distribution.logits.to(to_able))
    elif isinstance(distribution, torch.distributions.Normal):
        return torch.distributions.Normal(loc=distribution.loc.to(to_able),
                                          scale=distribution.scale.to(to_able))
    else:
        raise NotImplementedError(f"Don't know how to process {distribution.__class__.__name__}")


def get_clip_value_loss_fn(value_loss_fn, value_eps_clip):

    def foo(value, old_value, target_value):
        value_loss_original = value_loss_fn(value, target_value)

        value_clipped = old_value + (value - old_value).clamp(-value_eps_clip, value_eps_clip)
        value_loss_clipped = value_loss_fn(value_clipped, target_value)

        value_loss = torch.max(value_loss_original, value_loss_clipped)
        return value_loss

    return foo


def init_value_loss_fn(value_loss_name,
                       value_loss_config,
                       clip_value: bool = False,
                       value_eps_clip: Optional[float] = 0.2):
    value_loss_collection = ['mse', 'huber', 'smoothl1']
    assert value_loss_name in value_loss_collection, (
        f'Value loss name {value_loss_name}'
        f'does not match any implemented loss functions ({value_loss_collection})')

    if value_loss_name == 'mse':
        value_loss_cls = torch.nn.MSELoss
    elif value_loss_name == 'huber':
        value_loss_cls = torch.nn.HuberLoss
    elif value_loss_name == 'smoothl1':
        value_loss_cls = torch.nn.SmoothL1Loss
    else:
        raise ValueError(f"Unknown loss function {value_loss_name}")

    value_loss_fn = value_loss_cls(reduction='none', **value_loss_config)

    if clip_value:
        value_loss_fn = get_clip_value_loss_fn(value_loss_fn, value_eps_clip)

    return value_loss_fn


def init_optimizer(parameters, optimizer_name, optimizer_config):
    optimizer_collection = ['adam', 'rmsprop', 'sgd', 'adamw']
    assert optimizer_name in optimizer_collection, (
        f'Optimizer name {optimizer_name} '
        f'does not match any implemented optimizers ({optimizer_collection}).')

    if optimizer_name == 'adam':
        optimizer_fn = torch.optim.Adam
    elif optimizer_name == 'rmsprop':
        optimizer_fn = torch.optim.RMSprop
    elif optimizer_name == 'sgd':
        optimizer_fn = torch.optim.SGD
    elif optimizer_name == 'adamw':
        optimizer_fn = torch.optim.AdamW
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

    optim = optimizer_fn(parameters, **optimizer_config)
    return optim


def get_grad_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm()**2
    return torch.sqrt(sum_grad)


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def scalar_transform(x: torch.Tensor, eps=1e-3):
    # TODO: unify scalar transform in muzero implmentation
    x = x.double()
    y = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x
    return y.float()


def inverse_scalar_transform(x: torch.Tensor, eps=1e-3):
    x = x.double()
    a = torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1
    y = torch.sign(x) * (torch.square(a / (2 * eps)) - 1)
    return y.float()
