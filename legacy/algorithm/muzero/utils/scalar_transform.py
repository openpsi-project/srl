import numpy as np
import torch


class DiscreteSupport(object):

    def __init__(self, min: float, max: float, delta=1.):
        assert min < max
        self.min = min
        self.max = max
        self.range = np.arange(min, max + delta, delta)
        self.size = len(self.range)
        self.delta = delta


def inverse_scalar_transform(logits, scalar_support, rescale=False):
    """ Reference from MuZero: Appendix F => Network Architecture
    & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
    """
    delta = scalar_support.delta
    value_probs = torch.softmax(logits, dim=-1)
    value_support = torch.from_numpy(np.array([x for x in scalar_support.range
                                               ])).repeat(*value_probs.shape[:-1],
                                                          1).to(device=value_probs.device)
    assert value_support.shape == value_probs.shape
    value = (value_support * value_probs).sum(-1, keepdim=True)

    if rescale:
        epsilon = 0.001
        sign = torch.ones(value.shape).float().to(value.device)
        sign[value < 0] = -1.0
        output = (((torch.sqrt(1 + 4 * epsilon * (torch.abs(value) + 1 + epsilon)) - 1) / (2 * epsilon))**2 -
                  1)
        output = sign * output

        nan_part = torch.isnan(output)
        output[nan_part] = 0.
        output[torch.abs(output) < epsilon] = 0.
    else:
        output = value
    return output


def scalar_transform(x, scalar_support):
    """ Reference from MuZero: Appendix F => Network Architecture
    & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
    """
    epsilon = 0.001
    sign = torch.ones(x.shape).float().to(x.device)
    sign[x < 0] = -1.0
    output = sign * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x
    return output


def phi(x, scalar_support):
    min = scalar_support.min
    max = scalar_support.max
    set_size = scalar_support.size
    delta = scalar_support.delta

    x.clamp_(min, max)
    x = (x - min) / delta

    x_low = x.floor()
    x_high = x.ceil()
    p_high = x - x_low
    p_low = 1 - p_high

    target = torch.zeros(*x.shape, set_size).to(x.device)
    x_high_idx, x_low_idx = x_high.long(), x_low.long()
    target.scatter_(-1, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
    target.scatter_(-1, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
    return target
