import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def prepare_observation_lst(observation_lst):
    """Prepare the observations to satisfy the input fomat of torch
    [B, S, W, H, C] -> [B, S x C, W, H]
    batch, stack num, width, height, channel
    """
    # B, S, W, H, C
    observation_lst = np.array(observation_lst, dtype=np.uint8)
    observation_lst = np.moveaxis(observation_lst, -1, 2)

    shape = observation_lst.shape
    observation_lst = observation_lst.reshape((shape[0], -1, shape[-2], shape[-1]))

    return observation_lst


def adjust_lr(lr_schedule, optimizer, step_count):
    # TODO: refactor to use base.timeutil.Scheduler
    total_steps = 0
    lr = None
    for sec in lr_schedule:
        sec = argparse.Namespace(**sec)
        if total_steps + sec.num_steps >= step_count:
            delta_step = step_count - total_steps
        else:
            delta_step = None
        if sec.name == "linear":
            if delta_step is not None:
                lr = delta_step / sec.num_steps * (sec.config["lr_end"] -
                                                   sec.config["lr_init"]) + sec.config["lr_init"]
        elif sec.name == "cosine":
            if delta_step is not None:
                lr = sec.config["lr_init"] * 0.5 * (1. + np.cos(np.pi * delta_step / sec.config["max_steps"]))
        elif sec.name == "decay":
            if delta_step is not None:
                lr = sec.config["lr_init"] * sec.config["lr_decay_rate"]**(delta_step //
                                                                           sec.config["lr_decay_steps"])
        else:
            raise RuntimeError(f"LR schedule named {sec.name} is not supported now.")

        if lr is not None:
            break

        total_steps += sec.num_steps

    if lr is None:
        # training is done
        # TODO(gaojiaxuan): define an error class for training finished
        raise RuntimeError("Training is finished!")
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr


def consist_loss_func(f1, f2):
    """Consistency loss function: similarity loss
    Parameters
    """
    f1 = F.normalize(f1, p=2., dim=-1, eps=1e-5)
    f2 = F.normalize(f2, p=2., dim=-1, eps=1e-5)
    return -(f1 * f2).sum(dim=-1)


def get_grad_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm()**2
    return math.sqrt(sum_grad)


def _n2t(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def _t2n(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x


def pad_sampled_actions(act_dim, actions_pool, policy_logits_pool):
    """pad zero actions, policy logits and use available mask
    Called when using sampled actions to do mcts
    """

    max_num = max([len(actions) for actions in actions_pool])
    if act_dim == -1:
        act_dim = max_num
    assert max_num <= act_dim

    available_action = np.array([
        np.concatenate([np.ones(
            (len(actions),)), np.zeros((act_dim - len(actions),))], axis=0) for actions in actions_pool
    ])
    policy_logits_pool = np.array([
        np.concatenate([policy_logits, np.zeros((act_dim - len(actions),))], axis=0)
        for policy_logits, actions in zip(policy_logits_pool, actions_pool)
    ])

    empty_action = np.zeros_like(actions_pool[0][0])
    empty_action = empty_action.reshape(1, *empty_action.shape)
    actions_pool = np.array([
        np.concatenate([actions, empty_action.repeat(act_dim - len(actions), axis=0)], axis=0)
        for actions in actions_pool
    ])

    return available_action, policy_logits_pool, actions_pool


class LinearSchedule:

    def __init__(self, v_init=0., v_end=1., t_end=0):
        self.v_init = v_init
        self.v_end = v_end
        self.t_end = t_end

    def eval(self, t):
        if t >= self.t_end:
            return self.v_end
        if t <= 0:
            return self.v_init
        v = self.v_init + (self.v_end - self.v_init) / (self.t_end + 1) * t
        return v
