import numpy as np
import torch.nn as nn
import torch

from api.trainer import register, TrainerStepResult, PytorchTrainer
from api.policy import Policy
from base.namedarray import recursive_apply, recursive_aggregate


class OnPlateau:
    """A checker indicating if the loss is on plateau.
    """

    def __init__(self, mode='min', patience=10, threshold=1e-4, threshold_mode='rel'):

        self.patience = patience
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.num_bad_epochs = 0

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self.num_bad_epochs = 0
            return True

        return False

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = torch.inf
        else:  # mode == 'max':
            self.mode_worse = -torch.inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode


class DAgger(PytorchTrainer):
    """Dataset Aggreagation, which is an imitation leanring algorithm.

    Ref:
        A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning
        Available at (https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf)
    """

    def get_checkpoint(self):
        checkpoint = self.policy.get_checkpoint()
        checkpoint.update({"optimizer_state_dict": self.optimizer.state_dict()})
        return checkpoint

    def load_checkpoint(self, checkpoint, **kwargs):
        if "optimizer_state_dict" in checkpoint.keys():
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.policy.load_checkpoint(checkpoint)

    def __init__(self, policy: Policy, **kwargs):
        self.max_grad_norm = kwargs.get('max_grad_norm')
        optimizer_name = kwargs.get('optimizer', 'adam')
        self.optimizer = self._init_optimizer(optimizer_name, kwargs.get('optimizer_config', {}))

        self.cur_buffer = None
        self.history_buffer = None
        self.buffer_size_per_training_iter = kwargs.get('buffer_size_per_training_iter', 1000)

        self.cur_train_iter = 1
        self.plateau_cheker = OnPlateau()

        self.frames = 0
        super(DAgger, self).__init__(policy)

    def _init_optimizer(self, optimizer_name, optimizer_config):
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

        optim = optimizer_fn(self.policy.parameters(), **optimizer_config)
        return optim

    def step(self, sample):
        bs = sample.reward.shape[1]

        if self.cur_train_iter > 1:
            # randomly select fresh samples
            indices = np.random.choice(bs, bs // self.cur_train_iter, replace=False)
            fresh_sample = sample[:, indices]

            # randomly select old samples from buffer
            buffer_size = self.history_buffer.reward.shape[1]
            indices = np.random.choice(buffer_size, bs - bs // self.cur_train_iter, replace=False)
            old_sample = self.history_buffer[:, indices]

            log_probs = self.policy.analyze(
                recursive_aggregate([fresh_sample, old_sample], lambda x: np.concatenate(x, 1)))
        else:
            log_probs = self.policy.analyze(sample)

        # maximize log likelihood
        loss = -log_probs.mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.policy.inc_version()
        self.frames += len(sample)

        self.cur_buffer = sample if self.cur_buffer is None else recursive_aggregate(
            [self.cur_buffer, sample], lambda x: np.concatenate(x, 1))
        if self.cur_buffer.reward.shape[1] > self.buffer_size_per_training_iter * 5:
            self.cur_buffer = self.cur_buffer[:, :self.buffer_size_per_training_iter * 2]

        advance_iter = self.plateau_cheker.step(loss.item())
        if advance_iter:
            self.cur_train_iter += 1
            # loss on plateau, stepping into the next training iteration
            cur_buffer_size = self.cur_buffer.reward.shape[1]
            if cur_buffer_size <= self.buffer_size_per_training_iter:
                new_buffer = self.cur_buffer
            else:
                indices = np.random.choice(cur_buffer_size, self.buffer_size_per_training_iter, replace=False)
                new_buffer = self.cur_buffer[:, indices]
            if self.history_buffer is None:
                self.history_buffer = new_buffer
            else:
                self.history_buffer = recursive_aggregate([self.history_buffer, new_buffer],
                                                          lambda x: np.concatenate(x, 1))
            self.cur_buffer = None

        # Logging
        elapsed_episodes = sample.info_mask.sum()
        if elapsed_episodes == 0:
            info = {}
        else:
            info = recursive_apply(sample.info * sample.info_mask, lambda x: x.sum()) / elapsed_episodes
        # TODO:  len(sample) equals sample_steps + bootstrap_steps, which is incorrect for frames logging
        stats = dict(loss=loss.item(), frames=self.frames, cur_train_iter=self.cur_train_iter, **info)
        return TrainerStepResult(stats=stats, step=self.policy.version, agree_pushing=advance_iter)


register('dagger', DAgger)
