import dataclasses
import numpy as np
import torch.nn as nn
import torch
import logging

from api.trainer import PytorchTrainer, register, TrainerStepResult, PyTorchGPUPrefetcher
from api.policy import Policy
from base.namedarray import recursive_apply
import legacy.algorithm.modules as modules

logger = logging.getLogger("qlearning")


@dataclasses.dataclass
class SampleAnalyzedResult:
    q_tot: torch.Tensor  # (T, B, 1) q_tot value of training policy
    target_q_tot: torch.Tensor  # (T, B, 1) q_tot value of target policy


class DeepQLearning(PytorchTrainer):
    """A general deep Q-learning trainer.
    
    It can be used for both single-agent and multi-agent environments.
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
        super(DeepQLearning, self).__init__(policy)
        # discount
        self.gamma = kwargs.get("gamma", 0.99)
        self.bootstrap_steps = kwargs.get("bootstrap_steps", 1)
        if self.bootstrap_steps <= 0:
            raise ValueError("bootstrap_steps of Q-Learning must be a positive integer!")
        self.burn_in_steps = kwargs.get("burn_in_steps", 0)
        # soft update
        self.use_soft_update = kwargs.get("use_soft_update", False)
        self.tau = kwargs.get("tau", 0.005)
        # hard update
        self.hard_update_interval = kwargs.get("hard_update_interval", 200)
        # grad norm
        self.max_grad_norm = kwargs.get("max_grad_norm", None)
        # value normalization
        self.use_popart = kwargs.get('use_popart', False)
        self.apply_scalar_transform = kwargs.get("apply_scalar_transform", True)
        if self.use_popart and self.apply_scalar_transform:
            raise ValueError("use_popart and apply_scalar_transform cannot be True simultaneously!")
        self.scalar_transform_eps = kwargs.get("scalar_transform_eps", 1e-3)
        # prioritized experience replay
        self.use_priority_weight = kwargs.get('use_priority_weight', False)
        self.priority_interpolation_eta = kwargs.get("priority_interpolation_eta", 0.9)
        # gpu prefetching
        self.gpu_prefetching = kwargs.get("gpu_prefetching", True)
        if self.gpu_prefetching and str(self.policy.device) != 'cpu':
            self.prefetcher = PyTorchGPUPrefetcher()
        # mixed precision training
        # FIXME: mixed precision will cause float point error in scalar transform
        self.mixed_precision = kwargs.get("mixed_precision", False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        optimizer_name = kwargs.get('optimizer', 'adam')
        self.optimizer = modules.init_optimizer(
            self.policy.parameters(),
            optimizer_name,
            kwargs.get('optimizer_config', dict(lr=5e-4, eps=1e-5, weight_decay=0.)),
        )

        value_loss_name = kwargs.get('value_loss', 'mse')
        self.value_loss_fn = modules.init_value_loss_fn(
            value_loss_name,
            kwargs.get('value_loss_config', {}),
        )

        self.last_hard_update_step = 0
        self.hard_update_count = 0
        self.steps = 0

        self.frames = 0
        self.total_timesteps = 0

    def step(self, sample):
        if self.gpu_prefetching and str(self.policy.device) != 'cpu':
            fetched_data = self.prefetcher.push(sample)
            if fetched_data is None:
                return TrainerStepResult({}, 0)
            else:
                sample, tensor_sample = fetched_data
        else:
            tensor_sample = recursive_apply(
                sample, lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=self.policy.device))

        sample_total_length = sample.on_reset.shape[0]
        sample_steps = sample_total_length - self.burn_in_steps - self.bootstrap_steps
        if sample_steps <= 0:
            raise RuntimeError("Sample length should be larger than bootstrap_steps for Q-learning!")

        stats = {}

        self.optimizer.zero_grad()

        # with torch.autocast(device_type=str(self.policy.device).split(":")[0],
        #                     dtype=torch.float16,
        #                     enabled=self.mixed_precision):
        # The length of analyzed_result is sample_total_length - burn_in_steps. Burn-in steps are removed.
        analyzed_result: SampleAnalyzedResult = self.policy.analyze(tensor_sample,
                                                                    burn_in_steps=self.burn_in_steps)

        valid_slice = slice(self.burn_in_steps, sample_total_length - self.bootstrap_steps)

        done = tensor_sample.done[self.burn_in_steps:sample_total_length - self.bootstrap_steps]

        # The length of q_tot is sample_steps.
        train_q_tot = analyzed_result.q_tot[:sample_steps]

        # The length of target_q_tot is sample_steps + bootstrap_steps.
        target_q_tot = analyzed_result.target_q_tot
        if self.use_popart:
            target_q_tot = self.policy.denormalize_target_value(target_q_tot)
        elif self.apply_scalar_transform:
            target_q_tot = modules.inverse_scalar_transform(target_q_tot, eps=self.scalar_transform_eps)

        # The length of target_q_tot becomes sample_steps.
        target_q_tot = modules.n_step_return(
            n=self.bootstrap_steps,
            reward=tensor_sample.reward[self.burn_in_steps:-1],  # length sample_steps+bootstrap_steps-1
            nex_value=target_q_tot[
                1:],  # target_q_tot is aligned with q_tot, so we need to move 1 step forward
            nex_done=tensor_sample.done[1 + self.burn_in_steps:sample_total_length],
            nex_truncated=tensor_sample.truncated[1 + self.burn_in_steps:sample_total_length],
            gamma=self.gamma,
        )

        if self.use_popart:
            target_q_tot = self.policy.normalize_value(target_q_tot)
        elif self.apply_scalar_transform:
            target_q_tot = modules.scalar_transform(target_q_tot, eps=self.scalar_transform_eps)

        loss_mask = 1. - done

        # with torch.autocast(device_type=str(self.policy.device).split(":")[0],
        #                     dtype=torch.float16,
        #                     enabled=self.mixed_precision):
        train_q_tot = train_q_tot.broadcast_to(target_q_tot.shape)
        value_loss = self.value_loss_fn(train_q_tot, target_q_tot)
        if self.use_priority_weight and "sampling_weight" in sample.metadata:
            loss_weight = torch.from_numpy(sample.metadata['sampling_weight']).to(dtype=torch.float32,
                                                                                  device=self.policy.device)
            value_loss *= loss_weight.view(loss_weight.shape[0], *((1,) * (len(value_loss.shape) - 2)))
        value_loss = value_loss * loss_mask
        value_loss = value_loss.sum() / loss_mask.sum()

        self.scaler.scale(value_loss).backward()
        self.scaler.unscale_(self.optimizer)

        if self.max_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        else:
            grad_norm = modules.get_grad_norm(self.policy.parameters())

        self.scaler.step(self.optimizer)
        self.scaler.update()

        train_q_tot = train_q_tot.detach()

        # Compute new priorities of (transformed) Q values.
        td_err = (train_q_tot - target_q_tot).abs()
        mean_td_err = (td_err * loss_mask).sum(dim=0) / loss_mask.sum(dim=0)
        max_td_err = (td_err * loss_mask).max(dim=0).values
        new_priorities = self.priority_interpolation_eta * max_td_err + (
            1 - self.priority_interpolation_eta) * mean_td_err
        # Average over dimensions except for time and batch.
        new_priorities = new_priorities.mean(dim=tuple(range(1, len(new_priorities.shape)))).cpu().numpy()

        # Denormalize just for logging.
        normalized_train_q_tot = train_q_tot
        normalized_target_q_tot = target_q_tot
        if self.use_popart:
            train_q_tot = self.policy.denormalize_value(train_q_tot)
            target_q_tot = self.policy.denormalize_value(target_q_tot)
            self.policy.update_popart(target_q_tot, mask=loss_mask)
        elif self.apply_scalar_transform:
            train_q_tot = modules.inverse_scalar_transform(train_q_tot)
            target_q_tot = modules.inverse_scalar_transform(target_q_tot)

        self.total_timesteps += loss_mask.sum().item()

        # update target network
        if self.use_soft_update:
            self.policy.soft_target_update(self.tau)
        else:
            if (self.steps - self.last_hard_update_step) / self.hard_update_interval >= 1.:
                self.policy.hard_target_update()
                self.last_hard_update_step = self.steps
                self.hard_update_count += 1
        self.steps += 1

        loss_mask = loss_mask.to(torch.bool)
        stats.update(
            dict(value_loss=value_loss.item(),
                 grad_norm=grad_norm.item(),
                 train_q_tot=torch.masked_select(train_q_tot, loss_mask).mean().item(),
                 target_q_tot=torch.masked_select(target_q_tot, loss_mask).mean().item(),
                 normalized_train_q_tot=torch.masked_select(normalized_train_q_tot, loss_mask).mean().item(),
                 normalized_target_q_tot=torch.masked_select(normalized_target_q_tot,
                                                             loss_mask).mean().item(),
                 mean_td_err=torch.masked_select(td_err, loss_mask).mean().item(),
                 max_td_err=torch.masked_select(td_err, loss_mask).max().item(),
                 new_priorities=new_priorities.mean().item(),
                 priority_weight=sample.metadata['sampling_weight'].mean().item()
                 if "sampling_weight" in sample.metadata else 0.0,
                 epsilon=sample.policy_state.epsilon.mean(),
                 frames=self.frames,
                 hard_update_count=self.hard_update_count,
                 total_timesteps=self.total_timesteps))

        valid_sample = sample[valid_slice]

        self.frames += np.prod(valid_sample.on_reset.shape)
        self.policy.inc_version()

        # Logging
        elapsed_episodes = valid_sample.info_mask.sum()
        if elapsed_episodes == 0:
            info = {}
        else:
            info = recursive_apply(valid_sample.info * valid_sample.info_mask,
                                   lambda x: x.sum()) / elapsed_episodes
        stats.update(info)
        return TrainerStepResult(stats=stats, step=self.steps, priorities=new_priorities)


register('q-learning', DeepQLearning)
