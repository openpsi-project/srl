from collections import defaultdict
from typing import Optional
from torch.nn import functional as F
import dataclasses
import logging
import numpy as np
import torch.nn as nn
import torch

from legacy.algorithm import modules
from legacy.algorithm.modules import init_optimizer
from api.trainer import PytorchTrainer, register, TrainerStepResult, PyTorchGPUPrefetcher
from api.policy import Policy
from base.namedarray import recursive_apply

import time

logger = logging.getLogger("MAPPO")


@dataclasses.dataclass
class SampleAnalyzedResult:
    """PPO loss computation requires:
    REMEMBER that you drop the last action.
    1. Old / New action probabilities
    2. State Values
    3. Rewards
    4. Entropy
    """
    old_action_log_probs: torch.Tensor  # For chosen actions only, flattened
    new_action_log_probs: torch.Tensor  # for chosen actions only, flattened
    state_values: torch.Tensor  # len(state_values) = len(new_action_probs) + 1
    entropy: Optional[torch.Tensor] = None


@dataclasses.dataclass
class PPOStepResult:
    advantage: torch.Tensor
    entropy: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    done: torch.Tensor
    truncated: torch.Tensor
    clip_ratio: torch.Tensor
    importance_weight: torch.Tensor
    value_targets: torch.Tensor
    denorm_value: Optional[torch.Tensor]


class MultiAgentPPO(PytorchTrainer):
    """Multi Agent Proximal Policy Optimization

    Ref:
        The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games
        Available at (https://arxiv.org/pdf/2103.01955.pdf)
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
        super().__init__(policy)
        # discount & clip
        self.discount_rate = kwargs.get("discount_rate", 0.99)
        self.gae_lambda = kwargs.get("gae_lambda", 0.97)
        self.eps_clip = kwargs.get("eps_clip", 0.2)
        self.clip_value = kwargs.get("clip_value", False)
        self.dual_clip = kwargs.get("dual_clip", True)
        self.c_clip = kwargs.get("c_clip", 3)
        # burn-in: policy states are re-computed upon data reuse.
        self.burn_in_steps = kwargs.get("burn_in_steps", 0)
        # value tracer
        self.vtrace = kwargs.get("vtrace", False)
        self.recompute_adv_on_reuse = kwargs.get("recompute_adv_on_reuse", True)
        self.recompute_adv_among_epochs = kwargs.get("recompute_adv_among_epochs", False)
        # value loss
        self.normalize_old_value = kwargs.get("normalize_old_value", False)
        if self.clip_value and self.normalize_old_value != getattr(policy, "denormalize_value_during_rollout",
                                                                   False):
            raise ValueError(
                "Trainer `normalize_old_value` and policy `denormalize_value_during_rollout` should be consistent!"
            )
        self.value_eps_clip = kwargs.get("value_eps_clip", self.eps_clip)
        self.value_loss_weight = kwargs.get('value_loss_weight', 0.5)
        # entropy
        self.entropy_bonus_weight = kwargs.get('entropy_bonus_weight', 0.01)
        self.entropy_decay_per_steps = kwargs.get("entropy_decay_per_steps", None)
        self.entropy_bonus_decay = kwargs.get("entropy_bonus_decay", 0.99)
        # gradient norm
        self.max_grad_norm = kwargs.get('max_grad_norm')
        # value normalization
        self.popart = kwargs.get('popart', False)
        self.bootstrap_steps = kwargs.get("bootstrap_steps", 1)
        # ppo_epochs: How many updates per sample.
        self.ppo_epochs = kwargs.get("ppo_epochs", 1)

        optimizer_name = kwargs.get('optimizer', 'adam')
        self.optimizer = init_optimizer(self.policy.parameters(), optimizer_name,
                                        kwargs.get('optimizer_config', {}))

        value_loss_name = kwargs.get('value_loss', 'mse')
        self.value_loss_fn = modules.init_value_loss_fn(value_loss_name,
                                                        kwargs.get('value_loss_config', {}),
                                                        clip_value=self.clip_value,
                                                        value_eps_clip=self.value_eps_clip)

        self.frames = 0

        self.prefetcher = PyTorchGPUPrefetcher()

    @torch.no_grad()
    def _compute_adv_and_value_target(self, sample, analyzed_result: SampleAnalyzedResult):
        if self.popart:
            trace_target_value = self.policy.denormalize_value(
                sample.analyzed_result.value) * (1 - sample.done)
        else:
            trace_target_value = sample.analyzed_result.value * (1 - sample.done)

        # If Vtrace, importance_ratio.size()[0] == sample_length - 1,
        # but will be clipped to sample_length - bootstrap in self._compute_loss
        # If Gae, importance_ratio.size()[0] == sample_length - bootstrap
        if self.vtrace:
            imp_ratio = (analyzed_result.new_action_log_probs - analyzed_result.old_action_log_probs).exp()
        else:
            imp_ratio = None

        adv = modules.gae_trace(sample.reward[:-1],
                                trace_target_value,
                                sample.truncated,
                                sample.done,
                                sample.on_reset,
                                vtrace=self.vtrace,
                                imp_ratio=imp_ratio,
                                gamma=self.discount_rate,
                                lmbda=self.gae_lambda)
        value_target = adv + trace_target_value[:-1]
        return adv, value_target

    def _compute_loss(self, sample, analyzed_result: SampleAnalyzedResult, loss_mask):
        sample_steps = sample.on_reset.shape[0]

        adv = sample.analyzed_result.adv
        value_target = sample.analyzed_result.ret
        old_value = sample.analyzed_result.value if not self.normalize_old_value else self.policy.normalize_value(
            sample.analyzed_result.value)

        # To compute Vtrace, the length of analyzed_result may be larger than sample_steps.
        state_values = analyzed_result.state_values[:sample_steps]
        entropy = analyzed_result.entropy[:sample_steps]
        importance_ratio = (analyzed_result.new_action_log_probs -
                            analyzed_result.old_action_log_probs).exp()[:sample_steps]

        if sample.done is not None and sample.done.sum() > 0:
            logger.debug(
                f"Done values normalized "
                f"{(sample.analyzed_result.value * sample.done).sum().item() / sample.done.sum().item():.4f} "
                f"denormed {(value_target * sample.done).sum().item() / sample.done.sum().item():.4f}")
        if sample.truncated is not None and sample.truncated.sum() > 0:
            logger.debug(
                f"Truncated values normalized "
                f"{(sample.analyzed_result.value * sample.truncated).sum().item() / sample.truncated.sum().item():.4f} "
                f"denormed {(value_target * sample.truncated).sum().item() / sample.truncated.sum().item():.4f}"
            )

        # 3. critic loss
        denorm_value_target = None
        if self.popart:
            denorm_value_target = value_target
            value_target = self.policy.normalize_value(value_target)

        if self.clip_value:
            assert sample.analyzed_result.value is not None
            value_loss = self.value_loss_fn(state_values, old_value, value_target)
        else:
            value_loss = self.value_loss_fn(state_values, value_target)

        value_loss = (value_loss * loss_mask).sum() / loss_mask.sum()

        # 4. actor loss
        norm_adv = modules.masked_normalization(adv, loss_mask)
        surrogate1: torch.Tensor = importance_ratio * norm_adv
        surrogate2: torch.Tensor = torch.clamp(importance_ratio, 1 - self.eps_clip,
                                               1 + self.eps_clip) * norm_adv
        if self.dual_clip:
            surrogate3: torch.Tensor = -torch.sign(norm_adv) * self.c_clip * norm_adv
            policy_loss = -torch.max(torch.min(surrogate1, surrogate2), surrogate3)
        else:
            policy_loss = -torch.min(surrogate1, surrogate2)

        policy_loss = (policy_loss * loss_mask).sum() / loss_mask.sum()

        entropy_loss = -(entropy * loss_mask).sum() / loss_mask.sum()

        # final loss of clipped objective PPO
        loss = policy_loss + self.value_loss_weight * value_loss + self.entropy_bonus_weight * entropy_loss

        loss_mask = loss_mask.to(torch.bool)
        return loss, PPOStepResult(
            advantage=torch.masked_select(adv, loss_mask),
            entropy=-entropy_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            done=sample.done,
            truncated=sample.truncated,
            importance_weight=torch.masked_select(importance_ratio, loss_mask),
            clip_ratio=torch.masked_select((surrogate2 < surrogate1).float(), loss_mask),
            value_targets=torch.masked_select(value_target, loss_mask),
            denorm_value=(torch.masked_select(denorm_value_target, loss_mask)
                          if denorm_value_target is not None else None),
        )

    def step(self, sample):
        from base.timeutil import Timing
        timing = Timing()
        if sample.truncated is None:
            sample.truncated = np.zeros_like(sample.done)
        if self.recompute_adv_on_reuse:
            sample.analyzed_result.adv = sample.analyzed_result.ret = None

        with timing.add_time("ppo/to_device"):
            fetched_data = self.prefetcher.push(sample)
            if fetched_data is None:
                return TrainerStepResult({}, 0)
            else:
                sample, tensor_sample = fetched_data

        # tensor_sample = recursive_apply(
        #     sample, lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=self.policy.device))
        sample_total_length = tensor_sample.on_reset.shape[0]

        train_stats = defaultdict(lambda: 0)

        for _ in range(self.ppo_epochs):
            with timing.add_time("ppo/compute_loss"):
                # for vtrace, we need to compute the importance ratio on each rewarding step
                tail_len = 1 if (self.vtrace and sample.analyzed_result.adv is None) else self.bootstrap_steps
                trainer_analyzed_result = self.policy.analyze(tensor_sample[:sample_total_length - tail_len],
                                                              target="ppo",
                                                              burn_in_steps=self.burn_in_steps)

                # If `bootstrap_steps` equals 0, GAE must have been computed and `tensor_sample.analyzed_result.adv` cannot be None.
                if tensor_sample.analyzed_result.adv is None:
                    dims = len(tensor_sample.analyzed_result.value.shape)
                    # Results of the next line are always 1-step shorter than tensor_sample for both GAE and Vtrace.
                    adv, value_target = self._compute_adv_and_value_target(tensor_sample,
                                                                           trainer_analyzed_result)
                    tensor_sample.analyzed_result.adv = F.pad(adv, (0,) * (dims * 2 - 1) + (1,))
                    sample.analyzed_result.adv = tensor_sample.analyzed_result.adv.cpu().numpy()
                    tensor_sample.analyzed_result.ret = F.pad(value_target, (0,) * (dims * 2 - 1) + (1,))
                    sample.analyzed_result.ret = tensor_sample.analyzed_result.ret.cpu().numpy()

                valid_data = tensor_sample[self.burn_in_steps:sample_total_length - self.bootstrap_steps]
                loss_mask = 1 - tensor_sample.on_reset[1 + self.burn_in_steps:1 + sample_total_length -
                                                       self.bootstrap_steps]

                if self.popart:
                    self.policy.update_popart(valid_data.analyzed_result.ret, mask=loss_mask)

                # st = time.perf_counter()
                # torch.cuda.synchronize()
                # logger.info(f"Cuda synchronization time 1: {time.perf_counter() - st:.3f} secs.")

                loss, step_result = self._compute_loss(valid_data, trainer_analyzed_result, loss_mask)

            with timing.add_time("ppo/backward"):
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # st = time.perf_counter()
                # torch.cuda.synchronize()
                # logger.info(f"Cuda synchronization time 2: {time.perf_counter() - st:.3f} secs.")

                if self.max_grad_norm is not None:
                    grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                else:
                    grad_norm = modules.get_grad_norm(self.policy.parameters())
                self.optimizer.step()

            with timing.add_time("ppo/misc"):
                if self.recompute_adv_among_epochs:
                    tensor_sample.analyzed_result.adv = tensor_sample.analyzed_result.ret = None
                    sample.analyzed_result.adv = sample.analyzed_result.ret = None

                # Reference for the following workaround:
                # https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict
                epoch_stats = {
                    f.name: getattr(step_result, f.name).detach().mean().item()
                    for f in dataclasses.fields(step_result) if getattr(step_result, f.name) is not None
                }
                train_stats['grad_norm'] += grad_norm.mean().item()
                for k, v in epoch_stats.items():
                    train_stats[k] += v

        with timing.add_time("ppo/misc"):
            for k in train_stats:
                train_stats[k] /= self.ppo_epochs

            # Increase the version by 1 instead of `self.ppo_epochs`,
            # because intermediate parameters will not be used for rollout.
            self.policy.inc_version()

            # entropy bonus coefficient decay
            if self.entropy_decay_per_steps and self.policy.version % self.entropy_decay_per_steps == 0:
                self.entropy_bonus_weight *= self.entropy_bonus_decay

            # Logging
            valid_slice = slice(self.burn_in_steps,
                                sample_total_length - self.bootstrap_steps)  # works when bootstrap_steps=0

            self.frames += np.prod(sample.on_reset[valid_slice].shape)

            elapsed_episodes = sample.info_mask[valid_slice].sum()
            if elapsed_episodes == 0:
                info = {}
            else:
                info = recursive_apply(sample.info[valid_slice] * sample.info_mask[valid_slice],
                                       lambda x: x.sum()) / elapsed_episodes

            stats = dict(frames=int(self.frames), **train_stats, **info)
        # logger.info(timing)
        return TrainerStepResult(stats=stats, step=self.policy.version)


register('mappo', MultiAgentPPO)
