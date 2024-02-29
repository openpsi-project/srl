from typing import Optional, Union, Dict, List
import dataclasses
import torch.nn as nn
import torch

import logging

from legacy.algorithm.modules import init_optimizer
from legacy.algorithm.muzero.utils.utils import adjust_lr, consist_loss_func, get_grad_norm
from api.trainer import PytorchTrainer, register, TrainerStepResult, PyTorchGPUPrefetcher
from api.policy import Policy
from base.namedarray import recursive_apply
import time

logger = logging.getLogger("MuZero trainer")


@dataclasses.dataclass
class MuzeroSampleAnalyzedResult:
    """Muzero loss computation requires:
    1. value
    2. value prefix
    3. policy logits
    
    EfficientZero additionally requires:
    4. dynamic state projection
    5. present observation projection

    These tensors are of shape (batch_size, num_unroll_steps+1, *D)
    """
    value: torch.Tensor
    target_value: torch.Tensor
    value_prefix: torch.Tensor
    target_value_prefix: torch.Tensor
    policy_logits: torch.Tensor
    policy_entropy: torch.Tensor
    target_policy: torch.Tensor
    mask: torch.Tensor
    dynamic_proj: Union[torch.Tensor, None]
    observation_proj: Union[torch.Tensor, None]


@dataclasses.dataclass
class MuZeroStepResult:
    # value
    target_value: torch.Tensor
    scaled_value: torch.Tensor

    # value prefix
    target_value_prefix: torch.Tensor
    scaled_value_prefix: torch.Tensor

    # policy entropy
    entropy: torch.Tensor
    target_entropy: torch.Tensor

    # Loss
    consistency_loss: Optional[Union[torch.Tensor, float]]
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    value_prefix_loss: torch.Tensor
    entropy_loss: torch.Tensor
    total_loss: torch.Tensor


class MuZero(PytorchTrainer):
    """
    MuZero 
    Ref: 
        Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model
        https://www.nature.com/articles/s41586-020-03051-4
    
    EfficientZero 
    Ref: 
        Mastering Atari Games with Limited Data
        http://arxiv.org/abs/2111.00210
        
        This implemention heavily referenced EfficientZero released code: 
        https://github.com/YeWR/EfficientZero
    """

    def get_checkpoint(self):
        checkpoint = self.policy.get_checkpoint()
        checkpoint.update({"optimizer_state_dict": self.optimizer.state_dict()})
        return checkpoint

    def load_checkpoint(self, checkpoint, **kwargs):
        if "optimizer_state_dict" in checkpoint.keys():
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.policy.load_checkpoint(checkpoint=checkpoint)

    def __init__(
        self,
        policy: Policy,
        # tree search
        num_unroll_steps: int = 5,
        td_steps: int = 5,
        # optimization
        optimizer_name: str = "sgd",
        optimizer_config: dict = dict(lr=0.2, weight_decay=1e-4, momentum=0.9),
        lr_schedule: List[Dict] = None,
        max_grad_norm: float = 5,
        # self-supervised model
        do_consistency: bool = False,
        # loss
        start_train_value: int = 0,
        reward_loss_coeff: float = 1,
        value_loss_coeff: float = 1,
        policy_loss_coeff: float = 1,
        consistency_coeff: float = 1,
        entropy_coeff: float = 0.,
    ):
        super().__init__(policy)

        # tree search
        self.num_unroll_steps = num_unroll_steps
        self.td_steps = td_steps
        # optimizer
        self.lr_schedule = lr_schedule or [
            dict(name="linear", num_steps=1000, config=dict(lr_init=0., lr_end=0.2)),
            dict(name="decay", num_steps=2e5, config=dict(lr_init=0.2, lr_decay_rate=0.1,
                                                          lr_decay_steps=1e5)),
        ]
        self.optimizer = init_optimizer(self.policy.parameters(), optimizer_name, optimizer_config)
        # gradient normalization
        self.max_grad_norm = max_grad_norm
        # self-supervised model
        self.do_consistency = do_consistency
        # loss
        self.start_train_value = start_train_value
        self.reward_loss_coeff = reward_loss_coeff
        self.value_loss_coeff = value_loss_coeff
        self.policy_loss_coeff = policy_loss_coeff
        self.consistency_coeff = consistency_coeff
        self.entropy_coeff = entropy_coeff
        # value rescale
        self.inverse_reward_transform = self.policy.inverse_reward_transform
        self.inverse_value_transform = self.policy.inverse_value_transform
        # value & value prefix loss
        self.scalar_reward_loss = self.policy.scalar_reward_loss
        self.scalar_value_loss = self.policy.scalar_value_loss

        self.prefetcher = PyTorchGPUPrefetcher()

    def _compute_loss(self, analyzed_result):
        length = self.num_unroll_steps + 1
        batch_size = analyzed_result.mask.shape[1]

        scaled_value = self.inverse_value_transform(analyzed_result.value)
        scaled_value_prefix = self.inverse_reward_transform(analyzed_result.value_prefix)
        policy_logits = analyzed_result.policy_logits  # softmax for discrete action space should be taken in the network for consistency between discrete and continuous action space
        policy_entropy = analyzed_result.policy_entropy

        mask = analyzed_result.mask

        if self.policy.version <= self.start_train_value:
            analyzed_result.target_value = torch.zeros_like(analyzed_result.target_value)

        value_loss = (self.scalar_value_loss(analyzed_result.value,
                                             analyzed_result.target_value).reshape(length, batch_size,
                                                                                   1)).sum(dim=0)
        policy_loss = (
            -(policy_logits * analyzed_result.target_policy).sum(dim=-1).reshape(length, batch_size, 1) *
            mask).sum(dim=0)
        # ignore the initial step, which is always 0
        value_prefix_loss = (self.scalar_reward_loss(analyzed_result.value_prefix,
                                                     analyzed_result.target_value_prefix).reshape(
                                                         length, batch_size, 1)[1:]).sum(dim=0)
        if self.do_consistency:
            consistency_loss = (
                consist_loss_func(analyzed_result.dynamic_proj, analyzed_result.observation_proj).reshape(
                    length, batch_size, 1) * mask).sum(dim=0)
        else:
            consistency_loss = 0

        entropy_loss = -(policy_entropy.reshape(length, batch_size, 1) * mask).sum(dim=0)

        loss = self.policy_loss_coeff * policy_loss + self.reward_loss_coeff * value_prefix_loss + self.consistency_coeff * consistency_loss + self.value_loss_coeff * value_loss + self.entropy_coeff * entropy_loss
        loss = loss.mean()  # average over batch size, as consistent as EfficientZero

        step_result = MuZeroStepResult(
            # value
            target_value=analyzed_result.target_value.mean(),
            scaled_value=scaled_value.mean(),

            # value prefix
            target_value_prefix=analyzed_result.target_value_prefix.mean(),
            scaled_value_prefix=scaled_value_prefix.mean(),

            # policy
            entropy=policy_entropy.mean(),
            target_entropy=-(
                (analyzed_result.target_policy * torch.log(analyzed_result.target_policy + 1e-9)).sum(
                    dim=-1)).mean(),  # entropy of the result of MCTS result

            # Loss
            consistency_loss=consistency_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            value_prefix_loss=value_prefix_loss,
            entropy_loss=entropy_loss,
            total_loss=loss)
        return loss, step_result

    def step(self, sample):
        lr = adjust_lr(self.lr_schedule, self.optimizer, self.policy.version)

        tik = time.perf_counter()
        fetched_data = self.prefetcher.push(sample)
        if fetched_data is None:
            return TrainerStepResult({}, 0)
        else:
            sample, tensor_sample = fetched_data
        t0 = time.perf_counter()

        analyzed_result: MuzeroSampleAnalyzedResult = self.policy.analyze(tensor_sample,
                                                                          do_consistency=self.do_consistency)
        t1 = time.perf_counter()
        loss, step_result = self._compute_loss(analyzed_result)
        t2 = time.perf_counter()

        gradient_scale = 1 / self.num_unroll_steps
        loss.register_hook(lambda grad: grad * gradient_scale)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        t3 = time.perf_counter()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        t4 = time.perf_counter()

        # representation_grad_norm = get_grad_norm(self.policy.net.module.representation_network.parameters())
        # dynamics_grad_norm = get_grad_norm(self.policy.net.module.dynamics_network.parameters())
        # prediction_grad_norm = get_grad_norm(self.policy.net.module.prediction_network.parameters())

        self.policy.inc_version()

        elapsed_episodes = sample.info_mask.sum()
        if elapsed_episodes == 0:
            info = {}
        else:
            info = recursive_apply(sample.info * sample.info_mask, lambda x: x.sum()) / elapsed_episodes
        # Reference for the following workaround:
        # https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict
        stats = dict(
            **{
                f.name: getattr(step_result, f.name).detach().mean().item() if isinstance(
                    getattr(step_result, f.name), torch.Tensor) else getattr(step_result, f.name)
                for f in dataclasses.fields(step_result) if getattr(step_result, f.name) is not None
            },
            **info,
            lr=lr,
            # representation_grad_norm=representation_grad_norm,
            # dynamics_grad_norm=dynamics_grad_norm,
            # prediction_grad_norm=prediction_grad_norm,
        )
        t5 = time.perf_counter()
        logger.info(
            f"Train step total: {t5-tik:.3f}. To device: {t0 - tik:.3f}, analyze: {t1 - t0:.3f}, "
            f"Compute loss: {t2 - t1:.3f}, backward: {t3 - t2:.3f}, step: {t4 - t3:.3f}, misc: {t5 - t4:.3f}")
        return TrainerStepResult(stats=stats, step=self.policy.version)


register('muzero', MuZero)
