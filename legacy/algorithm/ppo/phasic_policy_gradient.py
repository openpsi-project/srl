from typing import List, Optional
import dataclasses
import numpy as np
import torch
import torch.nn.functional as F

from legacy.algorithm import modules
from api.trainer import SampleBatch, TrainerStepResult, register
from legacy.algorithm.ppo.mappo import MultiAgentPPO
from base.namedarray import recursive_aggregate, recursive_apply, NamedArray
import api.policy


@dataclasses.dataclass
class PPGPhase1AnalyzedResult:
    """ Same as PPO algorithm. The following is needed. aux_value is returned for completeness.
    1. Old / New action probabilities
    2. State Values
    3. Rewards
    4. Entropy
    """
    old_action_log_probs: torch.Tensor  # For chosen actions only, flattened
    new_action_log_probs: torch.Tensor  # for chosen actions only, flattened
    aux_values: torch.Tensor  # Not used for gradient update.
    state_values: torch.Tensor  # len(state_values) = len(new_action_probs) + 1
    reward: torch.Tensor  # len(reward) = len(new_action_probs)
    entropy: Optional[torch.Tensor] = None


@dataclasses.dataclass
class PPGPhase2AnalyzedResult:
    """Data required to compute the Joint Loss during auxiliary phase, as detailed in the PPG paper.
    Entries:
        action_distribution: distribution returned by the current policy. Loss will be computed between this and
            the distribution computed upon entering the auxiliary phase.
        auxiliary_value: Must be something return from the sample backbone of the actor. The auxiliary phase
            degenerates otherwise.
        predicted_value: value returned by the value head.
    """
    action_dists: List[torch.distributions.Distribution]
    auxiliary_value: torch.Tensor
    predicted_value: torch.Tensor


@dataclasses.dataclass
class AuxiliaryStepResult:
    """Key statistics of an auxiliary step.
d    """
    auxiliary_value_loss: torch.Tensor
    value_head_loss: torch.Tensor
    policy_distance: torch.Tensor


class _PPGLocalCache:

    class CacheEntry(NamedArray):
        """PPG stores locally the state and target values.
        """

        def __init__(
            self,
            sample: SampleBatch,
            # obs: Observation
            # policy_state: algorithm.policy.PolicyState
            # on_reset: np.ndarray
            # done: np.ndarray
            # value_targets: np.ndarray
            action_dists: Optional[List[torch.distributions.Distribution]] = None):
            super().__init__(sample=sample, action_dists=action_dists)

    @property
    def full(self):
        return len(self.__cache) == self.__cache_size

    @property
    def data(self):
        return self.__cache

    def clear(self):
        self.__cache = []

    def __init__(self, cache_size, stack_size: Optional[int] = None):
        """Initialization Method of class _PPGLocalCache.
        Args:
            cache_size (int): how many batches to be stored, such that OOM can be avoided.
            stack_size (int): whether to batch the local cache
        """
        self.__cache_size = cache_size
        self.__stack_size = stack_size
        self.__cache = []

    def put(self, sample):
        while len(self.__cache) >= self.__cache_size:
            self.__cache.pop(0)
        self.__cache.append(sample)

    def get(self):
        if self.__stack_size:
            return [
                recursive_aggregate(self.__cache[i:i + self.__stack_size],
                                    lambda x: np.concatenate(x, axis=1))
                for i in range(0, len(self.__cache), self.__stack_size)
            ]
        else:
            return self.__cache


class MultiAgentPPG(MultiAgentPPO):
    """Phasic Policy Gradient Algorithm
    Ref: Phasic Policy Gradient (https://arxiv.org/abs/2009.04416)

    Note:
        Phasic Policy Gradient is built on top of PPO algorithm. Inheritance between algorithms are generally
        discouraged.

    TODO: Current implementation is slow due to unnecessary reshaping and type conversion.
    """

    def load_checkpoint(self, checkpoint, **kwargs):
        super(MultiAgentPPG, self).load_checkpoint(checkpoint)
        if "aux_optimizer_state_dict" in checkpoint.keys():
            self.ppg_aux_optimizer.load_state_dict(checkpoint["aux_optimizer_state_dict"])

    def get_checkpoint(self):
        checkpoint = super(MultiAgentPPG, self).get_checkpoint()
        checkpoint.update({"aux_optimizer_state_dict": self.ppg_aux_optimizer.state_dict()})
        return checkpoint

    def __init__(self, policy: api.policy.Policy, **kwargs):
        super(MultiAgentPPG, self).__init__(policy, **kwargs)
        self.ppo_optimizer = self.optimizer
        self.__ppo_epochs = kwargs.get("ppo_epochs", 5)
        self.__ppo_iterations = kwargs.get("ppo_iterations", 32)
        self.__ppg_epochs = kwargs.get("ppg_epochs", 6)
        self.__beta_clone = kwargs.get("beta_clone", 1)
        self.__value_head_weight = kwargs.get("aux_value_head_weight", 1)
        self.__local_cache_stack_size = kwargs.get("local_cache_stack_size", None)
        self.__cache = _PPGLocalCache(cache_size=self.__ppo_iterations,
                                      stack_size=self.__local_cache_stack_size)
        optimizer_name = kwargs.get("ppg_optimizer", "adam")
        self.ppg_aux_optimizer = modules.init_optimizer(self.policy.parameters(), optimizer_name,
                                                        kwargs.get("ppg_optimizer_config", {}))
        self._kld = torch.distributions.kl_divergence
        self._mse = torch.nn.MSELoss(reduction="none")

    def _policy_distance(self, dist1, dist2, done):
        """KL-Divergence
        """
        undone = (1. - done[..., 0])
        return (self._kld(dist1, dist2) * undone).sum() / undone.sum()

    def _paper_value_loss(self, value, target_value, done):
        return 1 / 2 * (self._mse(value, target_value) * (1. - done)).sum() / (1. - done).sum()

    def step(self, sample: SampleBatch):
        """One step of PPG update.
        One sample first goes through several PPO epochs before it is added to local cache.
        When entering the auxiliary phase, the algorithm analyze every sample in its local cache.
        """
        # Phase 1
        tensor_sample = recursive_apply(
            sample, lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=self.policy.device))
        sample.adv = None
        for _ in range(self.ppo_epochs):
            tail_len = 1 if (self.vtrace and sample.adv is None) else self.bootstrap_steps
            phase_1_analyzed_result = self.policy.analyze(tensor_sample[:-tail_len], target="ppg_ppo_phase")
            if sample.adv is None:
                dims = len(tensor_sample.value.shape)
                tensor_sample.adv = F.pad(self._compute_adv(tensor_sample, phase_1_analyzed_result),
                                          (0,) * (dims * 2 - 1) + (1,))
                sample.adv = tensor_sample.adv.cpu().numpy()
            done = tensor_sample.on_reset[1:]

            ppo_loss, phase1_metrics = self._compute_loss(tensor_sample[:-self.bootstrap_steps],
                                                          tensor_sample.adv, phase_1_analyzed_result, done)
            # Only to make DDP work.
            thisisnotaloss = self._mse(phase_1_analyzed_result.aux_values,
                                       phase_1_analyzed_result.aux_values.detach())
            phase_1_loss = ppo_loss + thisisnotaloss.mean()

            self.optimizer.zero_grad()
            phase_1_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.policy.inc_version()

        # Put data into local cache.
        train_episode_length = sample.on_reset.shape[0] - self.bootstrap_steps
        if self.__ppg_epochs > 0:
            store_value_target = phase1_metrics.denorm_value if self.popart else phase1_metrics.value_targets

            self.__cache.put(
                _PPGLocalCache.CacheEntry(
                    sample=SampleBatch(
                        obs=sample.obs[:train_episode_length],
                        policy_state=sample.policy_state[:train_episode_length],
                        info_mask=sample.info_mask[:train_episode_length],
                        on_reset=sample.on_reset[:train_episode_length],
                        value=store_value_target.cpu().detach().numpy(),
                    ),
                    action_dists=None,
                ))

        phase2_metrics = None
        if self.__cache.full:
            # Prepare cached data for auxiliary phase.
            for entry in self.__cache.data:
                tensor_sample = recursive_apply(
                    entry.sample,
                    lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=self.policy.device))
                phase_2_analyzed_result: PPGPhase2AnalyzedResult = self.policy.analyze(tensor_sample,
                                                                                       target="ppg_aux_phase")
                entry.action_dists = [
                    modules.distribution_detach_to_cpu(d) for d in phase_2_analyzed_result.action_dists
                ]
                if self.popart:
                    # If value head is popart, the stored target_values are denormalized version.
                    entry.sample.value = self.policy.normalize_value(
                        torch.from_numpy(entry.sample.value).to(self.policy.device)).cpu().detach().numpy()

            # Phase 2
            for cache_sample in self.__cache.get():
                for _ in range(self.__ppg_epochs):
                    tensor_sample = recursive_apply(
                        cache_sample.sample,
                        lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=self.policy.device))
                    ppg_analyze_result: PPGPhase2AnalyzedResult = self.policy.analyze(tensor_sample,
                                                                                      target="ppg_aux_phase")
                    aux_loss, phase2_metrics = self._compute_aux_loss(cache_sample, ppg_analyze_result)
                    self.ppg_aux_optimizer.zero_grad()
                    aux_loss.backward()
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.ppg_aux_optimizer.step()
                    self.policy.inc_version()

            self.__cache.clear()

        elapsed_episodes = sample.info_mask.sum()
        if elapsed_episodes == 0:
            info = {}
        else:
            info = recursive_apply(sample.info * sample.info_mask, lambda x: x.sum()) / elapsed_episodes
        # TODO:  len(sample) equals sample_steps + bootstrap_steps, which is incorrect for frames logging
        stats = dict(**info)
        if phase1_metrics is not None:
            stats.update(
                dict(
                    **{
                        f"ppo_{f.name}": getattr(phase1_metrics, f.name).detach().mean().item()
                        for f in dataclasses.fields(phase1_metrics)
                        if getattr(phase1_metrics, f.name) is not None
                    }))
        if phase2_metrics is not None:
            stats.update(
                dict(
                    **{
                        f"ppg_{f.name}": getattr(phase2_metrics, f.name).detach().mean().item()
                        for f in dataclasses.fields(phase2_metrics)
                        if getattr(phase2_metrics, f.name) is not None
                    }))
        return TrainerStepResult(stats=stats, step=self.policy.version)

    def _compute_aux_loss(self, entry: _PPGLocalCache.CacheEntry,
                          ppg_analyze_result: PPGPhase2AnalyzedResult):
        d_ = ppg_analyze_result.predicted_value
        value_target = torch.from_numpy(entry.sample.value).to(d_)
        done = torch.from_numpy(entry.sample.info_mask).to(d_)
        policy_distance = torch.sum(
            torch.stack([
                self._policy_distance(modules.distribution_to(d1, d_), d2, done=done)
                for d1, d2 in zip(entry.action_dists, ppg_analyze_result.action_dists)
            ]))
        aux_value_loss = self._paper_value_loss(ppg_analyze_result.auxiliary_value, value_target, done)
        value_head_loss = self._paper_value_loss(ppg_analyze_result.predicted_value, value_target, done)
        aux_loss = aux_value_loss + self.__beta_clone * policy_distance + \
                   self.__value_head_weight * value_head_loss
        return aux_loss, AuxiliaryStepResult(auxiliary_value_loss=aux_value_loss,
                                             value_head_loss=value_head_loss,
                                             policy_distance=policy_distance)


register("mappg", MultiAgentPPG)
