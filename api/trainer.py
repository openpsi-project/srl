from abc import ABC
from typing import Union, Dict, Optional, List
import dataclasses
import numpy as np
import torch
import torch.distributed as dist

from base.namedarray import NamedArray, recursive_apply
from api.environment import Action
import api.policy
import api.config


class SampleBatch(NamedArray):
    # `SampleBatch` is the general data structure and will be used for ALL the algorithms we implement.
    # There could be some entries that may not be used by a specific algorithm,
    # e.g. log_probs and value are not used by DQN, which will be left as None.

    # `obs`, `on_reset`, `action`, `reward`, and `info` are environment-related data entries.
    # `obs` and `on_reset` can be obtained once environment step is preformed.

    def __init__(
            self,
            obs: NamedArray,
            on_reset: np.ndarray = None,
            done: np.ndarray = None,
            truncated: np.ndarray = None,

            # `action` and `reward` can be obtained when the inference is done.
            action: Action = None,
            reward: np.ndarray = None,

            # Currently we assume info contains all the information we want to gather in an environment.
            # It is NOT agent-specific and should include summary information of ALL agents.
            info: NamedArray = None,

            # `info_mask` is recorded for correctly recording summary info when there are
            # multiple agents and some agents may die before an episode is done.
            info_mask: np.ndarray = None,

            # In some cases we may need Policy State. e.g. Partial Trajectory, Mu-Zero w/o reanalyze.
            policy_state: api.policy.PolicyState = None,

            # `analyzed_result` records algorithm-related analyzed results
            analyzed_result: api.policy.AnalyzedResult = None,

            # Policy-ralted infos.
            policy_name: np.ndarray = None,
            policy_version_steps: np.ndarray = None,
            actor_worker_post_timestamp: np.ndarray = None,
            actor_worker_flush_timestamp: np.ndarray = None,
            trainer_worker_recv_timestamp: np.ndarray = None,
            trainer_worker_batch_timestamp: np.ndarray = None,

            # timestamps
            send_timestamp: np.ndarray = None,
            buffer_recv_timestamp: np.ndarray = None,

            # Metadata.
            sampling_weight: np.ndarray = None,
            **kwargs):
        super(SampleBatch, self).__init__(
            obs=obs,
            on_reset=on_reset,
            done=done,
            truncated=truncated,
            action=action,
            reward=reward,
            info=info,
            info_mask=info_mask,
            policy_state=policy_state,
            analyzed_result=analyzed_result,
            policy_name=policy_name,
            policy_version_steps=policy_version_steps,
            send_timestamp=send_timestamp,
            buffer_recv_timestamp=buffer_recv_timestamp,
            actor_worker_post_timestamp=actor_worker_post_timestamp,
            actor_worker_flush_timestamp=actor_worker_flush_timestamp,
            trainer_worker_recv_timestamp=trainer_worker_recv_timestamp,
            trainer_worker_batch_timestamp=trainer_worker_batch_timestamp,
        )
        self.register_metadata(sampling_weight=sampling_weight,)


class TrajPostprocessor(ABC):
    """Post-process trajectories in actor workers before sending to trainers.
    
    Basically computing returns, e.g., GAE or n-step return.
    """

    def process(self, memory: List[SampleBatch]):
        raise NotImplementedError()


class NullTrajPostprocessor(TrajPostprocessor):

    def process(self, memory: List[SampleBatch]):
        return memory


@dataclasses.dataclass
class TrainerStepResult:
    stats: Dict  # Stats to be logged.
    step: int  # current step count of trainer.
    agree_pushing: Optional[bool] = True  # whether agree to push parameters
    priorities: Optional[np.ndarray] = None  # New priorities of the PER buffer.


class Trainer:

    @property
    def policy(self) -> api.policy.Policy:
        """Running policy of the trainer.
        """
        raise NotImplementedError()

    def step(self, samples: SampleBatch) -> TrainerStepResult:
        """Advances one training step given samples collected by actor workers.

        Example code:
          ...
          some_data = self.policy.analyze(sample)
          loss = loss_fn(some_data, sample)
          self.optimizer.zero_grad()
          loss.backward()
          ...
          self.optimizer.step()
          ...

        Args:
            samples (SampleBatch): A batch of data required for training.

        Returns:
            TrainerStepResult: Entry to be logged by trainer worker.
        """
        raise NotImplementedError()

    def distributed(self, **kwargs):
        """Make the trainer distributed.
        """
        raise NotImplementedError()

    def get_checkpoint(self, *args, **kwargs):
        """Get checkpoint of the model, which typically includes:
        1. Policy state (e.g. neural network parameter).
        2. Optimizer state.
        Return:
            checkpoint to be saved.
        """
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint, **kwargs):
        """Load a saved checkpoint.
        Args:
            checkpoint: checkpoint to be loaded.
        """
        raise NotImplementedError()


class PytorchTrainer(Trainer, ABC):

    @property
    def policy(self) -> api.policy.Policy:
        return self._policy

    def __init__(self, policy: api.policy.Policy):
        """Initialization method of Pytorch Trainer.
        Args:
            policy: Policy to be trained.

        Note:
            After initialization, access policy from property trainer.policy
        """
        if policy.device != "cpu":
            torch.cuda.set_device(policy.device)
            torch.cuda.empty_cache()
        self._policy = policy

    def distributed(self, rank, world_size, init_method, **kwargs):
        is_gpu_process = all([
            torch.distributed.is_nccl_available(),
            torch.cuda.is_available(),
            self.policy.device != "cpu",
        ])
        dist.init_process_group(backend="nccl" if is_gpu_process else "gloo",
                                init_method=init_method,
                                rank=rank,
                                world_size=world_size)
        self.policy.distributed()

    def __del__(self):
        if dist.is_initialized():
            dist.destroy_process_group()


from base.namedarray import recursive_apply


class PyTorchGPUPrefetcher:
    """Prefetch sample into GPU in trainer.
    
    Reference: https://github.com/NVIDIA/apex/blob/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet/main_amp.py#L256.
    """

    def __init__(self):
        self.stream = torch.cuda.Stream()
        self.nex_numpy_sample = None
        self.nex_torch_sample = None
        self.initialized_prefetching = False

    def _preload(self, sample):
        self.nex_numpy_sample = sample
        with torch.cuda.stream(self.stream):
            # NOTE: Use `.to(device)` instead of `.cuda` will not accerlate data loading.
            self.nex_torch_sample = recursive_apply(self.nex_numpy_sample,
                                                    lambda x: torch.from_numpy(x).cuda(non_blocking=True))
            self.nex_torch_sample = recursive_apply(self.nex_torch_sample, lambda x: x.float())

    def push(self, sample):
        if not self.initialized_prefetching:
            self._preload(sample)
            self.initialized_prefetching = True
            return None
        torch.cuda.current_stream().wait_stream(self.stream)
        numpy_sample = self.nex_numpy_sample
        torch_sample = self.nex_torch_sample
        self._preload(sample)
        return numpy_sample, torch_sample


ALL_TRAINER_CLASSES = {}


def register(name, trainer_class):
    ALL_TRAINER_CLASSES[name] = trainer_class


def make(cfg: Union[str, api.config.Trainer], policy_cfg: Union[str, api.config.Policy]) -> Trainer:
    if isinstance(cfg, str):
        cfg = api.config.Trainer(type_=cfg)
    if isinstance(policy_cfg, str):
        policy_cfg = api.config.Policy(type_=policy_cfg)
    cls = ALL_TRAINER_CLASSES[cfg.type_]
    policy = api.policy.make(policy_cfg)
    policy.train_mode()  # To be explicit.
    return cls(policy=policy, **cfg.args)


ALL_TRAJ_POSTPROCESSOR_CLASSES = {}


def register_traj_postprocessor(name, cls_):
    ALL_TRAJ_POSTPROCESSOR_CLASSES[name] = cls_


register_traj_postprocessor('null', NullTrajPostprocessor)


def make_traj_postprocessor(cfg: Union[str, api.config.TrajPostprocessor]):
    if isinstance(cfg, str):
        cfg = api.config.TrajPostprocessor(cfg)
    augmenter_type = cfg if isinstance(cfg, str) else cfg.type_
    cls = ALL_TRAJ_POSTPROCESSOR_CLASSES[augmenter_type]
    return cls(**cfg.args)
