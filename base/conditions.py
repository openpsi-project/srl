import numpy as np
import queue
import torch

from api import config


class Condition:
    """Defines a condition to be checked.
    """

    def is_met_with(self, data) -> bool:
        """Check whether passed data satisfies this condition.
        Args:
            data[key-value]: data to be checked.
        Returns:
            check_passed (bool)
        """
        raise NotImplementedError()

    def reset(self):
        """Reset condition to initial state.
        """
        raise NotImplementedError()


class SimpleBoundCondition:

    def __init__(self, field, lower_limit=None, upper_limit=None):
        self.field = field
        self.lower_limit = lower_limit or -np.inf
        self.upper_limit = upper_limit or np.inf

    def is_met_with(self, data):
        if self.field not in data:
            raise ValueError(f"target field {self.field} not found when checking condition {self}")
        if isinstance(data[self.field], (np.ndarray, torch.Tensor)):
            return self.lower_limit < data[self.field].mean() < self.upper_limit
        else:
            return self.lower_limit < data[self.field] < self.upper_limit

    def reset(self):
        return

    def __str__(self):
        return f" {self.lower_limit} < {self.field} < {self.upper_limit}"


class ConvergedCondition:

    def __init__(self, value_field, step_field, warmup_step=0, duration=100, confidence=0.9, threshold=1e-2):
        """Check if the target value is converged.
        Args:
            value_field: target value field. E.g., episode_return.
            step_field: step field. E.g., version.
            warmup_step: always return False when step < warmup_step.
            duration: all values within the last duration steps are cached to check convergence.
            confidence: what percentage of cached values to use for convergence check. E.g., if confidence is 
              0.9, then only 90% values are used, the smallest 5% and largest 5% values are ignored.
            threshold: the acceptable difference between the largest and the smallest value within the 
              confidence interval.
        """

        self.value_field = value_field
        self.step_field = step_field
        self.warmup_step = warmup_step
        self.duration = duration
        self.confidence = confidence
        self.threshold = threshold
        self.__head_step = None
        self.__step_queue = None
        self.__value_queue = None

        self.reset()

    def reset(self):
        self.__head_step = None
        self.__step_queue = queue.Queue()
        self.__value_queue = queue.Queue()

    def is_met_with(self, data):
        if self.value_field not in data or self.step_field not in data:
            raise ValueError(f"target field {self.value_field} or {self.step_field} not found when checking "
                             f"condition {self}")
        if isinstance(data[self.step_field], (np.ndarray, torch.Tensor)):
            step = data[self.step_field].mean()
        else:
            step = data[self.step_field]
        if isinstance(data[self.value_field], (np.ndarray, torch.Tensor)):
            value = data[self.value_field].mean()
        else:
            value = data[self.value_field]

        if step < self.warmup_step:
            return False
        self.__step_queue.put(step)
        self.__value_queue.put(value)

        if self.__head_step is None:
            self.__head_step = self.__step_queue.get()
            return False
        if step - self.__head_step < self.duration:
            return False

        # Check convergence.
        values = np.sort(self.__value_queue.queue)
        idx = int(len(values) * (1 - self.confidence) / 2)
        # Not converged, if the last value falls outside the confidence range.
        converged = (max(values[-1 - idx], value) - min(values[idx], value) <= self.threshold)
        # Update queues.
        while step - self.__head_step >= self.duration:
            self.__head_step = self.__step_queue.get()
            self.__value_queue.get()
        return converged

    def __str__(self):
        return f"{self.value_field} converged"


def make(cfg: config.Condition):
    if cfg.type_ == config.Condition.Type.SimpleBound:
        return SimpleBoundCondition(**cfg.args)
    elif cfg.type_ == config.Condition.Type.Converged:
        return ConvergedCondition(**cfg.args)
    else:
        raise NotImplementedError()
