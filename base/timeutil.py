from abc import ABC
from typing import Callable, List
import dataclasses
import math
import prometheus_client
import time
import threading

INFINITE_DURATION = 60 * 60 * 24 * 365 * 1000


class FrequencyControl:
    """An utility to control the execution of code with a time or/and step frequency.
    """

    def __init__(self, frequency_seconds=None, frequency_steps=None, initial_value=False):
        """Initialization method of FrequencyControl.
        Args:
            frequency_seconds: Minimal interval between two trigger.
            frequency_steps: Minimal number of steps between two triggers.
            initial_value: In true, the first call of check() returns True.

        NOTE:
            - If both frequency_seconds and frequency_steps are None, the checking will always return False except
             for the specified initial value.
            - If passed both, both frequency and steps conditions have to be met for check() to return True.
            - If one is passed, checking on the other condition will be ignored.
        """
        self.frequency_seconds = frequency_seconds
        self.frequency_steps = frequency_steps
        self.__start_time = time.monotonic()
        self.__steps = 0
        self.__last_time = time.monotonic()
        self.__last_steps = 0
        self.__interval_seconds = self.__interval_steps = None
        self.__initial_value = initial_value
        self.__lock = threading.Lock()

    @property
    def total_seconds(self):
        return time.monotonic() - self.__start_time

    @property
    def total_steps(self):
        return self.__steps

    @property
    def interval_seconds(self):
        return self.__interval_seconds

    @property
    def interval_steps(self):
        return self.__interval_steps

    def check(self, steps=1):
        """Check whether frequency condition is met.
        Args:
            steps: number of step between this and the last call of check()

        Returns:
            flag: True if condition is met, False other wise
        """
        with self.__lock:
            now = time.monotonic()
            self.__steps += steps

            if self.__initial_value:
                self.__last_time = now
                self.__last_steps = self.__steps
                self.__initial_value = False
                return True

            self.__interval_seconds = now - self.__last_time
            self.__interval_steps = self.__steps - self.__last_steps
            if self.frequency_steps is None and self.frequency_seconds is None:
                return False
            if self.frequency_seconds is not None and self.__interval_seconds < self.frequency_seconds:
                return False
            if self.frequency_steps is not None and self.__interval_steps < self.frequency_steps:
                return False
            self.__last_time = now
            self.__last_steps = self.__steps

            return True

    def reset_time(self):
        self.__last_time = time.monotonic()


class PrometheusSummaryObserve:

    def __init__(self, prometheus_metric: prometheus_client.Summary):
        self.metric = prometheus_metric
        self.time = None

    def __enter__(self):
        self.time = time.monotonic_ns()
        return self.time

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time = time.monotonic_ns() - self.time
        self.metric.observe(self.time / 1e9)


@dataclasses.dataclass
class Scheduler(ABC):
    init_value: float
    total_iters: int

    def __post_init__(self):
        if self.total_iters <= 0:
            raise ValueError("total_iters should be a positive number.")

    def get(self, step: int) -> float:
        """Get the scheduled value at the current `step`."""
        if step < 0 or step > self.total_iters:
            raise ValueError(
                f"Scheduler step should be in the interval [0, {self.total_iters}]. Input {step}.")
        return self._get(step)

    def _get(self, step):
        raise NotImplementedError()

    @property
    def final_value(self):
        return self.get(step=self.total_iters)


@dataclasses.dataclass
class ConstantScheduler(Scheduler):

    def _get(self, *args, **kwargs) -> float:
        return self.init_value


@dataclasses.dataclass
class LinearScheduler(Scheduler):
    end_value: float

    def _get(self, step: int) -> float:
        return (self.end_value - self.init_value) / self.total_iters * step + self.init_value


@dataclasses.dataclass
class ExponentialScheduler(Scheduler):
    decay: float

    def _get(self, step: int) -> float:
        return self.init_value * self.decay**step


@dataclasses.dataclass
class CosineDecayScheduler(Scheduler):
    end_value: float

    def __post_init__(self):
        super().__post_init__()
        if self.end_value >= self.init_value:
            raise ValueError("end_value should be smaller than init_value!")

    def _get(self, step: int) -> float:
        delta = self.init_value - self.end_value
        return delta * 0.5 * (1 + math.cos(math.pi / self.total_iters * step)) + self.end_value


@dataclasses.dataclass
class ChainedScheduler:
    schedulers: List[Scheduler]

    @property
    def total_iters(self):
        return sum(x.total_iters for x in self.schedulers)

    @property
    def init_value(self):
        return self.schedulers[0].init_value

    @property
    def final_value(self):
        return self.schedulers[-1].final_value

    def __post_init__(self):
        for i in range(len(self.schedulers) - 1):
            # Float point err 1e-8.
            if abs(self.schedulers[i + 1].get(0) - self.schedulers[i].final_value) > 1e-8:
                raise ValueError(f"Values should be consecutive between "
                                 f"the {i}-th ({type(self.schedulers[i])}) and "
                                 f"the {i+1}-th {type(self.schedulers[i+1])} schedulers! "
                                 f"End value is {self.schedulers[i].final_value} and the "
                                 f"next init value is {self.schedulers[i + 1].get(0)}.")

    def get(self, step: int) -> float:
        for s in self.schedulers:
            if step > s.total_iters:
                step -= s.total_iters
            else:
                return s.get(step)


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)


import time
from collections import deque

EPS = 1e-8


class AvgTime:

    def __init__(self, num_values_to_avg):
        self.values = deque([], maxlen=num_values_to_avg)

    def __str__(self):
        avg_time = sum(self.values) / max(1, len(self.values))
        return f'{avg_time:.4f}'

    @property
    def value(self):
        return sum(self.values) / max(1, len(self.values))


class TimingContext:

    def __init__(self, timer, key, additive=False, average=None):
        self._timer = timer
        self._key = key
        self._additive = additive
        self._average = average
        self._time_enter = None

    def __enter__(self):
        self._time_enter = time.time()

    def __exit__(self, type_, value, traceback):
        if self._key not in self._timer:
            if self._average is not None:
                self._timer[self._key] = AvgTime(num_values_to_avg=self._average)
            else:
                self._timer[self._key] = 0

        time_passed = max(time.time() - self._time_enter, EPS)  # EPS to prevent div by zero

        if self._additive:
            self._timer[self._key] += time_passed
        elif self._average is not None:
            self._timer[self._key].values.append(time_passed)
        else:
            self._timer[self._key] = time_passed


class Timing(AttrDict):

    def timeit(self, key):
        return TimingContext(self, key)

    def add_time(self, key):
        return TimingContext(self, key, additive=True)

    def time_avg(self, key, average=500):
        return TimingContext(self, key, average=average)

    def __str__(self):
        s = ''
        i = 0
        for key, value in self.items():
            str_value = f'{value:.4f}' if isinstance(value, float) else str(value)
            s += f'{key}: {str_value}'
            if i < len(self) - 1:
                s += ', '
            i += 1
        return s
