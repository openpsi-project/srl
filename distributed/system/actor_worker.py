from typing import Dict, List, Optional, Tuple, Callable
import copy
import dataclasses
import enum
import logging
import numpy as np
import re
import threading
import time
import random

from api.curriculum import make as make_curriculum, Curriculum
from base.namedarray import recursive_aggregate
from base.timeutil import FrequencyControl
from base.timeutil import PrometheusSummaryObserve as ObserveTime
import api.config
import api.policy
import api.trainer
import api.environment as env_base
import base.network
import distributed.system.worker_base as worker_base
import distributed.system.inference_stream as inference_stream
import distributed.system.sample_stream as sample_stream
import distributed.base.monitoring

_MAX_POLL_STEPS = 16
_INFERENCE_FLUSH_FREQUENCY_SECONDS = 0.01


class AgentStateError(Exception):

    def __init__(self, state, method_name):
        super().__init__(f"Agent cannot {method_name} when in state {state}.")


class AgentState(enum.Enum):
    """The internal state of an Agent to track method invocations.

    Agent methods must be called in the following sequence:
                           observe()          ready_to_step()       consume_inference_result()
        READY_FOR_OBSERVATION → WAITING_FOR_ACTION → UNCOMSUMED_ACTION_READY → READY_FOR_OBSERVATION → ...
    get_action() can be called after calling consume_inference_result().

    Calls in other orders will raise AgentStateError.
    """
    READY_FOR_OBSERVATION = 0
    WAITING_FOR_ACTION = 1
    UNCOMSUMED_ACTION_READY = 2


logger = logging.getLogger('aw')

# def n_step_return(n: int,
#                   reward,
#                   nex_value,
#                   nex_done,
#                   nex_truncated,
#                   gamma: float,
#                   high_precision: Optional[bool] = True):
#     if high_precision:
#         reward, nex_value, nex_done, nex_truncated = map(lambda x: x.astype(np.float64),
#                                                          [reward, nex_value, nex_done, nex_truncated])

#     T = nex_value.shape[0] - n + 1
#     assert T >= 1
#     ret = np.zeros_like(reward[:T])
#     discount = np.ones_like(reward[:T])
#     for i in range(n):
#         ret += reward[i:i + T] * discount
#         # If the next step is truncated, bootstrap value in advance.
#         # In following iterations, `discount` will be 0 and truncated returns will not be updated.
#         ret += discount * gamma * nex_truncated[i:i + T] * nex_value[i:i + T]
#         discount *= gamma * (1 - nex_done[i:i + T]) * (1 - nex_truncated[i:i + T])
#     return (ret + discount * nex_value[n - 1:n - 1 + T]).astype(np.float32)

# def scalar_transform(x: np.ndarray, eps=1e-3):
#     # TODO: unify scalar transform in muzero implmentation
#     x = x.astype(np.float64)
#     y = np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + eps * x
#     return y.astype(np.float32)

# def inverse_scalar_transform(x: np.ndarray, eps=1e-3):
#     x = x.astype(np.float64)
#     a = np.sqrt(1 + 4 * eps * (np.abs(x) + 1 + eps)) - 1
#     y = np.sign(x) * (np.square(a / (2 * eps)) - 1)
#     return y.astype(np.float32)


@dataclasses.dataclass
class _AgentSampleFlow:
    """This class defines when/how data is packed and sent to trainer workers."""
    send_after_done: bool
    send_full_trajectory: bool
    sample_steps: int  # effective only when send_full_trajectory=False
    bootstrap_steps: int
    burn_in_steps: int  # used to warm up policy states
    pad_trajectory: bool  # effective only when send_full_trajectory=True
    env_max_num_steps: int  # effective only when pad_trajectory=True
    traj_process_fn: Optional[Callable[[List[api.trainer.SampleBatch]], List[
        api.trainer.SampleBatch]]] = lambda x: x  # Process trajectories before sending (e.g. computing GAE).

    def __post_init__(self):
        if self.sample_steps <= 0:
            raise ValueError("Sample steps should be a positive number.")
        if self.bootstrap_steps < 0:
            raise ValueError("Bootstrap steps should be a non-negative number.")
        if self.burn_in_steps < 0:
            raise ValueError("Burn-in length should be a non-negative number.")
        if self.send_full_trajectory and self.burn_in_steps != 0:
            raise ValueError("Burn-in should be turned off if we send full trajectories!")
        # Fill memory with empty sample batches.
        # Entries will be filled with 0 when calling recursive_aggregate in `self.get`.
        self.__memory = [api.trainer.SampleBatch(obs=None) for _ in range(self.burn_in_steps)]
        self.__traj_cache = []

    def __pad_memory_to(self, target_length: int):
        if len(self.__memory) > target_length:
            raise ValueError("Padding target length is smaller than memory length!")
        # Entries will be filled with 0 when calling recursive_aggregate in `self.get`.
        last_step = api.trainer.SampleBatch(obs=None)
        last_step.done = np.ones_like(
            self.__memory[-1].done)  # such that padded steps will be masked in loss computation
        self.__memory += [last_step] * (target_length - len(self.__memory))

    def push(self, sample: api.trainer.SampleBatch):
        """Adds a new sample step.
        Args:
            sample: new step to be added to memory.
        """
        if not (self.send_full_trajectory or self.send_after_done):
            # we eagerly send batches without waiting for episode finish
            # e.g. when the episode is too long
            self.__memory.append(sample)
        else:
            self.__traj_cache.append(sample)
            if np.logical_or(sample.truncated, sample.done).all():  # agent finish
                # this cannot be garbage done step because such steps will not be returned by _AgentMemoryCache
                self.__memory += self.traj_process_fn(self.__traj_cache)
                self.__traj_cache = []

    def get(self, on_reset: bool):
        """Gets a sample to be sent.
        Args:
            on_reset: meant to be used with send-full-trajectory and send-after-done. If true, the SampleFlow will
            trigger all end-of-episode logic.
        Returns:
            samples, as aggregated namedarray; None if no sample is ready.
        """
        sample = None
        if not self.send_full_trajectory:  # we send batch
            if not self.send_after_done or (self.send_after_done and on_reset):
                traj_steps = self.burn_in_steps + self.sample_steps + self.bootstrap_steps
                if len(self.__memory) >= traj_steps:
                    sample = recursive_aggregate(self.__memory[:traj_steps], np.stack)
                    self.__memory = self.__memory[self.sample_steps:]
        else:  # we send full trajectories
            if on_reset and len(self.__memory) > 0:  # only send when the trajectory is complete
                if self.pad_trajectory and len(self.__memory) < self.env_max_num_steps:
                    self.__pad_memory_to(self.env_max_num_steps + self.bootstrap_steps)
                sample = recursive_aggregate(self.__memory, np.stack)
                self.__memory = []

        ######################################## special design for r2d2 ########################################
        # if sample is not None and sample.length(0) > self.bootstrap_steps:
        #     ret = scalar_transform(
        #         n_step_return(self.bootstrap_steps,
        #                       reward=sample.reward[:-1],
        #                       nex_value=inverse_scalar_transform(sample.analyzed_result.target_value[1:]),
        #                       nex_done=sample.done[1:],
        #                       nex_truncated=sample.truncated[1:],
        #                       gamma=0.997,
        #                       high_precision=True))
        #     # logger.info(f"1 sample length: {sample.length(0)}, return shape {ret.shape}")
        #     sample.analyzed_result.ret = np.zeros_like(sample.analyzed_result.target_value)
        #     sample = sample[:self.burn_in_steps + self.sample_steps]
        #     # logger.info(f"sample length: {sample.length(0)}, return shape {ret.shape}")
        #     sample.analyzed_result.ret[:] = ret
        #     assert sample.length(0) == self.burn_in_steps + self.sample_steps, sample.length(0)
        ######################################## special design for r2d2 ########################################
        return sample


@dataclasses.dataclass
class _AgentInferenceMaker:
    """This class defines when/how an inference request is posted to policy workers."""
    deterministic_action: bool

    def make_inference_request(
        self,
        env_result: env_base.StepResult,
        policy_state: api.policy.PolicyState,
        step_count: int,
        on_reset: bool,
    ):
        """Wraps the observation, and other state variables as a rollout request.

        Args:
            env_result: api.environment.StepResult
            policy_state: current policy state of the agent.
            step_count: the step count of the environment. Note that in async environment, this is not equal to the
                number of actions that this agent has made.
            on_reset: whether this is the first step of a new episode. Meant to be used when policy has a memory state.

        Returns:
            RolloutRequest, which is ready to be sent to the inference stream.

        """
        is_episode_finished = np.logical_or(env_result.done, env_result.truncated).all()
        is_consecutive_finish = is_episode_finished and on_reset

        if (is_episode_finished and env_result.done.all()) or is_consecutive_finish:
            # Don't issue a request if the end step is a ``done`` step or a finish step (i.e., done or truncated) after done.
            return None

        obs = base.namedarray.from_dict(env_result.obs)
        base_shape = env_result.reward.shape[:-1]
        request = api.policy.RolloutRequest(
            obs=obs,
            policy_state=policy_state,
            is_evaluation=np.full((*base_shape, 1), fill_value=self.deterministic_action, dtype=np.uint8),
            on_reset=np.full((*base_shape, 1), fill_value=on_reset, dtype=np.uint8),
            step_count=np.full((*base_shape, 1), fill_value=step_count, dtype=np.int32),
        )
        return request


@dataclasses.dataclass
class _AgentMemoryCache:
    """This class defines when/how we record memory during rollout.
    If send-concise-info, memory is only generated at the end of each episode. The generated memory step will include
        the first observation and the final reward/info.
    The typical usage will be in the following order.
        - cache_new_step -> inference -> update action -> get action ->
            execute in environment -> update reward -> make memory step -> append to _AgentSampleFlow.
    """
    send_concise_info: bool
    update_concise_step: bool

    def __post_init__(self):
        self.__cached_step: api.trainer.SampleBatch = None

    def update_cached_step(self, **kwargs):
        """Amend the cached step.
        Args:
            **kwargs: key-value pairs of data to be amended.
        """
        for k, v in kwargs.items():
            self.__cached_step[k] = v

    def truncate_cached_step(self):
        """Sets the cached step to "truncated". If the cached step is a done-step, truncation is omitted.
        """
        if self.__cached_step is not None:
            # for every non-done step, set truncated=True
            self.__cached_step.truncated = 1 - self.__cached_step.done

    def cache_new_step(
        self,
        env_result: env_base.StepResult,
        policy_state: api.policy.PolicyState,
        on_reset: bool,
    ):
        """Cache a new step. Note that the action and reward are yet to be filled. They should be updated through method
             `update_cached_step.`

        Args:
            env_result:
            policy_state:
            on_reset:

        Returns:

        """
        obs = base.namedarray.from_dict(env_result.obs)
        self.__cached_step = api.trainer.SampleBatch(
            obs=obs,
            policy_state=policy_state,
            on_reset=np.full((*env_result.reward.shape[:-1], 1), fill_value=on_reset, dtype=np.uint8),
            done=env_result.done.astype(np.uint8),
            truncated=env_result.truncated.astype(np.uint8),
            info_mask=np.array([0], dtype=np.uint8),  # Will be amended when the next env.step is ready.
            reward=np.zeros_like(env_result.reward),  # Will be amended when the next env.step is ready.
            policy_version_steps=np.array([-1], dtype=np.int64),  # Will be amended during inference.
        )

    def get_action(self):
        """Returns the action in the cached step.
        """
        return self.__cached_step.action if self.__cached_step is not None else None

    def make_memory_step(
        self,
        env_result: env_base.StepResult,
        policy_state: api.policy.PolicyState,
        on_reset: bool,
    ):
        """Makes a new memory step.
        Args:
            env_result:
            policy_state:
            on_reset:

        Returns:
            completed_memory_step: The PREVIOUS memory step.
        """
        is_episode_finished = np.logical_or(env_result.done, env_result.truncated).all()

        complete_memory_step = None

        if on_reset:  # the first step or garbage steps after death
            complete_memory_step = self.__cached_step
            if not is_episode_finished:  # the first step
                self.cache_new_step(env_result, policy_state, on_reset)
            else:  # garbage steps (consecutive dones)
                self.__cached_step = None
        else:  # intermediate steps (including done/truncated step)
            self.update_cached_step(
                reward=env_result.reward,
                info=base.namedarray.from_dict(env_result.info),
                info_mask=np.array([is_episode_finished], dtype=np.uint8),
            )
            if not self.send_concise_info:
                complete_memory_step = self.__cached_step
                self.cache_new_step(env_result, policy_state, on_reset)
            else:
                if self.update_concise_step:
                    self.update_cached_step(
                        obs=base.namedarray.from_dict(env_result.obs),
                        policy_state=policy_state,
                        on_reset=np.full((*env_result.reward.shape[:-1], 1),
                                         fill_value=on_reset,
                                         dtype=np.uint8),
                        done=env_result.done.astype(np.uint8),
                        truncated=env_result.truncated.astype(np.uint8),
                    )
                # inform _AgentSampleFlow to send this trajectory
                self.update_cached_step(done=np.array([is_episode_finished], dtype=np.uint8))

        return complete_memory_step


class Agent:
    """System control of an agent in the environment.
    """

    def __init__(
        self,
        inference_client: inference_stream.InferenceClient,
        sample_producer: sample_stream.SampleProducer,
        deterministic_action: bool,
        sample_steps: int,
        bootstrap_steps: int,
        burn_in_steps: int,
        send_after_done: bool,
        send_full_trajectory: bool,
        trajectory_postprocessor: api.config.TrajPostprocessor,
        pad_trajectory: bool,
        send_concise_info: bool,
        update_concise_step: bool,
        stack_frames: int,
        env_max_num_steps: int,
        index: int,
    ):
        """Initialization method of agent(system-terminology).
        Args:
            inference_client: where to post rollout-request and receive rollout-results.
            sample_producer: where to post collected trajectories.
            deterministic_action: whether action is selected deterministically(argmax),
            as oppose to stochastically(sample).
            sample_steps: length of the sample to be sent. Effective only if send_full_trajectory=False.
            bootstrap_steps: number of additional steps appended to each sent sample. Temporal-difference style value
            tracing benefits from bootstrapping.
            send_after_done: whether to send sample only if the episode is done.
            send_full_trajectory: send full trajectories instead of sample of fixed length. Mostly used when episodes
            are of fixed length, or when agent is running evaluation.
            pad_trajectory: pad the full trajectories to fixed length. Useful when full trajectories are needed but
            length of the environment episodes varies. Effective only when send_full_trajectory=True.
            send_concise_info: If True, each episode is contracted in one time step, with the first observation and the
            last episode-info.
            env_max_num_steps: length of padding if pad_trajectory=True. Note that agents won't reset themselves
            according to this value.
            index: the index of this agent in the environment
        """
        # sample stream
        self.__sample_producer = sample_producer

        self.__sample_flow = _AgentSampleFlow(
            send_after_done=send_after_done,
            send_full_trajectory=send_full_trajectory,
            sample_steps=sample_steps,
            bootstrap_steps=bootstrap_steps,
            burn_in_steps=burn_in_steps,
            pad_trajectory=pad_trajectory,
            env_max_num_steps=env_max_num_steps,
            traj_process_fn=api.trainer.make_traj_postprocessor(trajectory_postprocessor).process,
        )

        # inference stream
        self.__inference_id: Optional[int] = None
        self.__inference_client = inference_client
        # Shared memory: register agent
        self.__inference_index = self.__inference_client.register_agent()
        self.__inference_maker = _AgentInferenceMaker(deterministic_action=deterministic_action)

        # cache and internal states
        self.__state = AgentState.READY_FOR_OBSERVATION
        self.__policy_state = inference_client.default_policy_state
        self.__was_episode_finished = True  # aka on_reset
        self.__step_count = 0
        self.__memory_cache = _AgentMemoryCache(
            send_concise_info=send_concise_info,
            update_concise_step=update_concise_step,
        )
        self.__skip_next_action = False

        self.__stack_frames = stack_frames
        if stack_frames != 0:
            raise RuntimeError("Framestack is not supported on actor workers now.")
        self.__index = index
        self.__logger = logging.getLogger(f"Agent {self.__index}")

    @property
    def state(self):
        """Get the current state of the agent.
        Returns:
            state of the agent.
        """
        return self.__state

    def ready_to_step(self):
        """Whether this agent is ready for step/reset.
        Returns:
            True if the agent is ready for calling agent.observe(), False otherwise.
        """
        if self.__inference_id is not None and self.__state != AgentState.UNCOMSUMED_ACTION_READY:
            if self.__state != AgentState.WAITING_FOR_ACTION:
                raise AgentStateError(self.state, "ready_to_step")
            is_ready = self.__inference_client.is_ready([self.__inference_id], [self.__inference_index])
            if is_ready:
                self.__state = AgentState.UNCOMSUMED_ACTION_READY
            return is_ready
        else:
            return True

    def ready_to_reset(self):
        return self.__was_episode_finished and self.ready_to_step()

    def observe(self, env_result: Optional[env_base.StepResult], truncate: bool = False):
        """Process a new observation at agent level.

        The entering state must be READY_FOR_OBSERVATION.
        The agent will transit to WAITING_FOR_ACTION if a new inference request is posted
        and remain state unchanged otherwise.

        In this function, the agent will possibly
            - issue an inference request
            - update the memory
            - send samples to trainer
        depending on done, truncated, and self.__was_episode_finished. These functionalities are implemented in
        _AgentSampleFlow, _AgentInferenceMaker, and _AgentMemoryCache respectively.

        Args:
            env_result(environment.env_base.StepResult): Step result of this agent.
            truncate: whether to truncate on this observation.
        """
        if self.__state != AgentState.READY_FOR_OBSERVATION:
            raise AgentStateError(self.state, "observe")

        # in benchmark mode, skip this part
        if env_result is None:  # Corner case, skip observing.
            self.__skip_next_action = True
            # In async environments, some agent may not have observation when the episode is truncated.
            # In such cases, the Agent is forced to reset.
            if truncate and not self.__was_episode_finished:
                self.__memory_cache.truncate_cached_step()
            self.__was_episode_finished = self.__was_episode_finished or truncate
            return
        else:
            self.__skip_next_action = False

        # Process system truncate.
        if not env_result.done.all() and not (env_result.done == 0).all():
            raise ValueError("`done` in env_result should be the same across all controlled entities.")
        if not env_result.truncated.all() and not (env_result.truncated == 0).all():
            raise ValueError("`truncated` in env_result should be the same across all controlled entities.")
        env_truncate = env_result.truncated
        if env_truncate is None:
            env_truncate = np.zeros_like(env_result.done)
        if not (env_result.truncated * env_result.done == 0).all():
            raise ValueError("`truncated` and `done` cannot be True simultaneously!"
                             " Check the step result returned by you environment.")
        # Force to truncate allies that are not done.
        env_result.truncated = (1 - env_result.done) * np.logical_or(env_truncate, truncate)

        # Issue an inference request.
        request = self.__inference_maker.make_inference_request(env_result, self.__policy_state,
                                                                self.__step_count,
                                                                self.__was_episode_finished)
        if request is not None:
            self.__inference_id = self.__inference_client.post_request(request, index=self.__inference_index)
            self.__state = AgentState.WAITING_FOR_ACTION

        # Record data in memory.
        # FIXME: matrix game will always return None
        new_memory_step = self.__memory_cache.make_memory_step(env_result,
                                                               policy_state=self.__policy_state,
                                                               on_reset=self.__was_episode_finished)
        if new_memory_step is not None:
            self.__sample_flow.push(new_memory_step)

        # Post sample. We must do this after updating memory.
        sample = self.__sample_flow.get(self.__was_episode_finished)
        if sample is not None:
            self.__sample_producer.post(sample)

        is_episode_finished = np.logical_or(env_result.done, env_result.truncated).all()
        self.__step_count += 1
        if is_episode_finished:
            self.__step_count = 0
        self.__was_episode_finished = is_episode_finished

    def consume_inference_result(self):
        """Consume inference result from the inference_client to the memory_cache.
        """
        if self.__inference_id is not None:  # we indeed had a request
            if self.__state != AgentState.UNCOMSUMED_ACTION_READY:
                raise AgentStateError(self.state, "consume_inference_result")
            [r] = self.__inference_client.consume_result([self.__inference_id], [self.__inference_index])
            self.__memory_cache.update_cached_step(
                **{
                    k: v
                    for k, v in r.items()
                    if k not in ["policy_state", 'client_id', 'request_id', 'received_time']
                })
            self.__policy_state = r.policy_state
            self.__inference_id = None
            self.__state = AgentState.READY_FOR_OBSERVATION
        # otherwise do nothing

    def get_action(self):
        if self.__state != AgentState.READY_FOR_OBSERVATION:
            raise AgentStateError(self.state, "get_action")
        return self.__memory_cache.get_action() if not self.__skip_next_action else None


class _EnvTarget:
    """Represents the current state of a single environment instance.
    """

    def __init__(self, env, max_num_steps, agents: List[Agent], curriculum: Optional[Curriculum]):
        """Initialization method of _EnvTarget.
        Args:
            env: the environment.
            max_num_steps: the maximum number of step that the environment is allowed to run before it is kill by the system.
            agents: the agents(system-level terminology) of the EnvTarget.
        """
        self.__env = env
        self.__max_num_steps = max_num_steps
        self.__agents = agents
        self.__curriculum = curriculum
        self.__curriculum_update_freq_control = FrequencyControl(frequency_seconds=1, initial_value=True)

        self.__step_count = 0
        self.monitor = None
        self.__last_reset_time = None

    def init_monitor(self, monitor: distributed.base.monitoring.Monitor):
        self.monitor = monitor

    def ready_to_reset(self):
        """The EnvTarget is done if all agents are done.
        """
        for ag in self.__agents:
            if not ag.ready_to_reset():
                return False
        return True

    def reset(self):
        """Reset the environment target.
        """
        with ObserveTime(prometheus_metric=self.monitor.metric("marl_actor_agent_action_time_seconds")):
            for ag in self.__agents:
                ag.consume_inference_result()

        if self.__curriculum is not None:
            if self.__curriculum_update_freq_control.check():
                self.__env.set_curriculum_stage(self.__curriculum.get_stage())

        with ObserveTime(prometheus_metric=self.monitor.metric("marl_actor_env_reset_time_seconds")):
            env_results = self.__env.reset()
        for ag, env_result in zip(self.__agents, env_results):
            ag.observe(env_result=env_result, truncate=False)

        self.__step_count = 0

        if self.__last_reset_time is not None:
            logger.info(f"Env target one episode time: {time.perf_counter() - self.__last_reset_time} secs.")
        self.__last_reset_time = time.perf_counter()

    def ready_to_step(self):
        """An EnvTarget is ready to step if all agents are ready.
        """
        for ag in self.__agents:
            if not ag.ready_to_step():
                return False
        return True

    def step(self):
        """Consume rollout requests and perform environment steps.
        NOTE: When environments have part of the agents done. The target stops appending the observations and
            actions of the done-agents to the trajectory_memory. However, the requests of these done-agents are
            sent to the inference client for API compatibility considerations. This feature may cause performance
            issue when the environment is e.g. a battle royal game and your chosen inference client fails to filter
            the request of the dead agents.
        """
        with ObserveTime(prometheus_metric=self.monitor.metric("marl_actor_agent_action_time_seconds")):
            for ag in self.__agents:
                ag.consume_inference_result()
            actions = [ag.get_action() for ag in self.__agents]

        with ObserveTime(prometheus_metric=self.monitor.metric("marl_actor_env_step_time_seconds")):
            env_results = self.__env.step(actions)

        self.__step_count += 1
        truncate = False
        if self.__max_num_steps and self.__step_count >= self.__max_num_steps:
            truncate = True

        with ObserveTime(prometheus_metric=self.monitor.metric("marl_actor_agent_observation_time_seconds")):
            for agent, env_result in zip(self.__agents, env_results):
                agent.observe(env_result=env_result, truncate=truncate)


class _EnvGroup:
    """Represents multiple replicas of the same environment setup.
    """

    def __init__(self, targets: List[_EnvTarget], execution: str = "ring", decorrelate_seconds: float = 0.0):
        """Initialization of environment group.
        Args:
            targets: a list of environments to be run.
            execution: execution method, can be "ring"(execute in order) or "threaded".
        """
        if execution not in ("ring", "threaded"):
            raise NotImplementedError(f"Unknown EnvGroup execution method: {execution}")
        self.__execution_method = execution
        self.__targets = targets
        self.__ring_index = 0
        self.__threads = None
        self.__thread_lock = None
        self.__thread_running = None
        self.__threaded_steps, self.__threaded_resets = None, None
        self.__is_first_reset = True
        self.__decorrelate_seconds = decorrelate_seconds

        if execution == "threaded":
            self.__threads = [
                threading.Thread(target=self._run_threaded, args=(target,), daemon=True) for target in targets
            ]
            self.__thread_lock = threading.Lock()
            self.__thread_running = False
            self.__threaded_steps, self.__threaded_resets = 0, 0

        self.logger = logging.getLogger("EnvGroup")
        self.logger.info(
            f"group of {len(self.__targets)} environments setup in {self.execution_method} mode.")

    @property
    def execution_method(self):
        return self.__execution_method

    def maybe_start_threads(self):
        """Start all environment threads. Effective only in threaded execution.
        """
        if self.__threads is None:
            return
        self.logger.info(f"Starting {len(self.__threads)} env threads.")
        [env_thread.start() for env_thread in self.__threads]
        self.__thread_running = True

    def __ring_head(self) -> _EnvTarget:
        """Returns the current target.
        """
        return self.__targets[self.__ring_index]

    def __ring_rotate(self):
        """Move to the next target.
        """
        self.__ring_index = (self.__ring_index + 1) % len(self.__targets)

    def poll(self) -> Tuple[int, int]:
        """For ring execution, run one step. For threaded execution, collect stats from all threads.
        Returns:
            step_count
            reset_count
        """
        count, batch = 0, 0
        if self.execution_method == "ring":
            target = self.__ring_head()
            if target.ready_to_reset():
                if self.__is_first_reset:
                    self.__is_first_reset = False
                    time.sleep(self.__decorrelate_seconds)
                target.reset()
                batch += 1
            elif target.ready_to_step():
                target.step()
            else:
                return count, batch
            self.__ring_rotate()
            count += 1
        elif self.execution_method == "threaded":
            for t in self.__threads:
                if not t.is_alive():
                    raise RuntimeError("Dead environment thread.")
            count, batch = self.__get_threaded_count()
        else:
            raise NotImplementedError()
        return count, batch

    def __inc_threaded_count(self, step=0, resets=0):
        with self.__thread_lock:
            self.__threaded_steps += step
            self.__threaded_resets += resets

    def __get_threaded_count(self) -> Tuple[int, int]:
        """
        Returns:
            step_count
            reset_count
        """
        with self.__thread_lock:
            s, r = self.__threaded_steps, self.__threaded_resets
            self.__threaded_steps, self.__threaded_resets = 0, 0
            return s, r

    def _run_threaded(self, target):
        while True:
            if target.all_done():
                target.unmask_all_info()
                target.reset()
                self.__inc_threaded_count(resets=1)
            elif target.ready_to_step():
                target.step()
            else:
                time.sleep(0.005)
                continue
            self.__inc_threaded_count(step=1)


class ConfigError(RuntimeError):
    pass


class ActorWorker(worker_base.Worker):
    """Actor Worker holds a ring of environment target and runs the head of the ring at each step.
    """

    def __init__(self, server=None, lock=None):
        super().__init__(server=server, lock=lock)
        self.config = None
        self.__env_group = None
        self.__env_targets = None
        self.__flush_per_step = None
        self.__inference_clients: List[Optional[inference_stream.InferenceClient]] = []
        self.__sample_producers: List[Optional[sample_stream.SampleProducer]] = []
        self.__curriculum = None

    def _configure(self, cfg: api.config.ActorWorker):
        self.config = cfg
        # The inference_clients and sample_producer are passed to _EnvGroups. For now targets share streams.
        self.__inference_clients, self.__sample_producers = self._make_stream_clients(
            cfg.inference_streams, cfg.sample_streams)

        if cfg.curriculum_config is not None:
            self.__curriculum = make_curriculum(cfg.curriculum_config, cfg.worker_info)

        r = self.config.worker_info

        targets = []
        for index in range(cfg.ring_size):
            env = env_base.make(cfg.env[index])
            agents = self.__make_agents(cfg.agent_specs,
                                        agent_count=env.agent_count,
                                        env_max_num_steps=cfg.max_num_steps)
            target = _EnvTarget(env,
                                max_num_steps=cfg.max_num_steps,
                                agents=agents,
                                curriculum=self.__curriculum)
            # When initiating, also kick off the environment by resetting and sending requests.
            targets.append(target)

        # only used in monitor initialization
        self.__env_targets = targets
        self.__env_group = _EnvGroup(targets,
                                     execution=cfg.execution_method,
                                     decorrelate_seconds=self.config.decorrelate_seconds)
        self.__flush_frequency_control = FrequencyControl(
            frequency_steps=max(cfg.ring_size // cfg.inference_splits, 1),
            frequency_seconds=_INFERENCE_FLUSH_FREQUENCY_SECONDS)

        self.avg_poll_response_time = 0
        self.avg_step_time = 0
        self.avg_inf_flush_time = 0
        self.avg_sample_flush_time = 0
        self.poll_step_count = 0

        return r

    def _reconfigure(self, inference_stream_idx=None, inference_stream_kwargs=None):
        if inference_stream_idx is not None:
            if isinstance(inference_stream_idx, int):
                inf = self.__inference_clients[inference_stream_idx]
                if isinstance(inf, inference_stream.InlineInferenceClient):
                    inf.configure_population(**inference_stream_kwargs)
                else:
                    raise NotImplementedError(f"Cannot reconfigure inference stream of type {type(inf)}")
            else:
                for idx, kwargs in zip(inference_stream_idx, inference_stream_kwargs):
                    inf = self.__inference_clients[idx]
                    if isinstance(inf, inference_stream.InlineInferenceClient):
                        inf.configure_population(**kwargs)
                    else:
                        raise NotImplementedError(f"Cannot reconfigure inference stream of type {type(inf)}")

    def start_monitoring(self):
        r = super().start_monitoring()
        metrics = dict(marl_actor_env_step_time_seconds="Summary",
                       marl_actor_env_reset_time_seconds="Summary",
                       marl_actor_agent_action_time_seconds="Summary",
                       marl_actor_agent_observation_time_seconds="Summary",
                       marl_actor_num_steps="Counter",
                       marl_actor_num_episodes="Counter")
        self.monitor.update_metrics(metrics)

        for t in self.__env_targets:
            t.init_monitor(self.monitor)
        for sp in self.__sample_producers:
            sp.init_monitor(self.monitor)

        return r

    def start(self):
        self.__env_group.maybe_start_threads()
        super(ActorWorker, self).start()

    def _poll(self):
        step_count, reset_count = 0, 0

        while step_count < _MAX_POLL_STEPS:

            # Poll results

            # t0 = time.monotonic()
            for inf in self.__inference_clients:
                inf.poll_responses()
            # t1 = time.monotonic()

            # Run environments.
            sc, rc = self.__env_group.poll()

            # t2 = time.monotonic()
            # Flush requests.
            if self.__env_group.execution_method == "threaded" or self.__flush_frequency_control.check(
                    steps=sc):
                for inf in self.__inference_clients:
                    inf.flush()

            # t3 = time.monotonic()

            # Actively reload parameters for inline inference client.
            if rc > 0:
                for inf in self.__inference_clients:
                    if isinstance(inf, inference_stream.InlineInferenceClient):
                        inf.load_parameter()

            # Flush samples.
            for spp in self.__sample_producers:
                spp.flush()

            # t4 = time.monotonic()

            # self.poll_step_count += 1
            # self.avg_poll_response_time = (self.avg_poll_response_time * (self.poll_step_count - 1) + (t1 - t0)) / self.poll_step_count
            # self.avg_step_time = (self.avg_step_time * (self.poll_step_count - 1) + (t2 - t1)) / self.poll_step_count
            # self.avg_inf_flush_time = (self.avg_inf_flush_time * (self.poll_step_count - 1) + (t3 - t2)) / self.poll_step_count
            # self.avg_sample_flush_time = (self.avg_sample_flush_time * (self.poll_step_count - 1) + (t4 - t3)) / self.poll_step_count

            step_count += sc
            reset_count += rc
            if sc == 0:
                break

        # self.logger.info("Actor worker timing: poll_response_time: %.6f, step_time: %.6f, inf_flush_time: %.6f, sample_flush_time: %.6f"\
        #                  % (self.avg_poll_response_time, self.avg_step_time, self.avg_inf_flush_time, self.avg_sample_flush_time))
        self.monitor.metric("marl_actor_num_steps").inc(step_count)
        self.monitor.metric("marl_actor_num_episodes").inc(reset_count)

        return worker_base.PollResult(sample_count=step_count, batch_count=reset_count)

    def _make_stream_clients(self, infs: List[api.config.InferenceStream],
                             spls: List[api.config.SampleStream]):
        """Establish inference client and sample producer.
        Args:
            infs: List of InferenceStream configuration. Client side will be established.
            spls: List of SampleStream configuration. Producer side will be established.

        Returns:
            inference_clients: a list of established InferenceClient[s].
            sample_producers: a list of established SampleProducer[s].
        """
        inference_clients = [(inference_stream.make_client(inf, self.config.worker_info,
                                                           self.multiprocessing_lock)) for inf in infs]
        for s in inference_clients:
            s.default_policy_state = s.get_constant("default_policy_state")
        sample_producers = [(sample_stream.make_producer(spl, self.config.worker_info)) for spl in spls]
        return inference_clients, sample_producers

    def _match_stream(self, spec):
        if isinstance(spec.inference_stream_idx, int):
            inf = self.__inference_clients[spec.inference_stream_idx]
        else:
            raise NotImplementedError("We do not know how to use zipped inference streams yet.")

        if isinstance(spec.sample_stream_idx, int):
            sap = self.__sample_producers[spec.sample_stream_idx]
        else:
            sap = sample_stream.zip_producers(
                [self.__sample_producers[idx] for idx in spec.sample_stream_idx])
        return inf, sap

    def __make_agents(self, agent_specs: List[api.config.AgentSpec], agent_count: int,
                      env_max_num_steps: int):
        """Setup agents.
        Args:
            agent_specs: specifications of agents matching pattern.
            agent_count: total number of agents.
            env_max_num_steps: maximum number of steps of the hosting environment. Useful for trajectory padding.

        Returns:
            list of agents.
        """
        agents = [None for _ in range(agent_count)]
        for spec in agent_specs:
            inf, sap = self._match_stream(spec)
            if not spec.send_full_trajectory and spec.sample_steps <= 0:
                raise ValueError("When send_full_trajectory is False. sample_steps must be positive!")
            if spec.deterministic_action and not spec.send_full_trajectory:
                raise ValueError("Only sending full trajectory is supported in evaluation mode.")
            for j in range(agent_count):
                if agents[j] is None and re.fullmatch(spec.index_regex, str(j)):
                    agents[j] = Agent(inference_client=inf,
                                      sample_producer=sap,
                                      deterministic_action=spec.deterministic_action,
                                      sample_steps=spec.sample_steps,
                                      bootstrap_steps=spec.bootstrap_steps,
                                      burn_in_steps=spec.burn_in_steps,
                                      send_after_done=spec.send_after_done,
                                      send_full_trajectory=spec.send_full_trajectory,
                                      pad_trajectory=spec.pad_trajectory,
                                      trajectory_postprocessor=spec.trajectory_postprocessor,
                                      send_concise_info=spec.send_concise_info,
                                      update_concise_step=spec.update_concise_step,
                                      stack_frames=spec.stack_frames,
                                      env_max_num_steps=env_max_num_steps,
                                      index=spec.inference_stream_idx)
        for j in range(agent_count):
            if agents[j] is None:
                raise ConfigError(f"Agent {j} has no matched specification.")
        return agents
