from typing import List, Optional
import copy
import numpy as np
import prometheus_client
import re
import warnings

from base.namedarray import recursive_aggregate
from base.timeutil import PrometheusSummaryObserve as ObserveTime
from local.system import worker_base, inference_stream, sample_stream
import api.config
import api.policy
import api.trainer
import base.network
import api.environment as env_base

_MAX_POLL_STEPS = 16
_MAX_UNEXPECTED_ENVSTEP_FAILURES = 1

# Prometheus Summary (EnvTarget)
METRIC_ACTOR_ENV_STEP_TIME_SECONDS = prometheus_client.Summary(
    "marl_actor_env_step_time_seconds", "", ["host", "experiment", "trial", "worker", "worker_id"])
METRIC_ACTOR_ENV_RESET_TIME_SECONDS = prometheus_client.Summary(
    "marl_actor_env_reset_time_seconds", "", ["host", "experiment", "trial", "worker", "worker_id"])
METRIC_ACTOR_AGENT_ACTION_TIME_SECONDS = prometheus_client.Summary(
    "marl_actor_agent_action_time_seconds", "", ["host", "experiment", "trial", "worker", "worker_id"])
METRIC_ACTOR_AGENT_OBSERVATION_TIME_SECONDS = prometheus_client.Summary(
    "marl_actor_agent_observation_time_seconds", "", ["host", "experiment", "trial", "worker", "worker_id"])

# Prometheus Counter (Actor Worker)
METRIC_ACTOR_NUM_STEPS = prometheus_client.Counter("marl_actor_num_steps", "",
                                                   ["host", "experiment", "trial", "worker", "worker_id"])
METRIC_ACTOR_NUM_EPISODES = prometheus_client.Counter("marl_actor_num_episodes", "",
                                                      ["host", "experiment", "trial", "worker", "worker_id"])


class Agent:
    """System control of an agent in the environment.
    """

    def __init__(self, inference_client: inference_stream.InferenceClient,
                 sample_producer: sample_stream.SampleProducer, deterministic_action: bool, sample_steps: int,
                 bootstrap_steps: int, send_after_done: bool, send_full_trajectory: bool,
                 pad_trajectory: bool, send_concise_info: bool, stack_frames: int, env_max_num_steps: int,
                 index: int):
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
            index: the index of this agent in the environment.
        """
        self.__inference_client = inference_client
        self.__sample_producer = sample_producer
        self.__deterministic_action = deterministic_action
        self.__sample_steps = sample_steps
        self.__bootstrap_steps = bootstrap_steps
        self.__send_after_done = send_after_done
        self.__send_full_trajectory = send_full_trajectory
        self.__pad_trajectory = pad_trajectory
        self.__send_concise_info = send_concise_info
        self.__stack_frames = stack_frames
        self.__env_max_num_steps = env_max_num_steps
        self.__index = index

        self.__done = True
        self.__step_count = 0
        self.__stacked_recent_frames = None
        self.__inference_id: Optional[int] = None
        self.__memory: List[api.trainer.SampleBatch] = []
        self.__policy_state = inference_client.default_policy_state

    def __maybe_stack_observation(self, obs):
        if self.__stack_frames == 0:
            return obs
        elif self.__stack_frames == 1:
            return obs[np.newaxis, :]
        elif self.__stacked_recent_frames is None:
            self.__stacked_recent_frames = recursive_aggregate([obs] * (self.__stack_frames - 1), np.stack)
            return self.__maybe_stack_observation(obs)
        else:
            frames = recursive_aggregate([self.__stacked_recent_frames, obs[np.newaxis, :]],
                                         lambda x: np.concatenate(x, axis=0))
            self.__stacked_recent_frames = frames[1:]
            return frames

    @property
    def done(self):
        """Whether this agent is done for the current episode. True on environment initialization.
        """
        return self.__done

    def terminate(self):
        """Force the agent to start a new episode.
        """
        self.__done = True

    def is_ready(self):
        """Whether this agent is ready for the next step.
        Inference_id is None if the agent is not waiting for action to step.
        """
        return self.__inference_id is None or self.__inference_client.is_ready([self.__inference_id])

    def unmask_info(self):
        """Unmask the most recent step of the agent.
        """
        if self.__memory:
            self.__memory[-1].info_mask[:] = 1

    def __maybe_post_batches(self):
        if self.__send_full_trajectory or self.__send_after_done:
            return
        self.__post_batches()

    def __post_batches(self):
        while len(self.__memory) >= self.__sample_steps + self.__bootstrap_steps:
            self.__sample_producer.post(
                recursive_aggregate(self.__memory[:self.__sample_steps + self.__bootstrap_steps], np.stack))
            self.__memory = self.__memory[self.__sample_steps:]

    def __maybe_post_all(self):
        if not self.__send_full_trajectory:
            if self.__send_after_done:
                self.__post_batches()
            return
        if len(self.__memory) > 0:
            if self.__pad_trajectory:
                last_step = copy.deepcopy(self.__memory[-1])
                last_step.on_reset[:] = 1
                last_step.info_mask[:] = 0
                self.__memory += [last_step
                                  ] * (self.__env_max_num_steps + self.__bootstrap_steps - len(self.__memory))
            self.__sample_producer.post(recursive_aggregate(self.__memory, np.stack))
            self.__memory = []

    def __update_policy_state(self, policy_state):
        self.__policy_state = policy_state

    def observe(self, env_result: Optional[env_base.StepResult]):
        """Process a new observation at agent level.
        Args:
            env_result(environment.env_base.StepResult): Step result of this agent.
        """
        if self.__done:
            # This step is `on_reset`.
            self.__maybe_post_all()
            self.__maybe_post_batches()
            self.__update_policy_state(self.__inference_client.default_policy_state)
        else:
            self.__maybe_post_batches()

        if env_result is None:
            return
        if env_result.done.all() and self.__done:
            return

        obs = base.namedarray.from_dict(env_result.obs)
        info = base.namedarray.from_dict(env_result.info)
        self.__amend_prev_step(reward=env_result.reward)
        if not self.__send_concise_info or self.__done:
            # Append a new step if we need dense sample or the first step of each episode.
            self.__memory.append(
                api.trainer.SampleBatch(
                    obs=obs,
                    policy_state=self.__policy_state,
                    on_reset=np.full((*env_result.reward.shape[:-1], 1),
                                     fill_value=self.__done,
                                     dtype=np.uint8),
                    reward=np.zeros_like(env_result.reward),
                    policy_version_steps=np.array([-1], dtype=np.int64),  # Will be amended.
                ))
        is_done = env_result.done.all()
        self.__amend_prev_step(info=info, info_mask=np.array([is_done], dtype=np.uint8))

        # Send request.
        if not is_done:
            request = api.policy.RolloutRequest(
                obs=self.__maybe_stack_observation(obs=obs),
                policy_state=self.__policy_state,
                is_evaluation=np.full((*env_result.reward.shape[:-1], 1),
                                      fill_value=self.__deterministic_action,
                                      dtype=np.uint8),
                # Cause we use recursive aggregate, we need an identifier for default policy state.
                on_reset=np.full((*env_result.reward.shape[:-1], 1), fill_value=self.__done, dtype=np.uint8),
                step_count=np.full((*env_result.reward.shape[:-1], 1),
                                   fill_value=self.__step_count,
                                   dtype=np.int32))
            self.__inference_id = self.__inference_client.post_request(request, flush=False)
            self.__step_count += 1
        else:
            self.__step_count = 0
            self.__stacked_recent_frames = None

        self.__done = is_done

    def get_action(self):
        """Get rollout result from inference client.
        """
        if self.__inference_id is not None:
            [r] = self.__inference_client.consume_result([self.__inference_id])
            self.__amend_prev_step(**{k: v for k, v in r.items() if k != "policy_state"})
            self.__update_policy_state(r.policy_state)
            self.__inference_id = None
            return r.action
        else:
            return None

    def __amend_prev_step(self, **kwargs):
        if self.__memory:
            for k, v in kwargs.items():
                self.__memory[-1][k] = v


class _EnvTarget:
    """Represents the current state of a single environment instance.
    """

    def __init__(self, env, max_num_steps, agents: List[Agent], monitoring_kwargs):
        """Initialization method of _EnvTarget.
        Args:
            env: the environment.
            max_num_steps: the maximum number of step that the environment is allowed to run before it is kill by the system.
            agents: the agents(system-level terminology) of the EnvTarget.
        """
        self.__env = env
        self.__max_num_steps = max_num_steps
        self.__agents = agents

        self.__step_count = 0
        self.__step_failure_cnt = 0

        # Monitoring
        self.__metric_action_time = METRIC_ACTOR_AGENT_ACTION_TIME_SECONDS.labels(**monitoring_kwargs)
        self.__metric_observation_time = METRIC_ACTOR_AGENT_OBSERVATION_TIME_SECONDS.labels(
            **monitoring_kwargs)
        self.__metric_reset_time = METRIC_ACTOR_ENV_RESET_TIME_SECONDS.labels(**monitoring_kwargs)
        self.__metric_step_time = METRIC_ACTOR_ENV_STEP_TIME_SECONDS.labels(**monitoring_kwargs)

    def all_done(self):
        """The EnvTarget is done if all agents are done.
        """
        return all([ag.done for ag in self.__agents])

    def unmask_all_info(self):
        """Unmask the most recent episode info of all agents.
        """
        for ag in self.__agents:
            ag.unmask_info()

    def reset(self):
        """Reset the environment target.
        """
        with ObserveTime(prometheus_metric=self.__metric_reset_time):
            env_results = self.__env.reset()
        for ag, env_result in zip(self.__agents, env_results):
            ag.observe(env_result)

        self.__step_count = 0

    def step(self):
        """Consume rollout requests and perform environment steps.
        NOTE: When environments have part of the agents done. The target stops appending the observations and
            actions of the done-agents to the trajectory_memory. However, the requests of these done-agents are
            sent to the inference client for API compatibility considerations. This feature may cause performance
            issue when the environment is e.g. a battle royal game and your chosen inference client fails to filter
            the request of the dead agents.
        """
        with ObserveTime(prometheus_metric=self.__metric_action_time):
            actions = [ag.get_action() for ag in self.__agents]
        try:
            with ObserveTime(prometheus_metric=self.__metric_step_time):
                env_results = self.__env.step(actions)
        except Exception as e:
            # TODO:  figure out the hide-and-seek quadrant bug
            # in case of unexpected bugs (like nan in MuJoCo)
            self.__step_failure_cnt += 1
            warnings.warn(
                f"Exception {e} occured for {self.__step_failure_cnt} times during env.step! The trajectory memory is abandoned."
            )
            self.reset()
            if self.__step_failure_cnt >= _MAX_UNEXPECTED_ENVSTEP_FAILURES:
                raise ValueError(
                    f"env.step has failed unexpectedly for a maximum number of {_MAX_UNEXPECTED_ENVSTEP_FAILURES} times"
                )
            return

        with ObserveTime(prometheus_metric=self.__metric_observation_time):
            for agent, env_result in zip(self.__agents, env_results):
                agent.observe(env_result=env_result)

        self.__step_count += 1
        if self.__max_num_steps and self.__step_count >= self.__max_num_steps:
            self.__force_terminate()

    def ready_to_step(self):
        """An EnvTarget is ready to step if all agents are ready.
        """
        for ag in self.__agents:
            if not ag.is_ready():
                return False
        return True

    def __force_terminate(self):
        for ag in self.__agents:
            ag.terminate()


class _EnvRing:
    """Represents multiple replicas of the same environment setup.
    """

    def __init__(self, targets: List[_EnvTarget]):
        self.targets = targets
        self.index = 0

    @property
    def head(self) -> _EnvTarget:
        """Returns the current target.
        """
        return self.targets[self.index]

    def rotate(self):
        """Move to the next target.
        """
        self.index = (self.index + 1) % len(self.targets)


class ConfigError(RuntimeError):
    pass


class ActorWorker(worker_base.Worker):
    """Actor Worker holds a ring of environment target and runs the head of the ring at each step.
    """

    def __init__(self, server=None):
        super().__init__(server=server)
        self.config = None
        self.__ring: _EnvRing = _EnvRing([])
        self.__steps = 0
        self.__flush_per_step = None
        self.__inference_clients: List[Optional[inference_stream.InferenceClient]] = []
        self.__sample_producers: List[Optional[sample_stream.SampleProducer]] = []

    def _configure(self, cfg: api.config.ActorWorker):
        self.config = cfg
        # The inference_clients and sample_producer are passed to _EnvTargets. For now targets share streams.
        self.__inference_clients, self.__sample_producers = self._make_stream_clients(
            cfg.inference_streams, cfg.sample_streams)

        r = self.config.worker_info
        monitoring_kwargs = dict(host=base.network.gethostname(),
                                 experiment=r.experiment_name,
                                 trial=r.trial_name,
                                 worker=r.worker_type,
                                 worker_id=r.worker_index)
        self.__metric_actor_num_steps = METRIC_ACTOR_NUM_STEPS.labels(**monitoring_kwargs)
        self.__metric_actor_num_episodes = METRIC_ACTOR_NUM_EPISODES.labels(**monitoring_kwargs)

        targets = []
        for _ in range(cfg.ring_size):
            env = env_base.make(cfg.env)
            agents = self.__make_agents(cfg.agent_specs,
                                        agent_count=env.agent_count,
                                        env_max_num_steps=cfg.max_num_steps)
            target = _EnvTarget(env,
                                max_num_steps=cfg.max_num_steps,
                                agents=agents,
                                monitoring_kwargs=monitoring_kwargs)
            # When initiating, also kick off the environment by resetting and sending requests.
            targets.append(target)

        self.__ring = _EnvRing(targets)
        self.__flush_frequency_steps = max(cfg.ring_size // cfg.inference_splits, 1)

        return r

    def _poll(self):
        count = 0
        batch_count = 0
        while count < _MAX_POLL_STEPS:
            target = self.__ring.head
            if target.all_done():
                target.unmask_all_info()
                for inf in self.__inference_clients:
                    if inf.type == api.config.InferenceStream.Type.INLINE:
                        inf.load_parameter()
                target.reset()
                self.__metric_actor_num_episodes.inc(1)
                batch_count += 1
            elif target.ready_to_step():
                target.step()
                self.__metric_actor_num_steps.inc(1)
            else:
                break
            self.__steps += 1
            if self.__steps % self.__flush_frequency_steps == 0:
                for inf in self.__inference_clients:
                    inf.flush()
            self.__ring.rotate()
            count += 1
        return worker_base.PollResult(sample_count=count, batch_count=batch_count)

    def _make_stream_clients(self, infs: List[api.config.InferenceStream],
                             spls: List[api.config.SampleStream]):
        inference_clients = [(inference_stream.make_client(inf, self.config.worker_info)) for inf in infs]
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
                                      send_after_done=spec.send_after_done,
                                      send_full_trajectory=spec.send_full_trajectory,
                                      pad_trajectory=spec.pad_trajectory,
                                      send_concise_info=spec.send_concise_info,
                                      stack_frames=spec.stack_frames,
                                      env_max_num_steps=env_max_num_steps,
                                      index=spec.inference_stream_idx)
        for j in range(agent_count):
            if agents[j] is None:
                raise ConfigError(f"Agent {j} has no matched specification.")
        return agents
