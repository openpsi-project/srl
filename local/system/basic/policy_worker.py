from typing import List
import logging
import numpy as np
import queue

from base.namedarray import recursive_aggregate
from base.timeutil import FrequencyControl
from local.system import worker_base, inference_stream
import api.config
import api.policy
import base.network
import local.system.parameter_db
import local.system.parameter_db as db


class PolicyWorker(worker_base.Worker):

    def __init__(self, server=None):
        super().__init__(server)

        self.experiment_name = None
        self.policy_name = None
        self.config = None
        self.__load_policy_name = None  # Name of foreign policy might different from domestic policy.
        self.__load_absolute_path = None
        self.__stream = None
        self.__policy = None
        self.__param_db = None
        self.__policy_identifier = None
        self.__pull_frequency_control = None
        self.__pull_fail_count = 0
        self.__max_pull_fails = 60
        self.__requests_buffer = []
        self.__is_paused = False
        self.__parameter_service_client = None

        # Queues in between the data pipeline.
        self.__inference_queue = queue.Queue(1)
        self.__respond_queue = queue.Queue(1)
        self.__param_queue = queue.Queue(1)

        # The mapping threads.
        self.__inference_thread = None
        self.__respond_thread = None
        self.__interrupt = False
        self._threads: List[worker_base.MappingThread] = []

        # Monitoring related below.
        self.__metric_rollout_delay_queueing = None
        self.__metric_rollout_delay_loading = None
        self.__metric_policy_gpu_step_time = None
        self.__metric_rollout_delay_gpu = None
        self.__metric_respond_delay_queueing = None
        self.__metric_respond_delay_cpu = None

        self.__log_frequency_control = FrequencyControl(frequency_seconds=10, initial_value=True)

    def _configure(self, cfg: api.config.PolicyWorker):
        self.logger = logging.getLogger(f"PW{cfg.worker_info.worker_index}")
        self.config = cfg
        self.experiment_name = cfg.worker_info.experiment_name

        self.policy_name = cfg.policy_name
        self.config.worker_info.policy_name = self.policy_name

        self.__policy: api.policy.Policy = api.policy.make(cfg.policy)
        self.__policy.eval_mode()

        # Parameter DB / Policy name related.
        if self.config.foreign_policy is not None:
            p = self.config.foreign_policy
            i = self.config.worker_info
            pseudo_worker_info = api.config.WorkerInformation(experiment_name=p.foreign_experiment_name
                                                              or i.experiment_name,
                                                              trial_name=p.foreign_trial_name or i.trial_name)
            self.__param_db = db.make_db(p.param_db, worker_info=pseudo_worker_info)
            self.__load_absolute_path = p.absolute_path
            self.__load_absolute_path = p.absolute_path
            self.__load_policy_name = p.foreign_policy_name or cfg.policy_name
            self.__policy_identifier = p.foreign_policy_identifier or cfg.policy_identifier
        else:
            self.__param_db = db.make_db(cfg.parameter_db, worker_info=self.config.worker_info)
            self.__load_policy_name = cfg.policy_name
            self.__policy_identifier = cfg.policy_identifier

        if cfg.parameter_service_client is not None and self.__load_absolute_path is None:
            self.__parameter_service_client = local.system.parameter_db.make_client(
                cfg.parameter_service_client, self.config.worker_info)
            self.__parameter_service_client.subscribe(experiment_name=self.__param_db.experiment_name,
                                                      trial_name=self.__param_db.trial_name,
                                                      policy_name=self.__load_policy_name,
                                                      tag=self.__policy_identifier,
                                                      callback_fn=self._put_ckpt)

        if cfg.policy.init_ckpt_dir is not None:
            raise DeprecationWarning("Use foreign policy instead.")
        self.__stream = inference_stream.make_server(cfg.inference_stream,
                                                     worker_info=self.config.worker_info)
        self.__stream.set_constant("default_policy_state", self.__policy.default_policy_state)
        self.__pull_frequency_control = FrequencyControl(
            frequency_seconds=cfg.pull_frequency_seconds,
            # If policy has a specified initial state, do not pull the
            # saved version immediately.
            initial_value=(cfg.policy.init_ckpt_dir is None))
        self.__max_pull_fails = cfg.pull_max_failures
        self.__bs = self.config.batch_size
        self.__pull_fail_count = 0

        self._threads.append(
            worker_base.MappingThread(self._inference,
                                      self.__interrupt,
                                      self.__inference_queue,
                                      self.__respond_queue,
                                      cuda_device=self.__policy.device))
        self._threads.append(
            worker_base.MappingThread(self._respond,
                                      self.__interrupt,
                                      self.__respond_queue,
                                      downstream_queue=None))
        [t.start() for t in self._threads]

        # Monitoring related below.
        kwargs = dict(host=base.network.gethostname(),
                      experiment=self.config.worker_info.experiment_name,
                      trial=self.config.worker_info.trial_name,
                      worker=self.config.worker_info.worker_type,
                      worker_id=self.config.worker_info.worker_index,
                      policy=self.config.worker_info.policy_name)
        return self.config.worker_info

    def _put_ckpt(self, checkpoint):
        while True:
            try:
                _ = self.__param_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self.__param_queue.put_nowait(checkpoint)
            self.logger.debug("Reset frequency control.")
            self.__pull_frequency_control.reset_time()
        except queue.Full:
            pass

    def _inference(self, agg_requests: api.policy.RolloutRequest):
        """Run inference for batched aggregated_request.
        """
        try:
            checkpoint = self.__param_queue.get_nowait()
            self.__policy.load_checkpoint(checkpoint)
            self.logger.debug(f"Loaded checkpoint version: {self.__policy.version}")
        except queue.Empty:
            pass
        responses = self.__policy.rollout(agg_requests)
        responses.client_id = agg_requests.client_id
        responses.request_id = agg_requests.request_id
        responses.received_time = agg_requests.received_time
        responses.policy_name = np.full(shape=agg_requests.client_id.shape, fill_value=self.policy_name)
        responses.policy_version_steps = np.full(shape=agg_requests.client_id.shape,
                                                 fill_value=self.__policy.version)
        return responses

    def _respond(self, responses: api.policy.RolloutResult):
        """Send rollout results.
        """
        self.__stream.respond(responses)

    def _stats(self):
        """TODO: add stats to wandb.
        """
        return {}

    def _poll(self):
        for t in self._threads:
            if not t.is_alive():
                raise RuntimeError("Exception in policy thread.")
        if self.__parameter_service_client is not None:
            if not self.__parameter_service_client.is_alive():
                raise RuntimeError("Exception in subscription thread.")

        # Pull parameters from server
        if self.__pull_frequency_control.check():
            self.logger.debug("Active pull.")
            while not self.__param_queue.empty():
                self.__param_queue.get()
            is_first_pull = self.__policy.version < 0
            self.__param_queue.put(self.__get_checkpoint_from_db(block=is_first_pull))

        samples = 0
        batches = 0
        # buffer requests, and record when the oldest requests is received.
        request_batch = self.__stream.poll_requests()
        self.__requests_buffer.extend(request_batch)

        if len(self.__requests_buffer) > 0:
            try:
                # If the inference has not started on the queued batch, make the batch larger instead of
                # initiating another batch.
                queued_requests = [self.__inference_queue.get_nowait()]
                samples -= queued_requests[0].length(dim=0)
                batches -= 1
            except queue.Empty:
                queued_requests = []
            agg_request = recursive_aggregate(queued_requests + self.__requests_buffer,
                                              lambda x: np.concatenate(x, axis=0))
            bs = min(self.__bs, agg_request.length(dim=0))
            self.__inference_queue.put_nowait(agg_request[:bs])
            samples += bs
            batches += 1
            if bs == agg_request.length(dim=0):
                self.__requests_buffer = []
            else:
                self.__requests_buffer = [agg_request[bs:]]

        if self.__log_frequency_control.check():
            self.logger.debug(f"Policy version: {self.__policy.version}")

        return worker_base.PollResult(sample_count=samples, batch_count=batches)

    def get_checkpoint(self):
        raise NotImplementedError("Getting State Dict from policy worker not supported.")

    def load_checkpoint(self, state_dict):
        """Update parameter of the policy worker
        """
        self.__policy.load_checkpoint(state_dict)

    def __get_checkpoint_from_db(self, block=False):
        if self.__load_absolute_path is not None:
            return self.__param_db.get_file(self.__load_absolute_path)
        else:
            return self.__param_db.get(name=self.__load_policy_name,
                                       identifier=self.__policy_identifier,
                                       block=block)

    def __stop_threads(self):
        self.__interrupt = True
        self.logger.debug(f"Stopping {len(self._threads)} local threads.")
        for t in self._threads:
            t.stop()
        if self.__parameter_service_client is not None:
            self.logger.debug("Stopping parameter subscription.")
            self.__parameter_service_client.stop_listening()

    def pause(self):
        super(PolicyWorker, self).pause()
        self.__is_paused = True

    def start(self):
        if self.__is_paused:
            # Set block=True to wait for trainers to push the first model when starting a new PSRO iteration.
            while not self.__param_queue.empty():
                self.__param_queue.get()
            self.__param_queue.put(self.__param_db.get(self.__get_checkpoint_from_db(block=True)))
            self.__is_paused = False

        if self.__parameter_service_client is not None:
            self.__parameter_service_client.start_listening()
        super(PolicyWorker, self).start()

    def _reconfigure(self, policy_name=None):
        if self.__parameter_service_client is not None:
            raise NotImplementedError("Please somebody implement parameter service for reconfiguring.")
        if policy_name is not None:
            self.config.policy_name = policy_name
            self.config.worker_info.policy_name = policy_name
            self.policy_name = policy_name

    def exit(self):
        self.__stop_threads()
        super(PolicyWorker, self).exit()

    def interrupt(self):
        self.__stop_threads()
        super(PolicyWorker, self).interrupt()
