import logging
import queue
import time

from base.timeutil import FrequencyControl
import api.environment as env_base
import api.config as config_pkg
import distributed.system.parameter_db
import distributed.system.worker_base as worker_base
import api.policy
import api.trainer
import base.buffer
import base.gpu_utils
import base.network
import distributed.system.sample_stream


class BufferWorker(worker_base.Worker):
    """Buffer worker preprocess the sample for the trainers.
    If data-augmenter is not None, buffer worker will run augmentation on the sample.
    If policy is not None, buffer worker will run re-analysis on the sample.
    """

    def __init__(self, server=None, lock=None):
        super(BufferWorker, self).__init__(server, lock)
        self.__from_sample_stream = None
        self.__to_sample_stream = None
        self.__post_rate = None
        self.__policy = None
        self.__policy_identifier = None
        self.__reanalyze_target = None
        self.__data_augmenter = None
        self.__unpack_batch = None
        self.__buffer = None
        self.__parameter_service_client = None
        self.__parameter_queue = queue.Queue(1)
        self.__raw_data_queue = queue.Queue(4)
        self.__augmented_queue = queue.Queue(4)
        self.__reanalyzed_queue = queue.Queue(4)
        self.__post_queue_handle = None
        self.__interrupt = False
        self.__log_frequency_seconds = FrequencyControl(frequency_seconds=10, initial_value=True)

    def _configure(self, cfg: config_pkg.BufferWorker) -> config_pkg.WorkerInformation:
        self.logger = logging.getLogger(f"BW{cfg.worker_info.worker_index}")
        self.config = cfg
        worker_info = cfg.worker_info
        self.__buffer = base.buffer.make_buffer(cfg.buffer_name, **cfg.buffer_args)
        self.__from_sample_stream = distributed.system.sample_stream.make_consumer(
            cfg.from_sample_stream, cfg.worker_info)
        self.__to_sample_stream = distributed.system.sample_stream.make_producer(
            cfg.to_sample_stream, cfg.worker_info)
        self.__unpack_batch = cfg.unpack_batch_before_post

        self.__threads = []
        queue1, queue2 = self.__raw_data_queue, self.__augmented_queue
        if cfg.data_augmenter is not None:
            self.__data_augmenter = env_base.make_augmenter(cfg.data_augmenter)
            self.__threads.append(
                worker_base.MappingThread(map_fn=self._augment,
                                          interrupt_flag=self.__interrupt,
                                          upstream_queue=queue1,
                                          downstream_queue=queue2))
            queue1, queue2 = self.__augmented_queue, self.__reanalyzed_queue
        else:
            queue1, queue2 = self.__raw_data_queue, self.__reanalyzed_queue

        if cfg.policy is not None:
            assert all([
                cfg.reanalyze_target is not None, cfg.policy_identifier is not None, cfg.policy_name
                is not None, cfg.parameter_db is not None
            ])
            self.__reanalyze_target = cfg.reanalyze_target
            self.__policy_name = cfg.policy_name
            self.__policy_identifier = cfg.policy_identifier
            self.__parameter_db = distributed.system.parameter_db.make_db(cfg.parameter_db, worker_info)
            self.__policy = api.policy.make(cfg.policy)
            self.__policy.eval_mode()
            self.__pull_freq_control = FrequencyControl(frequency_seconds=cfg.pull_frequency_seconds,
                                                        initial_value=True)
            self.__threads.append(
                worker_base.MappingThread(map_fn=self._reanalyze,
                                          interrupt_flag=self.__interrupt,
                                          upstream_queue=queue1,
                                          downstream_queue=queue2,
                                          cuda_device=self.__policy.device))
            self.__post_queue_handle = queue2
        else:
            self.__post_queue_handle = queue1

        if cfg.parameter_service_client is not None:
            self.__parameter_service_client = distributed.system.parameter_db.make_client(
                cfg.parameter_service_client, self.config.worker_info)
            self.__parameter_service_client.subscribe(experiment_name=self.__parameter_db.experiment_name,
                                                      trial_name=self.__parameter_db.trial_name,
                                                      policy_name=self.__policy_name,
                                                      tag=self.__policy_identifier,
                                                      callback_fn=self._put_ckpt)

        for t in self.__threads:
            t.start()
        return worker_info

    def start_monitoring(self):
        r = super().start_monitoring()
        metrics = dict(reanalyze_policy_version="Gauge",
                       data_augmentation_latency_seconds="Summary",
                       reanalyze_latency_seconds="Summary")
        self.monitor.update_metrics(metrics)
        return r

    def _put_ckpt(self, checkpoint):
        while True:
            try:
                _ = self.__parameter_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self.__parameter_queue.put_nowait(checkpoint)
            # self.logger.debug("Reset frequency control.")
            # self.__pull_freq_control.reset_time()
        except queue.Full:
            pass

    def _poll(self) -> worker_base.PollResult:
        for t in self.__threads:
            if not t.is_alive():
                raise RuntimeError("Exception in buffer worker gpu thread.")
        if self.__parameter_service_client is not None:
            if not self.__parameter_service_client.is_alive():
                raise RuntimeError("Exception in buffer worker subscription thread")

        batch_count = 0
        sample_count = 0
        self.__from_sample_stream.consume_to(buffer=self.__buffer, max_iter=64)
        if not self.__buffer.empty():
            try:
                self.__raw_data_queue.put_nowait(self.__buffer.get().sample)
            except queue.Full:
                pass

        if self.__policy is not None and self.__pull_freq_control.check():
            self.logger.debug("Active pull.")
            is_first_pull = self.__policy.version < 0
            while not self.__parameter_queue.empty():
                self.__parameter_queue.get()
            ckpt = self.__parameter_db.get(self.__policy_name,
                                           identifier=self.__policy_identifier,
                                           block=is_first_pull)
            self.__parameter_queue.put(ckpt)

        for _ in range(64):
            try:
                sample = self.__post_queue_handle.get_nowait()
                sample_count = len(sample)
                if self.__unpack_batch:
                    for i in range(0, sample.reward.shape[1]):
                        # The first axis is time, which is controlled by actor worker and buffer.
                        self.__to_sample_stream.post(sample[:, i])
                else:
                    self.__to_sample_stream.post(sample)
                batch_count += 1
            except queue.Empty:
                break
        self.__to_sample_stream.flush()
        if self.__log_frequency_seconds.check():
            self.logger.debug(f"Policy version: {self.__policy.version}")

        return worker_base.PollResult(sample_count=sample_count, batch_count=batch_count)

    def _reconfigure(self, config, src_policy=None) -> config_pkg.WorkerInformation:
        pass

    def _reanalyze(self, sample: api.trainer.SampleBatch):
        """Reanalyze a sample.
        """
        start = time.monotonic_ns()
        checkpoint = None
        while True:
            try:
                checkpoint = self.__parameter_queue.get_nowait()
                self.logger.debug("Clear parameter queue.")
            except queue.Empty:
                break

        if checkpoint is not None:
            self.logger.debug("Loaded checkpoint.")
            self.__policy.load_checkpoint(checkpoint)
            self.monitor.metric("reanalyze_policy_version").set(self.__policy.version)

        sample = self.__policy.reanalyze(sample, self.__reanalyze_target)
        latency = (time.monotonic_ns() - start) / 1e9
        self.logger.info(f"Reanalyze latency: {latency:.3f} seconds.")
        self.monitor.metric("reanalyze_latency_seconds").observe(latency)
        return sample

    def _augment(self, sample: api.trainer.SampleBatch):
        """Run data augmentation on a sample.
        """
        start = time.monotonic_ns()
        sample = self.__data_augmenter.process(sample)
        latency = (time.monotonic_ns() - start) / 1e9
        self.monitor.metric("data_augmentation_latency_seconds").observe(latency)
        return sample

    def interrupt(self):
        if self.__parameter_service_client is not None:
            self.__parameter_service_client.stop_listening()
        super(BufferWorker, self).interrupt()

    def start(self):
        if self.__parameter_service_client is not None:
            self.__parameter_service_client.start_listening()
        super(BufferWorker, self).start()
