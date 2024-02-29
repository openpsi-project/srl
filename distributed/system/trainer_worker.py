from typing import Optional, Tuple, Any
import datetime
import logging
import numpy as np
import queue
import time
import threading
import torch
import statistics

from legacy import algorithm, environment, population_based_training, experiments
from base.gpu_utils import set_cuda_device
from base.network import find_free_port
import api.config as config
import api.trainer
import base.buffer
import base.shared_memory
import base.timeutil
import base.names as names
import distributed.system.sample_stream
import distributed.system.worker_base as worker_base
import distributed.system.parameter_db as parameter_db
import distributed.base.name_resolve as name_resolve
import distributed.base.monitoring

DDP_MAX_TRAINER_STEP_DIFFERENCE = 3
PARAMETER_DB_GC_FREQUENCY_SECONDS = 300


class GPUThread:

    def __init__(self, buffer, trainer, is_master, worker_info, log_frequency_seconds, log_frequency_steps,
                 push_frequency_seconds, push_frequency_steps, preemption_steps, dist_kwargs,
                 train_for_seconds):
        self.logger = logging.getLogger("gpu-thread")
        self.timing = base.timeutil.Timing()
        self.dist_kwargs = dist_kwargs
        self.__logging_queue = queue.Queue(100)
        self.__checkpoint_push_queue = queue.Queue(8)
        self.__replay_entry = None

        self.__buffer = buffer
        self.__is_master = is_master
        self.__trainer: api.trainer.Trainer = trainer

        self.__interrupting = False
        self.__interrupt_at_step = 1e10
        self.__steps = 0
        self.__thread = threading.Thread(target=self._run, daemon=True)

        self.__preemption_steps = preemption_steps

        # Monitoring related below.
        self.__start_time_ns = time.monotonic_ns()
        self.monitor = None

        self.__logging_control = base.timeutil.FrequencyControl(frequency_seconds=log_frequency_seconds,
                                                                frequency_steps=log_frequency_steps)
        self.__push_control = base.timeutil.FrequencyControl(frequency_seconds=push_frequency_seconds,
                                                             frequency_steps=push_frequency_steps)

        self.__last_buffer_get_time = None
        self.__train_for_seconds = train_for_seconds
        self.__after_first_step = False
        self.__train_start: int = None

    def init_monitor(self, monitor: distributed.base.monitoring.Monitor):
        self.monitor = monitor

    @property
    def distributed_steps(self):
        return self.__steps

    def stats(self):
        duration = (time.monotonic_ns() - self.__start_time_ns) / 1e9
        return {
            "gpu_busy_time": self.monitor.metric("marl_trainer_gpu_step_time")._sum.get() / duration,
            "seconds_per_step": (self.monitor.metric("marl_trainer_gpu_step_time")._sum.get() /
                                 self.monitor.metric("marl_trainer_gpu_step_time")._count.get()),
            "staleness": (self.monitor.metric("marl_trainer_sample_staleness")._sum.get() /
                          self.monitor.metric("marl_trainer_sample_staleness")._count.get()),
            "reuses": (self.monitor.metric("marl_trainer_sample_reuses")._sum.get() /
                       self.monitor.metric("marl_trainer_sample_reuses")._count.get()),
        }

    def is_alive(self):
        return self.__thread.is_alive()

    def start(self):
        self.__thread.start()

    def _run(self):
        # set_cuda_device(self.__trainer.policy.device)
        self.__trainer.distributed(**self.dist_kwargs)
        cnt = 0
        while True:
            if self.__interrupting:
                self.__interrupt_loop()
                break
            self._run_step()
            cnt += 1
            if cnt % 100 == 0 and not self.__buffer.empty():
                total_time = sum(v for k, v in self.timing.items() if k.count('/') == 1)
                msg = "\n==========================================\n"
                for k, v in self.timing.items():
                    msg += "{} proportion: {:.3f}\n".format(k, v / total_time)
                msg += "==========================================\n"
                msg += "Total time: {:.3f} secs, total step: {}".format(total_time, cnt)
                self.logger.info(msg)

    def __interrupt_loop(self):
        self.logger.info("Entering stopping loop.")
        while self.__steps < self.__interrupt_at_step:
            if self.__replay_entry is None:
                break
            self._run_step_on_entry(self.__replay_entry)
        self.logger.info(f"Stopping at {self.__steps}!")

    def _run_step(self):
        if not self.__buffer.empty():
            with self.timing.add_time("gpu_thread/buffer_get"):
                self.__replay_entry = self.__buffer.get()
            with self.timing.add_time("gpu_thread/run_step"):
                tik = time.perf_counter()
                # self.logger.info(self.__replay_entry.sample.obs.obs.shape)
                self._run_step_on_entry(self.__replay_entry)
                # self.logger.info(f"One train step time {time.perf_counter() - tik:.3f}")
        else:
            with self.timing.add_time("gpu_thread/idle"):
                time.sleep(
                    0.005
                )  # to avoid locking the buffer. We should remove this line when our buffer is thread-safe.

    def _run_step_on_entry(self, replay_entry):
        if not self.__after_first_step:
            self.__after_first_step = True
            self.__train_start = time.time()
        else:
            if time.time() - self.__train_start > self.__train_for_seconds:
                raise RuntimeError(f"Training time exceeds {self.__train_for_seconds}. Finish!")
        with self.timing.add_time("gpu_thread/run_step/observe_metrics"):
            self.monitor.metric("marl_trainer_sample_reuses").observe(replay_entry.reuses)

            sample_policy_version = replay_entry.sample.average_of("policy_version_steps",
                                                                   ignore_negative=True)
            sample_policy_version_min = replay_entry.sample.min_of("policy_version_steps",
                                                                   ignore_negative=True)
            if sample_policy_version is None or np.isnan(
                    sample_policy_version_min) or sample_policy_version_min < 0:
                self.logger.debug(
                    f"Ignored sample with version: avg {sample_policy_version}, min {sample_policy_version_min}."
                )
                return

            sample_version_difference = self.__trainer.policy.version - sample_policy_version_min
            if sample_version_difference > self.__preemption_steps:
                self.logger.debug(
                    f"Ignored sample with version: avg {sample_policy_version}, min {sample_policy_version_min} "
                    f"(current policy version {self.__trainer.policy.version}).")
                return

            staleness = self.__trainer.policy.version - sample_policy_version
            self.monitor.metric("marl_trainer_trainer_steps").set(sample_policy_version)

        with self.timing.add_time("gpu_thread/run_step/trainer_step"):
            with self.monitor.metric("marl_trainer_gpu_step_time").time():
                self.monitor.metric("marl_trainer_sample_staleness").observe(staleness)
                # TODO: Temporary workaround to overwrite non-numerical field `policy_name`.
                replay_entry.sample.policy_name = None
                st = time.perf_counter()
                log_entry = self.__trainer.step(replay_entry.sample)
                # self.logger.info(f"Trainer step time: {time.perf_counter() - st:.3f} secs. Current version: {self.__trainer.policy.version}.")
                log_entry.stats['sample_min_policy_version'] = sample_policy_version_min
                log_entry.stats['sample_version_difference'] = sample_version_difference
                log_entry.stats['buffer_qsize'] = self.__buffer.qsize()
                # self.logger.info(f"Trainer buffer qsize: {self.__buffer.qsize()}, "
                #                  f"sample min policy version: {sample_policy_version_min}, "
                #                  f"trainer policy version: {self.__trainer.policy.version}, "
                #                  f"policy version version difference: {sample_version_difference}, "
                #                  f"staleness: {staleness}.")
                self.__steps += 1
                self.monitor.metric("marl_trainer_trainer_steps").set(self.__trainer.policy.version)

        with self.timing.add_time("gpu_thread/run_step/update_priorities"):
            if log_entry.priorities is not None and isinstance(self.__buffer,
                                                               base.buffer.PrioritizedReplayBuffer):
                self.__buffer.update_priorities(replay_entry.sampling_indices, log_entry.priorities)

        with self.timing.add_time("gpu_thread/run_step/misc"):
            samples = replay_entry.sample.length(0) * replay_entry.sample.length(1)
            self.monitor.metric("marl_trainer_sample_batch_size").observe(samples)

            if self.__logging_control.check(steps=samples):
                # start = time.time()
                # while True:
                #     try:
                #         _ = self.__logging_queue.get_nowait()
                #     except queue.Empty:
                #         break
                try:
                    self.__logging_queue.put((self.__logging_control.interval_steps, log_entry), block=False)
                except queue.Full:
                    pass
                # self.logger.debug("Logged stats, took time: %.2fs", time.time() - start)
            if self.__is_master and log_entry.agree_pushing and self.__push_control.check():
                start = time.time()
                while True:
                    try:
                        _ = self.__checkpoint_push_queue.get_nowait()
                    except queue.Empty:
                        break
                self.__checkpoint_push_queue.put(self.__trainer.get_checkpoint(), block=False)
                self.logger.debug("Pushed params, took time: %.2fs", time.time() - start)

    def get_step_result(self) -> Tuple[int, Optional[api.trainer.TrainerStepResult]]:
        """Get results of trainer step.
        Returns:
            samples: sample count of this trainer step.
            trainer_step_result: result of this trainer step.
        """
        try:
            return self.__logging_queue.get_nowait()
        except queue.Empty:
            return -1, api.trainer.TrainerStepResult({}, -1)

    def get_checkpoint(self) -> Any:
        """Get checkpoint published by the trainer.
        Returns:
            trainer_checkpoint: checkpoint to be saved/published.
        """
        try:
            return self.__checkpoint_push_queue.get_nowait()
        except queue.Empty:
            return None

    def stop_at_step(self, stop_at_step):
        self.__interrupting = True
        self.__interrupt_at_step = stop_at_step
        self.__thread.join(timeout=60)
        if self.__thread.is_alive():
            raise RuntimeError("Failed to join GPU thread. (timeout=15s)")


class TrainerWorker(worker_base.Worker):

    def __init__(self, server=None, lock=None):
        super().__init__(server, lock)
        self.timing = base.timeutil.Timing()
        self._timing_cnt = 0
        self.config = None
        self.policy_name = None
        self.gpu_thread = None
        self.__stream = None
        self.__buffer = None
        self.__param_db: Optional[parameter_db.ParameterDBClient] = None
        self.__ddp_env_resolved = False
        self.__is_master = False
        self.__ddp_init_address = None
        self.__ddp_rank = None
        self.__push_tagged_control = None
        self.__gc_frequency_control = base.timeutil.FrequencyControl(
            frequency_seconds=PARAMETER_DB_GC_FREQUENCY_SECONDS)

        # debug zerocopy
        self.consume_time = 0
        self.batch_time = 0

    def __stop_gpu_thread(self):
        """This method tells gpu thread when to stop running.
        """

        def find_safe_interrupt_step(my_step, assume_max_difference=DDP_MAX_TRAINER_STEP_DIFFERENCE):
            for i in range(my_step - assume_max_difference, my_step + assume_max_difference + 1):
                if i % (assume_max_difference * 2 + 1) == 0:
                    return i + assume_max_difference + 3  # +1 should be enough, +3 is just in case.
            else:
                raise RuntimeError("This is not possible.")

        if self.gpu_thread is not None:
            curr_step = self.gpu_thread.distributed_steps
            self.logger.info(
                f"I am at step {curr_step}. "
                f"I think step difference should be no-larger than {DDP_MAX_TRAINER_STEP_DIFFERENCE}.")
            stop_at_step = find_safe_interrupt_step(curr_step)
            self.logger.info(f"I think we could stop at step {stop_at_step}.")
            self.gpu_thread.stop_at_step(stop_at_step)
            self.gpu_thread = None

    def __start_gpu_thread(self, trainer, dist_kwargs):
        self.gpu_thread = GPUThread(buffer=self.__buffer,
                                    trainer=trainer,
                                    is_master=self.__is_master,
                                    worker_info=self.config.worker_info,
                                    log_frequency_seconds=self.config.log_frequency_seconds,
                                    log_frequency_steps=self.config.log_frequency_steps,
                                    push_frequency_seconds=self.config.push_frequency_seconds,
                                    push_frequency_steps=self.config.push_frequency_steps,
                                    preemption_steps=self.__preemption_steps,
                                    dist_kwargs=dist_kwargs,
                                    train_for_seconds=self.config.train_for_seconds)
        self.gpu_thread.start()

    def _stats(self):
        return dict(self.gpu_thread.stats(), consume_time=self.consume_time, batch_time=self.batch_time)

    def _configure(self, cfg: config.TrainerWorker):
        self.config = cfg
        self.policy_name = cfg.policy_name
        self.__foreign_policy = cfg.foreign_policy
        self.__experiment_name = self.config.worker_info.experiment_name
        self.__trial_name = self.config.worker_info.trial_name
        self.__worker_index = str(cfg.worker_info.worker_index)

        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_determinisitc
        self.__preemption_steps = cfg.preemption_steps

        self.__zero_copy = cfg.buffer_zero_copy
        if not isinstance(cfg.sample_stream,
                          str) and cfg.sample_stream.type_ == config.SampleStream.Type.SHARED_MEMORY:
            if self.__zero_copy:
                # Align buffer and sample stream configuration to store data in a zero-copy manner.
                cfg.sample_stream.reuses = cfg.buffer_args.get("reuses", cfg.sample_stream.reuses)
                cfg.sample_stream.batch_size = cfg.buffer_args.get("batch_size", cfg.sample_stream.batch_size)
                cfg.sample_stream.qsize = cfg.buffer_args.get("max_size", cfg.sample_stream.qsize)

        if not isinstance(
                cfg.sample_stream, str
        ) and cfg.sample_stream.type_ == config.SampleStream.Type.SHARED_MEMORY and self.__zero_copy:
            self.__buffer = base.buffer.make_buffer("simple_queue", max_size=100)
        else:
            self.__buffer = base.buffer.make_buffer(cfg.buffer_name, **cfg.buffer_args)

        self.logger.info("Before make consumer")
        self.__stream = distributed.system.sample_stream.make_consumer(cfg.sample_stream,
                                                                       worker_info=cfg.worker_info)
        self.logger.info("After make consumer")
        self.__param_db = parameter_db.make_db(cfg.parameter_db, worker_info=cfg.worker_info)

        # Reveal DDP identity of this worker to world.
        self.__reveal_ddp_identity()
        self.__ddp_env_resolved = False

        r = self.config.worker_info
        r.policy_name = self.policy_name
        return r

    def _reconfigure(self, policy_name=None, trainer_config=None, src_policy_name=None):
        if policy_name is not None:
            self.config.policy_name = policy_name
            self.config.worker_info.policy_name = policy_name
            self.policy_name = policy_name

        if trainer_config is not None:
            self.config.trainer = trainer_config
            self.__stop_gpu_thread()
            if policy_name is not None:
                # Train a new policy.
                self.__reveal_ddp_identity()
                self.__ddp_env_resolved = False
                wandb_name = self.config.worker_info.wandb_name
                new_wandb_args = dict(name=f"{policy_name}-{wandb_name}")
                self.monitor.new_wandb_run(new_wandb_args)
            elif src_policy_name is not None:
                # Clone parameters from the source policy.
                trainer = api.trainer.make(trainer_config, self.config.policy)
                try:
                    checkpoint = self.__param_db.get(src_policy_name)
                    checkpoint["steps"] = self.__param_db.get(self.policy_name)["steps"]
                    trainer.policy.load_checkpoint(checkpoint)
                    if self.__is_master:
                        self.__param_db.push(self.policy_name, checkpoint, version=str(checkpoint["steps"]))
                except FileNotFoundError as e:
                    self.logger.warning(f"No model of {src_policy_name} found.")
                    raise e
                dist_kwargs = dict(world_size=self.__world_size,
                                   rank=self.__ddp_rank,
                                   init_method=self.__ddp_init_address)
                self.__start_gpu_thread(trainer, dist_kwargs=dist_kwargs)
            else:
                self.logger.info("Either policy_name or src_policy_name should be specified to reconfigure "
                                 "trainer.")
                raise NotImplementedError

    def __reveal_ddp_identity(self):
        name_resolve.add_subentry(names.trainer_ddp_peer(self.__experiment_name, self.__trial_name,
                                                         self.policy_name),
                                  self.__worker_index,
                                  keepalive_ttl=30)

    def __setup_ddp_and_gpu_thread(self):
        """Setup pytorch ddp processes, and algorithms.
        """
        self.logger.info(f"Setup trainer worker {self.__worker_index} for policy {self.policy_name}")

        peers = list(
            sorted(
                name_resolve.get_subtree(
                    names.trainer_ddp_peer(self.__experiment_name, self.__trial_name, self.policy_name))))
        ddp_name_resolve = names.trainer_ddp_master(self.__experiment_name, self.__trial_name,
                                                    self.policy_name)

        assert len(peers) == len(set(peers)), f"Duplicated trainer worker index."
        self.__world_size = len(peers)

        self.__ddp_rank = peers.index(self.__worker_index)
        if self.__ddp_rank == 0:
            import socket
            host_ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            self.__ddp_init_address = f"tcp://{host_ip}:{port}"
            name_resolve.add(ddp_name_resolve, self.__ddp_init_address, keepalive_ttl=15)
        else:
            try:
                self.__ddp_init_address = name_resolve.wait(ddp_name_resolve, timeout=5)
            except TimeoutError:
                raise TimeoutError(
                    f"DDP trainer(index:{self.__worker_index}), rank {self.__ddp_rank} for policy "
                    f"{self.policy_name} wait for ddp_init_method timeout.")

        trainer = api.trainer.make(self.config.trainer, self.config.policy)
        if self.__ddp_rank == 0:
            self.__is_master = True

        if self.config.load_buffer_on_restart:
            try:
                self.__buffer = self.__param_db.get(f"{self.policy_name}_buffer_{self.__worker_index}",
                                                    identifier="latest")['buffer']
                self.logger.info(f"Loaded saved buffer from param_db.")
            except FileNotFoundError:
                self.logger.info(f"Saved buffer not found in param_db. Skip.")

        try:
            # Loading parameters for master in sufficient for pytorch DDP.
            # Things might be different in other cases.
            checkpoint = self.__param_db.get(self.policy_name, identifier="latest")
            trainer.load_checkpoint(checkpoint)
            self.logger.info(f"Loaded model with tag latest. You can re-run your "
                             f"experiment by deleting your saved model parameters from parameter DB.")
        except FileNotFoundError:
            self.__maybe_read_foreign_policy(trainer)
            if self.__is_master:
                self.logger.warning("No saved model found. This must be the first time you run this trial."
                                    "DDP master is pushing the first version.")
                self.logger.info(f"W&B resumed: {self.monitor.wandb_resumed()}")
                trainer.policy.inc_version()  # Increase policy version from -1 to 0. We start training now.
                self.__param_db.push(self.policy_name, trainer.get_checkpoint(), str(trainer.policy.version))
        dist_kwargs = dict(world_size=self.__world_size,
                           rank=self.__ddp_rank,
                           init_method=self.__ddp_init_address)
        self.__start_gpu_thread(trainer, dist_kwargs=dist_kwargs)
        if self.config.push_tag_frequency_minutes is not None:
            self.__push_tagged_control = base.timeutil.FrequencyControl(
                frequency_seconds=self.config.push_tag_frequency_minutes * 60, initial_value=True)

    def __maybe_read_foreign_policy(self, trainer):
        if self.__foreign_policy is not None:
            p = self.__foreign_policy
            spec = p.param_db
            e = p.foreign_experiment_name or self.__experiment_name
            f = p.foreign_trial_name or self.__trial_name
            pn = p.foreign_policy_name or self.policy_name
            i = p.foreign_policy_identifier or "latest"

            foreign_db = parameter_db.make_db(spec=spec,
                                              worker_info=config.WorkerInformation(experiment_name=e,
                                                                                   trial_name=f))
            if self.__foreign_policy.absolute_path is not None:
                checkpoint = foreign_db.get_file(self.__foreign_policy.absolute_path)
                self.logger.info(f"Loaded checkpoint: {self.__foreign_policy.absolute_path}")
            else:
                checkpoint = foreign_db.get(name=pn, identifier=i)
                self.logger.info(f"Loaded foreign parameter: {e} -> {f} -> {pn} -> {i}")
            trainer.policy.load_checkpoint(checkpoint)

    def start_monitoring(self):
        r = super().start_monitoring()

        metrics = dict(marl_trainer_gpu_step_time="Summary",
                       marl_trainer_sample_batch_size="Summary",
                       marl_trainer_sample_staleness="Summary",
                       marl_trainer_sample_reuses="Summary",
                       marl_trainer_trainer_steps="Gauge",
                       marl_trainer_received_sample_steps="Gauge")

        self.monitor.update_metrics(metrics)
        return r

    def _poll(self):
        if not self.__ddp_env_resolved:
            self.__setup_ddp_and_gpu_thread()
            self.gpu_thread.init_monitor(self.monitor)
            self.__stream.init_monitor(self.monitor)
            self.__ddp_env_resolved = True

        if not self.gpu_thread.is_alive():
            self.__save_buffer_if_necessary()
            raise RuntimeError("Exception in trainer worker gpu thread.")

        with self.timing.add_time("trainer_worker/consume"):
            # self.__stream.consume_to(self.__buffer, max_iter=1024)
            # With a bounded iteration count, logging and checkpoint can be processed with controlled delay.
            count = self.__stream.consume_to(self.__buffer, max_iter=1024)

        with self.timing.add_time("trainer_worker/log"):
            # Track and log training results.
            samples = 0
            batches = 0
            for _ in range(100):
                step_samples, trainer_step_result = self.gpu_thread.get_step_result()
                if step_samples <= 0:
                    break
                samples += step_samples
                batches += 1
                if len(trainer_step_result.stats) > 0 and self.__is_master:
                    self.logger.info("Logging stats: %s", trainer_step_result.stats)
                    self.monitor.log_wandb(trainer_step_result.stats, step=trainer_step_result.step)

        with self.timing.add_time("trainer_worker/checkpointing"):
            # Checkpoint.
            for _ in range(8):
                checkpoint = self.gpu_thread.get_checkpoint()
                if checkpoint is None:
                    break
                else:
                    ckpt = checkpoint
                    tags = []
                    if self.__push_tagged_control is not None and self.__push_tagged_control.check():
                        tags.append("latest_tagged")
                        tags.append(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
                        self.logger.info("Saving a tagged policy version: %s", tags[-1])
                    self.__param_db.push(self.policy_name, ckpt, version=str(ckpt["steps"]), tags=tags)
                    # TODO: make this configurable.

                if self.__gc_frequency_control.check():
                    self.__param_db.gc(self.policy_name, max_untagged_version_count=10)
        self._timing_cnt += 1
        if self._timing_cnt % 100 == 0:
            total_time = sum(v for k, v in self.timing.items() if k.count('/') == 1)
            msg = "\n==========================================\n"
            for k, v in self.timing.items():
                msg += "{} proportion: {:.3f}\n".format(k, v / total_time)
            msg += "==========================================\n"
            msg += "Total time: {:.3f} secs, total step: {}".format(total_time, self._timing_cnt)
            # self.logger.info(msg)
        return worker_base.PollResult(sample_count=samples, batch_count=batches)

    def __save_buffer_if_necessary(self):
        if self.config.save_buffer_on_exit:
            try:
                # Each trainer worker saves its own buffer.
                policy_version = self.__param_db.version_of(self.policy_name, identifier="latest")
            except FileNotFoundError:
                policy_version = 0
            self.__param_db.push(f"{self.policy_name}_buffer_{self.__worker_index}",
                                 dict(buffer=self.__buffer, steps=policy_version),
                                 version=str(policy_version),
                                 tags="latest")
            self.logger.info("Saved replay buffer in parameter db. "
                             "You can load the buffer by turning on the load_buffer_on_restart option"
                             " in your next run.")

    def exit(self):
        self.__save_buffer_if_necessary()
        super(TrainerWorker, self).exit()
        self.__stream.close()
        self.__stop_gpu_thread()

    def interrupt(self):
        self.__stop_gpu_thread()
        self.__stream.close()
        super(TrainerWorker, self).interrupt()
