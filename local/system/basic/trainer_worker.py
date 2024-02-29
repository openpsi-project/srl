from typing import Optional, Tuple, Any
import datetime
import logging
import numpy as np
import queue
import time
import threading
import torch

from base.gpu_utils import set_cuda_device
from base.network import find_free_port
from local.system.sample_stream import make_consumer
from local.system import worker_base
import api.config as config
import api.trainer
import base.buffer
import base.timeutil
import base.names as names
import local.system.parameter_db as parameter_db
import base.name_resolve as name_resolve

DDP_MAX_TRAINER_STEP_DIFFERENCE = 3
PARAMETER_DB_GC_FREQUENCY_SECONDS = 300


class GPUThread:

    def __init__(self, buffer, trainer, is_master, worker_info, log_frequency_seconds, log_frequency_steps,
                 push_frequency_seconds, push_frequency_steps, dist_kwargs):
        self.logger = logging.getLogger("gpu-thread")
        self.dist_kwargs = dist_kwargs
        self.__logging_queue = queue.Queue(8)
        self.__checkpoint_push_queue = queue.Queue(8)
        self.__replay_entry = None

        self.__buffer = buffer
        self.__is_master = is_master
        self.__trainer: api.trainer.Trainer = trainer

        self.__interrupting = False
        self.__interrupt_at_step = 1e10
        self.__steps = 0
        self.__thread = threading.Thread(target=self._run, daemon=True)

        # Monitoring related below.
        self.__start_time_ns = time.monotonic_ns()
        labels = dict(
            host=base.network.gethostname(),
            experiment=worker_info.experiment_name,
            trial=worker_info.trial_name,
            worker=worker_info.worker_type,
            worker_id=worker_info.worker_index,
            policy=worker_info.policy_name,
        )

        self.__logging_control = base.timeutil.FrequencyControl(frequency_seconds=log_frequency_seconds,
                                                                frequency_steps=log_frequency_steps)
        self.__push_control = base.timeutil.FrequencyControl(frequency_seconds=push_frequency_seconds,
                                                             frequency_steps=push_frequency_steps)

    @property
    def distributed_steps(self):
        return self.__steps

    def stats(self):
        duration = (time.monotonic_ns() - self.__start_time_ns) / 1e9
        return {}

    def is_alive(self):
        return self.__thread.is_alive()

    def start(self):
        self.__thread.start()

    def _run(self):
        set_cuda_device(self.__trainer.policy.device)
        self.__trainer.distributed(**self.dist_kwargs)
        while True:
            if self.__interrupting:
                self.__interrupt_loop()
                break
            self._run_step()

    def __interrupt_loop(self):
        self.logger.info("Entering stopping loop.")
        while self.__steps < self.__interrupt_at_step:
            if self.__replay_entry is None:
                break
            self._run_step_on_entry(self.__replay_entry)
        self.logger.info(f"Stopping at {self.__steps}!")

    def _run_step(self):
        if not self.__buffer.empty():
            self.__replay_entry = self.__buffer.get()
            self._run_step_on_entry(self.__replay_entry)
        else:
            time.sleep(
                0.005
            )  # to avoid locking the buffer. We should remove this line when our buffer is thread-safe.

    def _run_step_on_entry(self, replay_entry):

        sample_policy_version = replay_entry.sample.average_of("policy_version_steps", ignore_negative=True)
        sample_policy_version_min = replay_entry.sample.min_of("policy_version_steps", ignore_negative=True)
        if sample_policy_version is None or np.isnan(
                sample_policy_version_min) or sample_policy_version_min < 0:
            self.logger.debug(
                f"Ignored sample with version: avg {sample_policy_version}, min {sample_policy_version_min}.")
            return

        staleness = self.__trainer.policy.version - sample_policy_version
        # TODO: Temporary workaround to overwrite non-numerical field `policy_name`.
        replay_entry.sample.policy_name = None
        log_entry = self.__trainer.step(replay_entry.sample)
        self.__steps += 1

        samples = replay_entry.sample.length(0) * replay_entry.sample.length(1)

        if self.__logging_control.check(steps=samples):
            start = time.time()
            while True:
                try:
                    _ = self.__logging_queue.get_nowait()
                except queue.Empty:
                    break
            self.__logging_queue.put((self.__logging_control.interval_steps, log_entry), block=False)
            self.logger.debug("Logged stats, took time: %.2fs", time.time() - start)
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

    def __init__(self, server=None):
        super().__init__(server)
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
                                    dist_kwargs=dist_kwargs)
        self.gpu_thread.start()

    def _stats(self):
        return self.gpu_thread.stats()

    def _configure(self, cfg: config.TrainerWorker):
        self.config = cfg
        self.policy_name = cfg.policy_name
        self.__foreign_policy = cfg.foreign_policy
        self.__experiment_name = self.config.worker_info.experiment_name
        self.__trial_name = self.config.worker_info.trial_name
        self.__worker_index = str(cfg.worker_info.worker_index)

        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_determinisitc

        self.__buffer = base.buffer.make_buffer(cfg.buffer_name, **cfg.buffer_args)
        self.__stream = make_consumer(cfg.sample_stream, worker_info=cfg.worker_info)
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
                self._new_wandb_run(new_wandb_args)
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
                                  keepalive_ttl=5)

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
                    "{self.policy_name} wait for ddp_init_method timeout.")

        trainer = api.trainer.make(self.config.trainer, self.config.policy)
        if self.__ddp_rank == 0:
            self.__is_master = True

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
                self.logger.info(f"W&B resumed: {self.wandb_run.resumed}")
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

    def _poll(self):
        if not self.__ddp_env_resolved:
            self.__setup_ddp_and_gpu_thread()
            self.__ddp_env_resolved = True

        if not self.gpu_thread.is_alive():
            raise RuntimeError("Exception in trainer worker gpu thread.")

        # With a bounded iteration count, logging and checkpoint can be processed with controlled delay.
        self.__stream.consume_to(self.__buffer, max_iter=1024)

        # Track and log training results.
        samples = 0
        batches = 0
        for _ in range(8):
            step_samples, trainer_step_result = self.gpu_thread.get_step_result()
            if step_samples <= 0:
                break
            samples += step_samples
            batches += 1
            if len(trainer_step_result.stats) > 0 and self.__is_master:
                self.wandb_run.log(trainer_step_result.stats, step=trainer_step_result.step)

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

        return worker_base.PollResult(sample_count=samples, batch_count=batches)

    def exit(self):
        super(TrainerWorker, self).exit()
        self.__stop_gpu_thread()

    def interrupt(self):
        self.__stop_gpu_thread()
        super(TrainerWorker, self).interrupt()
