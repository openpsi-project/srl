from typing import Optional, Dict, List, Tuple
import dataclasses
import enum
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time

from distributed.infra.scheduler.utils import SlurmResource, SlurmTaskSpecification, write_hostfile
from distributed.system import RL_WORKERS
import base.gpu_utils
import base.names

logger = logging.getLogger("scheduler")


def log_path(job_name, task_name):
    if task_name is None:
        return f"/data/marl/logs/iclr24-benchmark/{base.names.USER_NAMESPACE}/{job_name}"
    else:
        return f"/data/marl/logs/iclr24-benchmark/{base.names.USER_NAMESPACE}/{job_name}/{task_name}"


class TaskState(enum.Enum):
    NOT_FOUND = 0
    PENDING = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5

    def active(self):
        return self == self.PENDING or self == self.RUNNING


@dataclasses.dataclass
class TaskInfo:
    name: str
    state: TaskState
    host: str = None  # The host on which the task is/was running. None if the task had not run.
    start_time: str = None
    slurm_id: str = None  # Slurm only. The Slurm id of the task.


class SchedulerError(Exception):
    pass


class TaskException(Exception):

    def __init__(self, job_name, task_name, host, reason: TaskState):
        super().__init__(f"Task {job_name}:{task_name} {reason} at node {host}")
        self.job_name = job_name
        self.task_name = task_name
        self.host = host
        self.reason = reason


class SchedulerClient:

    def __init__(self, job_name):
        self.job_name = job_name

    def submit(self, task_name, cmd, **kwargs):
        """Submits a task to the scheduler. Raises exception if the task is already running.

        Args:
            task_name: Name of the task. The job name is specified when initializing the client.
            cmd (str or List[str]): The command of the task process. If this is str, the command is parsed by
                shell; otherwise it is executed directly.
        """
        raise NotImplementedError()

    def submit_array(self, task_name, cmd, count, **kwargs):
        """Submits an array of tasks to the scheduler.

        Args:
            task_name: The tasks share the same name.
            cmd: Command template of the tasks that may contain an "{index}" format placeholder.
            count: Number of tasks. The indices of the tasks shall be 0..count-1.
        """
        for index in range(count):
            self.submit(task_name + "_" + str(index), cmd.format(index=index), **kwargs)

    def stop(self, task_name):
        """Stops a running task. Raises exception if there is no such task, but passes if the task has stopped
        either successfully or not.
        """
        raise NotImplementedError()

    def stop_all(self):
        """Stops the whole job.
        """
        raise NotImplementedError()

    def find(self, task_name) -> Optional[TaskInfo]:
        """Gets the status of a task of this job.

        Args:
            task_name: Name of the task.

        Returns:
            A TaskInfo if the task is found, or None otherwise.
        """
        raise NotImplementedError()

    def find_all(self, task_name_regex=".*") -> List[TaskInfo]:
        """Finds tasks.

        Args:
            task_name_regex: Task name regex.

        Returns:
            A list of found TaskInfo.
        """
        raise NotImplementedError()

    def wait(self, timeout=None, **kwargs):
        """Waits until all tasks submitted via this client instance finish.
        """
        raise NotImplementedError()


class LocalSchedulerClient(SchedulerClient):
    """Instead of talking to the scheduler server (the typical behaviour), this client starts tasks directly
    on the local host and keeps a collection of task processes.
    """

    def __init__(self, job_name, **kwargs):
        super().__init__(job_name)
        self._tasks: Dict[str, subprocess.Popen] = {}

    def __del__(self):
        self.wait()

    def submit(self, task_name, cmd, **kwargs):
        assert task_name not in self._tasks
        if 'pw' in task_name:
            # round-robin assigning gpus to policy workers
            if base.gpu_utils.gpu_count() >= 1:
                idx = int(task_name.split('_')[-1]) % base.gpu_utils.gpu_count()
                cmd = f"CUDA_VISIBLE_DEVICES={idx} {cmd}"
        process = subprocess.Popen(cmd, shell=isinstance(cmd, str))
        self._tasks[task_name] = process

    def stop(self, task_name):
        assert task_name in self._tasks
        logger.info("Stopping local process, pid: %d", self._tasks[task_name].pid)
        self._tasks[task_name].kill()
        self._tasks[task_name].wait()
        del self._tasks[task_name]

    def stop_all(self):
        for name in list(self._tasks):
            self.stop(name)

    def find(self, task_name):
        if task_name in self._tasks:
            return TaskInfo(name=task_name, state=TaskState.RUNNING, host="localhost")
        else:
            return TaskInfo(name=task_name, state=TaskState.NOT_FOUND)

    def find_all(self, task_name_regex=".*"):
        rs = []
        for name in self._tasks:
            if re.fullmatch(task_name_regex, name):
                rs.append(self.find(name))
        return rs

    def wait(self, timeout=None, update=False, **kwargs):
        logger.info("Waiting %d local running processes, pids: %s", len(self._tasks),
                    " ".join(str(task.pid) for task in self._tasks.values()))
        to_remove = []
        try:
            for key, task in self._tasks.items():
                task.wait(timeout)
                if update:
                    to_remove.append(key)
        except subprocess.TimeoutExpired:
            raise TimeoutError()
        finally:
            for k in to_remove:
                self._tasks.pop(k)


class SlurmSchedulerClient(SchedulerClient):
    """Uses Slurm (https://slurm.schedmd.com/overview.html).
    """
    SQUEUE_FIELDS = [
        "JobID",
        "State",
        "StartTime",
        "Name",
        "NodeList",
        "UserName",
        "MaxCPUs",
        "cpus-per-task",
        "NumTasks",
        "tres-alloc",
    ]

    STATUS_MAPPING = {
        "RUNNING": TaskState.RUNNING,
        "COMPLETING": TaskState.RUNNING,
        "PENDING": TaskState.PENDING,
        "CANCELLED": TaskState.CANCELLED,
        "FAILED": TaskState.FAILED,
        "COMPLETED": TaskState.COMPLETED,
        "OUT_OF_MEMORY": TaskState.FAILED,
    }

    def __init__(self, job_name):
        super().__init__(job_name)
        self._tasks = {}
        self.__pending_task_specs = []

    def submit(self, task_name, cmd, **kwargs):
        self.submit_array(task_name, cmd, count=1, **kwargs)

    def submit_array(
        self,
        task_name,
        cmd,
        count,
        environment_vars=None,
        cpu=1,
        gpu_type: str = "geforce",
        gpu=0,
        mem=1024,
        container_image="marl/marl-cpu",
        node_type: Optional[List[str]] = None,
        node_list: Optional[str] = None,
        exclude: Optional[str] = None,
        **kwargs,
    ):

        if environment_vars is None:
            environment_vars = {}

        # Resolve array and gres context.
        ntasks = count
        group_size = 1
        gpus = count * gpu
        if 0 < gpu < 1:
            # n-1, schedule multiple times
            group_size = math.floor(1 / gpu)
            ntasks = gpus = math.ceil(ntasks / group_size)
            cpu = cpu * group_size
            mem = mem * group_size
            gpu = 1

        resource_requirement = SlurmResource(mem=mem, cpu=cpu, gpu=gpu, gpu_type=gpu_type)
        task_spec = SlurmTaskSpecification(
            task_name=task_name,
            ntasks=ntasks,
            resource_requirement=resource_requirement,
            cmd=cmd,
            job_name=self.job_name,
            container_image=container_image,
            container_mounts="/data:/data",
            env_vars=environment_vars,
            nodelist=node_list,
            exclude=exclude,
            group_size=group_size,
            hostfile=(resource_requirement.gpu > 0),
        )
        self.__pending_task_specs.append(task_spec)
        logger.info("Registered Slurm task: %s (count=%s)", task_name, count)

    def __commit_one(self, spec: SlurmTaskSpecification):
        name = self.__slurm_name(spec.task_name)
        output = log_path(self.job_name, spec.task_name)
        os.makedirs(os.path.dirname(output), exist_ok=True, mode=0o775)

        multi_prog_file = log_path(self.job_name, spec.task_name) + ".multiprog"
        with open(multi_prog_file, "w") as f:
            f.write(f"0-{spec.ntasks-1} {spec.cmd.format(index='%t')}\n")

        hostfile = log_path(self.job_name, spec.task_name) + ".hostfile"
        if spec.hostfile:
            try:
                write_hostfile(spec.resource_requirement, spec.ntasks, hostfile, spec.node_type,
                               spec.nodelist, spec.exclude)
            except ValueError as e:
                for task_info in self._tasks.values():
                    self.stop(task_info.name)
                raise e

        mem = spec.resource_requirement.mem
        cpu = spec.resource_requirement.cpu
        gpu = spec.resource_requirement.gpu
        gpu_type = spec.resource_requirement.gpu_type
        ntasks = spec.ntasks
        node_list = spec.nodelist
        exclude = spec.exclude
        container_image = spec.container_image
        environment_vars = spec.env_vars

        # Setup resource allocation.
        lines = [
            '#!/bin/bash',
            f'#SBATCH --job-name={name}',
            f'#SBATCH --output={output}',
            f'#SBATCH --ntasks={ntasks}',
            f'#SBATCH --cpus-per-task={cpu}',
            f'#SBATCH --mem-per-cpu={mem // max(1, cpu)}',
        ]

        if gpu:
            assert gpu == 1, gpu
            lines += [f'#SBATCH --gpus-per-task={gpu_type}:{gpu}']
        else:
            lines += [f'#SBATCH --partition=cpu']
            if node_list is not None:
                lines += [f'#SBATCH --nodelist={node_list}']
            if exclude is not None:
                lines += [f'#SBATCH --exclude={exclude}']

        srun_env = os.environ.copy()
        if spec.hostfile:
            srun_env['SLURM_HOSTFILE'] = hostfile
            lines += [f'#SBATCH --distribution=arbitrary']

        # Setup step command.
        srun_flags = [
            f"--ntasks={ntasks}",
            f"--cpus-per-task={cpu}",
            f"--mem-per-cpu={mem // max(1, cpu)}",
            f"--container-image={container_image}",
            f"--container-mounts=/data:/data",
            f"--container-mount-home",
            f"--export={','.join(str(k)+'='+str(v) for k, v in environment_vars.items())}",
            f"--multi-prog",
        ]
        if gpu:
            srun_flags.append(f"--gpus-per-task={gpu_type}:{gpu}")

        srun_cmd = f'srun -l {" ".join(srun_flags)} {multi_prog_file}'
        if spec.group_size > 1:
            srun_cmd += f" -g {spec.group_size}"

        lines += [
            'echo "[Runner] StartTime: $(date -u)"',
            'echo "[Runner] Host: $(hostname)"',
            "echo '[Runner] Command: {}'".format(srun_cmd),
            "echo '[Runner] Log: {}'".format(output),
            'echo "[Runner] CudaVisible: $CUDA_VISIBLE_DEVICES"',
            'echo "[Runner] CudaMpsPerc: $CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"',
            srun_cmd,
            'RETCODE=$?',
            'echo "[Runner] FinishTime: $(date -u)"',
            'echo "[Runner] RetCode: $RETCODE"',
            'echo "[Runner] ------------"',
            'exit $RETCODE',
        ]
        logger.debug("\n".join(["SBATCH script:"] + lines))

        script = '\n'.join(lines).encode('ascii')
        r = subprocess.check_output(['sbatch', '--parsable'], input=script,
                                    env=srun_env).decode('ascii').strip()
        logger.info("Submitted Slurm task %s: %s", r, name)
        self._tasks.update({r: TaskInfo(name=spec.task_name, state=TaskState.PENDING)})
        return r

    def __commit_all(self):
        for task_spec in self.__pending_task_specs:
            self.__commit_one(task_spec)
            # Wait for a while to avoid allocating to the same node in slurm hostfile.
            time.sleep(5)
        self.__pending_task_specs = []

    def stop(self, task_name):
        r = self.find(task_name)
        if r is not None and r.state in {TaskState.RUNNING, TaskState.PENDING}:
            subprocess.check_call(["scancel", str(r.slurm_id)])
            logger.info("Cancelled Slurm task %d: %s", r.slurm_id, self.__slurm_name(task_name))
            time.sleep(0.2)
            self.__update_subset([r.slurm_id])

    def stop_all(self):
        rs = self.__query_tasks(list(self._tasks.keys()))
        ids = [r.slurm_id for r in rs if r.state in {TaskState.RUNNING, TaskState.PENDING}]
        group_ids = set([i.split("_")[0] for i in ids])
        logger.info(f"STOPPING SLURM IDS: {group_ids}")
        if len(ids) == 0:
            logger.info("No task to stop, skipping")
        else:
            subprocess.check_call(["scancel", ",".join(group_ids)])
            logger.info("Cancelled %d Slurm tasks: %s", len(group_ids), ",".join(group_ids))
        time.sleep(0.2)
        self.wait(check_status=(),
                  remove_status=(TaskState.CANCELLED, TaskState.NOT_FOUND, TaskState.FAILED,
                                 TaskState.COMPLETED))

    def find(self, task_name):
        for r in self._tasks.values():
            if r.name == task_name:
                self.__update_subset(r.slurm_id)
                return self._tasks[r.slurm_id]
        return TaskInfo(name=task_name, state=TaskState.NOT_FOUND)

    def find_all(self, task_name_regex=".*"):
        self.__update_all()
        rs = []
        for r in self._tasks.values():
            if re.fullmatch(task_name_regex, r.name):
                rs.append(r)
        return rs

    def __show_log(self, task_name):
        try:
            terminal_columns = os.get_terminal_size().columns
        except OSError:
            terminal_columns = shutil.get_terminal_size().columns
        logger.info(f"Showing log of task: {task_name}\n\n{'-'*terminal_columns}")
        subprocess.Popen(["tail", "-n50", log_path(self.job_name, task_name)]).wait(timeout=3)
        logger.info(f"End of log: {task_name}\n\n{'-'*terminal_columns}")

    def wait(
            self,
            timeout=None,
            check_status: Tuple[TaskState,
                                ...] = (TaskState.CANCELLED, TaskState.FAILED, TaskState.NOT_FOUND),
            remove_status: Tuple[TaskState, ...] = (TaskState.COMPLETED,),
            update=False,
    ):
        # before wait, commit all remaining pending task specs
        self.__commit_all()
        # begin wait
        deadline = None if timeout is None else time.time() + timeout
        left = set(self._tasks)
        logger.info(str(self._tasks))
        num_jobs_left = len(left)
        logger.info(f"Waiting for {num_jobs_left} jobs.")
        while len(left) > 0:
            if len(left) < num_jobs_left:
                num_jobs_left = len(left)
                logger.info(f"Waiting for {num_jobs_left} jobs.")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(f"Timeout waiting for {self.job_name}: {', '.join(sorted(left))}")
            try:
                self.__update_all()
            except subprocess.CalledProcessError:
                logger.warning(
                    "Calling squeue failed. Check slurm manually if you continue to see this warning.")
                time.sleep(30)
                continue
            for i in list(left):
                r = self._tasks[i]
                if r.slurm_id is None:
                    continue
                if r.state in check_status:
                    self.__show_log(r.name)
                    raise TaskException(job_name=self.job_name,
                                        task_name=r.name + "_" + i.split("_")[-1],
                                        host=r.host,
                                        reason=r.state)
                if r.state in remove_status:
                    logger.info(f"Task {r.name + '_' + i.split('_')[-1]} is {r.state}.(Removed)")
                    left.remove(r.slurm_id)
                    if update:
                        self._tasks.pop(r.slurm_id)
            time.sleep(2)

    def __slurm_name(self, task_name):
        return f"{self.job_name}:{task_name}"

    def __task_name(self, slurm_name):
        prefix = f"{self.job_name}:"
        if not slurm_name.startswith(prefix):
            raise ValueError(f"Slurm name '{slurm_name}' does not start with '{prefix}'")
        return slurm_name[len(prefix):]

    def __query_tasks(self, slurm_ids, status="all", delimiter="__PSI__"):
        squeue_format = f":.{delimiter},".join(SlurmSchedulerClient.SQUEUE_FIELDS)
        cmd = ["squeue", "-O", squeue_format, f"-t{status}"]
        if slurm_ids is not None:
            cmd += ["-j", ",".join([str(s) for s in slurm_ids])]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("ascii").strip()
        rs = []
        for line in output.split("\n")[1:]:
            job_id, state, start_time, slurm_name, node_list, *_ = line.split(delimiter)
            if slurm_ids is not None:
                assert slurm_name.startswith(f"{self.job_name}:")
            elif not slurm_name.startswith(f"{self.job_name}:"):
                continue
            task_name = self.__task_name(slurm_name)
            job_ids = self.__parse_job_ids(job_id)
            for ji in job_ids:
                rs.append(
                    TaskInfo(name=task_name,
                             state=SlurmSchedulerClient.STATUS_MAPPING[state],
                             host=node_list,
                             start_time=start_time,
                             slurm_id=ji.strip()))
        return rs

    def __parse_job_ids(self, job_id):
        """This method may be optimized as we no longer user array jobs.
        """
        if "[" in job_id and "]" in job_id and "-" in job_id:
            batch_id, idx_start, idx_end, _ = re.split("\[|]|-", job_id)
            job_ids = [batch_id + str(idx) for idx in range(int(idx_start), int(idx_end) + 1)]
        elif "[" in job_id and "]" in job_id:
            job_ids = [job_id.replace("[", "").replace("]", "")]
        else:
            job_ids = [job_id]
        return job_ids

    def __update_all(self):
        if not self._tasks:
            tasks = self.__query_tasks(None)
            self._tasks = {r.slurm_id: r for r in tasks}
        else:
            tasks = self.__query_tasks(list(self._tasks.keys()))
        for r in tasks:
            self._tasks[r.slurm_id] = r

    def __update_subset(self, slurm_ids):
        tasks = self.__query_tasks(slurm_ids=slurm_ids)
        for r in tasks:
            self._tasks[r.slurm_id] = r


def make(mode, job_name, **kwargs) -> SchedulerClient:
    if mode == "local":
        return LocalSchedulerClient(job_name, **kwargs)
    elif mode == "slurm":
        return SlurmSchedulerClient(job_name)
    else:
        raise NotImplementedError(f"Scheduler {mode} not found")
