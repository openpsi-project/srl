from typing import Optional, Callable, Literal, List, Dict, Tuple
import argparse
import dataclasses
import enum
import getpass
import subprocess
import logging
import random

logger = logging.getLogger("Slurm Utils")


class ResourceNotDivisible(Exception):
    pass


@dataclasses.dataclass
class SlurmResource:
    # a data class that represents a slurm resource quota
    mem: int = 0
    cpu: int = 0
    gpu_type: str = "geforce"
    gpu: int = 0

    def __mul__(self, other):
        if isinstance(other, int):
            return SlurmResource(mem=self.mem * other,
                                 cpu=self.cpu * other,
                                 gpu=self.gpu * other,
                                 gpu_type=self.gpu_type)
        else:
            raise TypeError("ResourceRequirement can only be multiplied by int.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, SlurmResource):
            # adding two resources with different gpu types
            new_gpu_type = self.gpu_type
            if self.gpu_type != other.gpu_type:
                # when two gpu types are different, use the gpu type whose corresponding gpu count is non-zero
                # if both non-zero, raise error
                new_gpu_type = self.gpu_type if other.gpu_type == 0 else other.gpu_type
                if self.gpu and other.gpu:
                    raise ValueError("Cannot add two different gpu types.")
            return SlurmResource(mem=self.mem + other.mem,
                                 cpu=self.cpu + other.cpu,
                                 gpu=self.gpu + other.gpu,
                                 gpu_type=new_gpu_type)
        else:
            raise TypeError("ResourceRequirement can only add another ResourceRequirement instance.")

    def __sub__(self, other):
        if isinstance(other, SlurmResource):
            if self.gpu_type != other.gpu_type:
                new_gpu_type = self.gpu_type if other.gpu == 0 else other.gpu_type
                if self.gpu > 0 and other.gpu > 0:
                    raise ValueError("Cannot subtract two different gpu types.")
            else:
                new_gpu_type = self.gpu_type

            return SlurmResource(mem=self.mem - other.mem,
                                 cpu=self.cpu - other.cpu,
                                 gpu=self.gpu - other.gpu,
                                 gpu_type=new_gpu_type)
        else:
            raise TypeError("ResourceRequirement can only subtract another ResourceRequirement instance.")

    def __floordiv__(self, number: int):
        if not (self.mem % number == 0 and self.cpu % number == 0 and self.gpu % number == 0):
            raise ResourceNotDivisible()
        return SlurmResource(mem=self.mem // number,
                             cpu=self.cpu // number,
                             gpu=self.gpu // number,
                             gpu_type=self.gpu_type)

    def __neg__(self):
        return SlurmResource(mem=-self.mem, cpu=-self.cpu, gpu=-self.gpu, gpu_type=self.gpu_type)

    def valid(self):
        # check if it is a valid resource requirement
        if self.gpu_type not in ["geforce", "tesla"]:
            return False
        if self.mem < 0 or self.cpu < 0 or self.gpu < 0:
            return False
        return True


@dataclasses.dataclass
class SlurmTaskSpecification:
    # contain all informations required for a slurm task
    job_name: str
    task_name: str
    ntasks: int
    resource_requirement: SlurmResource
    cmd: str
    container_image: str
    container_mounts: str
    env_vars: dict
    nodelist: str
    exclude: str
    hostfile: bool
    group_size: int
    node_type: Optional[List[str]] = None


def _parse_output_status_line(status):
    assert status.startswith("State=")
    status = status.split(" ")[0]
    status = status.split("=")[1]
    return status.split("+")


def _parse_output_tres_line(tres):
    tres = tres.split("=", maxsplit=1)[1]
    tres = tres.split(",")
    res = SlurmResource()
    if len(tres) == 0:
        return SlurmResource()
    for t in tres:
        if t.startswith("mem"):
            if t.endswith("M"):
                res.mem = int(t.split("=")[1].strip("M"))
            elif t.endswith("G"):
                res.mem = int(float(t.split("=")[1].strip("G")) * 1024)
            else:
                raise ValueError("Unknown memory unit.")
        elif t.startswith("cpu"):
            res.cpu = int(t.split("=")[1])
        elif t.startswith("gres/gpu"):
            prefix, sgpu = t.split("=")
            if ":" in prefix:
                res.gpu_type = prefix.split(":")[1]
            else:
                res.gpu = int(sgpu)
    return res


def _parse_nodelist(nodelist: str) -> List[str]:
    return subprocess.check_output([
        'scontrol',
        'show',
        'hostnames',
        nodelist,
    ]).decode('utf-8').strip().split('\n')


def get_slurm_node_resources(
    node_type: Optional[List[str]] = None,
    nodelist: Optional[str] = None,
    exclude: Optional[str] = None,
) -> Dict[str, SlurmResource]:
    all_hostnames: List[str] = _parse_nodelist(
        subprocess.check_output("sinfo -o \"%N\" --noheader", shell=True).decode("utf-8").strip())

    if nodelist is not None:
        valid_hostnames: List[str] = _parse_nodelist(nodelist)
    else:
        valid_hostnames = all_hostnames

    if exclude is not None:
        excluded_hostnames: List[str] = subprocess.check_output([
            'scontrol',
            'show',
            'hostnames',
            exclude,
        ]).decode('utf-8').strip().split('\n')
        for hn in excluded_hostnames:
            if hn in valid_hostnames:
                valid_hostnames.remove(hn)

    for hn in valid_hostnames:
        if hn not in all_hostnames:
            raise ValueError(f"Invalid host name: {hn}. Available host names: {all_hostnames}.")

    def _filter_node_type(node_type, node_name):
        if node_type is not None:
            if not isinstance(node_type, list):
                node_type = [node_type]
            nt_condition = []
            for nt in node_type:
                if nt == 'g1' and 'frl1g' not in node_name:
                    cond = False
                elif nt == 'g2' and 'frl2g' not in node_name:
                    cond = False
                elif nt == 'g8' and 'frl8g' not in node_name:
                    cond = False
                elif nt == 'a100' and 'frl8a' not in node_name and 'frl4a' not in node_name:
                    cond = False
                elif nt == 'a800' and "YL-com" not in node_name:
                    cond = False
                elif nt not in ['g1', 'g2', 'g8', 'a100', 'a800']:
                    raise ValueError("Unknown node type.")
                else:
                    cond = True
                nt_condition.append(cond)
            return any(nt_condition)
        else:
            return True

    valid_hostnames = list(filter(lambda x: _filter_node_type(node_type, x), valid_hostnames))

    if len(valid_hostnames) == 0:
        raise ValueError("No valid host name.")

    # execute `scontrol show node` to get node resources
    # return a list of SlurmResource
    o = subprocess.check_output(["scontrol", "show", "node"]).decode("utf-8")
    nodes = o.split("\n\n")
    all_rres = {}
    for node in nodes:
        if len(node) <= 1:
            continue
        ls = node.split("\n")
        node_name = ls[0].split(" ")[0].split("=")[1]
        if node_name not in valid_hostnames:
            continue
        ctres = SlurmResource()
        atres = SlurmResource()
        for l in ls:
            l = l.strip("\n").strip()
            if l.startswith("State"):
                status = _parse_output_status_line(l)
                if "DOWN" in status or "DRAIN" in status or "NOT_RESPONDING" in status:
                    break
            if l.startswith("CfgTRES"):
                ctres = _parse_output_tres_line(l)
            if l.startswith("AllocTRES"):
                atres = _parse_output_tres_line(l)
        if "8a" in node_name or "4a" in node_name or "YL-com" in node_name:
            ctres.gpu_type = atres.gpu_type = "tesla"
        else:
            ctres.gpu_type = atres.gpu_type = "geforce"
        rres = ctres - atres
        if rres.valid():
            all_rres[node_name] = rres
        else:
            all_rres[node_name] = SlurmResource(gpu_type=ctres.gpu_type)

    return all_rres


def allocate_to(
    res: SlurmResource,
    num_tasks: int,
    arres: List[Tuple[str, SlurmResource]],
) -> Tuple[int, Dict[str, int]]:
    # only support homogeneous tasks now
    # may cause conflict when other are allocating jobs concurrently
    # greedy allocation, search for the node that has the most gpu
    n = num_tasks
    # print(arres)
    allocated = {}
    for k, v in arres:
        task_count = 0
        while n > 0:
            try:
                v = v - res
            except ValueError as e:  # this indicates different GPU types
                # print(e)
                break
            if not v.valid():
                # print(v)
                break
            task_count += 1
            n -= 1
        allocated[k] = task_count
        # print(k, v, task_count)
    allocated = {k: v for k, v in allocated.items() if v > 0}
    return n, allocated


def _time_str_to_seconds(time_str: str) -> int:
    if len(time_str.split('-')) == 2:
        day, time_str = time_str.split('-')
    else:
        day = 0
    if len(time_str.split(':')) == 3:
        hour, minute, second = time_str.split(':')
    elif len(time_str.split(':')) == 2:
        hour = 0
        minute, second = time_str.split(':')
    else:
        raise ValueError()
    return ((int(day) * 24 + int(hour)) * 60 + int(minute)) * 60 + int(second)


def _get_slurm_job_specs(
) -> Tuple[List[Tuple[str, str, int, List[str]]], List[Tuple[str, str, int, List[str]]]]:
    # job_id, job_name, running_time, list of hostnames
    squeue_lines = subprocess.check_output(
        f"squeue -u {getpass.getuser()} -o \"%.18i %.80j %.10M %.6D %R\" -h", shell=True, text=True).strip()
    squeue_lines_ = subprocess.check_output(f"squeue -u {getpass.getuser()} -o \"%.2t\" -h",
                                            shell=True,
                                            text=True).strip()
    all_status = [line.strip() for line in squeue_lines_.split('\n')]
    running_job_sepcs, pending_job_specs = [], []
    for status, line in zip(all_status, squeue_lines.split("\n")):
        if status == 'R':
            job_id, job_name, running_time, num_nodes, nodelist = line.strip().split()
            running_time = _time_str_to_seconds(running_time)
            nodelist = _parse_nodelist(nodelist)
            assert len(nodelist) == int(num_nodes), (num_nodes, nodelist)
            running_job_sepcs.append((job_id, job_name, running_time, nodelist))
        # elif status == 'PD':
        #     pending_job_specs.append((job_id, job_name, running_time, nodelist))
    return running_job_sepcs, pending_job_specs


def _get_job_resource(job_id: str) -> SlurmResource:
    job_specs = subprocess.check_output(f"scontrol show job {job_id}", shell=True, text=True).strip()
    r = SlurmResource()
    for line in job_specs.split('\n'):
        line = line.strip()
        if not line.startswith("TRES="):
            continue
        resources = line[5:].split(',')
        for res in resources:
            if res.startswith('cpu'):
                r.cpu = int(res.split('=')[1])
            if res.startswith('gres/gpu:'):
                gpu_type, gpu = res.split(':')[1].split('=')
                r.gpu_type = gpu_type
                r.gpu = int(gpu)
            if res.startswith('gres/gpu='):
                r.gpu = int(res.split('=')[1])
            if res.startswith('mem='):
                mem = res.split('=')[1]
                if mem.endswith("M"):
                    r.mem = int(mem.strip("M"))
                elif mem.endswith("G"):
                    r.mem = int(float(mem.strip("G")) * 1024)
                else:
                    raise ValueError("Unknown memory unit.")
    return r


def write_hostfile(
    res: SlurmResource,
    num_tasks: int,
    hostfile_dir: str,
    node_type: Optional[List[str]] = None,
    nodelist: Optional[str] = None,
    exclude: Optional[str] = None,
):
    # only support homogeneous tasks now
    # NOTE: may cause conflict when other are allocating jobs concurrently
    arres = get_slurm_node_resources(node_type, nodelist, exclude)
    # greedy allocation, search for the node that has the most gpu
    sorted_arres = sorted([(k, v) for k, v in arres.items()],
                          key=lambda x: (x[1].gpu, x[1].cpu, x[1].mem),
                          reverse=True)
    remaining, allocated = allocate_to(res, num_tasks, sorted_arres)
    if remaining > 0:
        logger.warning("Not enough resources for allocation. "
                       "Allocate to nodes that have been occupied. "
                       "This job will be pending.")
        running_job_specs, _ = _get_slurm_job_specs()
        # Find nodes allocated by the user's running jobs. Regard them as free nodes.
        # Then re-allocate the resources.
        for job_id, *_, host_names in running_job_specs:
            r = _get_job_resource(job_id)
            try:
                r = r // len(host_names)
            except ResourceNotDivisible:
                continue
            for hn in host_names:
                if hn not in arres:
                    continue
                if '8a' in hn or 'YL-com' in hn:
                    r.gpu_type = "tesla"
                else:
                    r.gpu_type = "geforce"
                arres[hn] = arres[hn] + r
        arres_lis = sorted([(k, v) for k, v in arres.items()],
                           key=lambda x: (x[1].gpu, x[1].cpu, x[1].mem),
                           reverse=True)

        # print(">>>>>>>>>>>>>>>>>>>>>>")
        # for k, v in arres_lis:
        #     print(k, v)
        # print(sum([x[1] for x in arres_lis], start=SlurmResource()))

        random.shuffle(arres_lis)
        remaining, allocated = allocate_to(res, num_tasks, arres_lis)
        if remaining > 0:
            raise ValueError("Not enough resources.")

    with open(hostfile_dir, "w") as f:
        for k, v in allocated.items():
            for _ in range(v):
                f.write(f"{k}\n")
    return allocated


def show_resource(node_type: Optional[List], nodelist: str, exclude: str):
    all_rres = get_slurm_node_resources(node_type, nodelist=nodelist, exclude=exclude)
    for k, v in all_rres.items():
        print(k, v)
    print("In total: ", sum([x[1] for x in all_rres.items()], start=SlurmResource()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_type", nargs='+')
    parser.add_argument("--nodelist", type=str, default=None)
    parser.add_argument("--exclude", type=str, default=None)
    args = parser.parse_args()
    show_resource(args.node_type, args.nodelist, args.exclude)
