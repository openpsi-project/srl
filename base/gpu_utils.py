from typing import List
import itertools
import logging
import os
import platform
import socket
import time

logger = logging.getLogger("System-GPU")


def gpu_count():
    """Returns the number of gpus on a node. Ad-hoc to frl cluster.
    """
    if platform.system() == "Darwin":
        return 0
    elif platform.system() == "Windows":
        try:
            import torch
            return torch.cuda.device_count()
        except ImportError:
            return 0
    else:
        dev_directories = list(os.listdir("/dev/"))
        for cnt in itertools.count():
            if "nvidia" + str(cnt) in dev_directories:
                continue
            else:
                break
        return cnt


# def resolve_cuda_environment():
#     """Pytorch DDP does not work if more than one processes (with different environment variable CUDA_VISIBLE_DEVICES)
#      are inited on the same node(w/ multiple GPUS). This function works around the issue by setting another variable.
#      Currently all devices should use `base.gpu_utils.get_gpu_device()` to get the proper gpu device.
#     """
#     if "MARL_CUDA_DEVICES" in os.environ.keys():
#         return

#     cuda_devices = [str(i) for i in range(gpu_count())]
#     if "CUDA_VISIBLE_DEVICES" not in os.environ:
#         if len(cuda_devices) > 0:
#             os.environ["MARL_CUDA_DEVICES"] = "0"
#         else:
#             os.environ["MARL_CUDA_DEVICES"] = "cpu"
#     else:
#         if os.environ["CUDA_VISIBLE_DEVICES"] != "":
#             for s in os.environ["CUDA_VISIBLE_DEVICES"].split(","):
#                 assert s.isdigit() and s in cuda_devices, f"Cuda device {s} cannot be resolved."
#             os.environ["MARL_CUDA_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]  # Store assigned device.
#         else:
#             os.environ["MARL_CUDA_DEVICES"] = "cpu"  # Use CPU if no cuda device available.
#     os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_devices)  # Make all devices visible.


def isolate_cuda_device(worker_type, experiment_name, trial_name):
    import distributed.base.name_resolve as name_resolve
    import base.names as names
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        return

    # HACK: this only works for slurm mode
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NPROCS'])

    name_resolve_identifier = f"__type_{worker_type}"
    name_resolve.add_subentry(
        names.trainer_ddp_local_peer(experiment_name, trial_name, socket.gethostname(),
                                     name_resolve_identifier),
        rank,
        keepalive_ttl=90,
    )
    name_resolve.add_subentry(
        names.trainer_ddp_peer(experiment_name, trial_name, name_resolve_identifier),
        rank,
        keepalive_ttl=90,
    )
    logger.info(f"Rank {rank} waiting for peers, world size {world_size}...")
    while len(
            name_resolve.get_subtree(
                names.trainer_ddp_peer(experiment_name, trial_name, name_resolve_identifier))) < world_size:
        time.sleep(0.1)
    logger.info(f"Rank {rank} discovers all peers, resolving local rank...")
    local_peer_name = names.trainer_ddp_local_peer(
        experiment_name,
        trial_name,
        socket.gethostname(),
        name_resolve_identifier,
    )
    local_peers = list([str(x) for x in sorted([int(x) for x in name_resolve.get_subtree(local_peer_name)])])
    logger.info(f"Rank {rank} discovers local peers with global ranks {local_peers}")

    local_peer_index = local_peers.index(str(rank))
    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == len(local_peers):
        local_gpu_id = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))[local_peer_index]
    elif len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1:
        local_gpu_id = int(os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        raise RuntimeError(f"Invalid CUDA_VISIBLE_DEVICES {os.environ['CUDA_VISIBLE_DEVICES']}")

    logger.info(f"Worker type {worker_type} rank {rank} running on host {socket.gethostname()}, "
                f"local peer index: {local_peer_index}, local gpu id {local_gpu_id}.")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_gpu_id)


def get_gpu_device() -> List[str]:
    """
    Returns:
        List of assigned devices.
    """
    # assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1, os.environ['CUDA_VISIBLE_DEVICES']
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        return ['cpu']
    else:
        return [f'cuda:0']
    # if "MARL_CUDA_DEVICES" not in os.environ:
    #     resolve_cuda_environment()

    # if os.environ["MARL_CUDA_DEVICES"] == "cpu":
    #     return ["cpu"]
    # else:
    #     return [f"cuda:{device}" for device in os.environ["MARL_CUDA_DEVICES"].split(",")]


def set_cuda_device(device):
    """Set the default cuda-device. Useful on multi-gpu nodes. Should be called in every gpu-thread.
    """
    logger.info(f"Setting device to {device}.")
    if device != "cpu":
        import torch
        torch.cuda.set_device(device)


# resolve_cuda_environment()
