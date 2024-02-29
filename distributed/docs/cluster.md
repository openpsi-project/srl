# Cluster Information

## Login

Ask the administrators.

## Nodes

Nodes are named by their GPU resources. 

+ `frl1g[035-106]` : each with 256 AMD cores, 512 GB RAM, 1 NVIDIA GEFORCE 3090.
+ `frl2g[004-034]` : each with 256 AMD cores, 512 GB RAM, 2 NVIDIA GEFORCE 3090.
+ `frl8g[134-137]` : each with 128 AMD cores, 512 GB RAM, 8 NVIDIA GEFORCE 3090.
+ `frl8a[138-141]` : each with 128 AMD cores, 512 GB RAM, 8 NVIDIA TESLA A100.

[comment]: <> (+ `frlgpu[001-008]` &#40;`10.210.5.[34-41]`&#41;: GPU nodes with 128 AMD cores, 512 GB RAM, 8 NVIDIA 3090 each.)

[comment]: <> (+ `mu01` &#40;`10.210.5.250`&#41;: The NIS node, also serving the Slurm master, Prometheus master, and possibly REDIS. Do not)

[comment]: <> (  touch this server unless you know what you are doing.)

[comment]: <> (+ `frldisk002` &#40;`10.210.5.201`&#41;: The disk node serving NFS, including home.)

## Scheduler

We currently use [Slurm](https://slurm.schedmd.com/overview.html).

To view your tasks, run:
```
squeue  # See all.
squeue -u <USER>  # See only a user.
```

To cancel a task, run:
```
scancel <JOBID>
```

To cancel all your tasks, run:
```
scancel -u <USER>
```
