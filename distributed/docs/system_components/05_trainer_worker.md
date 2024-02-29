# Trainer Worker

_"Tell me and I forget, teach me and I may remember, involve me and I learn."_

### Initialization(Configuration)
The trainer worker requires the following fields to be configured.

1. trainer
2. policy
3. policy_name
4. a sample-stream.

Read [Policy Worker Initialization](04_policy_worker.md#initializationconfiguration) for a discussion on policy_name.

### Worker Logic
To optimize the usage of GPU, trainer worker runs two threads.
1. cpu-thread(main-thread): receiving and batching sample, push parameters, logging training data.
2. gpu-thread: gradient computation and parameter update.

### DDP related
##### Peer discovery
In a trial, the trainer workers with the same policy_name are considered as peers. Their gradients are synced through
[pytorch DistributedDataParallel framework](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html). 
- Upon configuration, the trainer write their policy name and worker index to system [name_resolving service](01_system_overview.md#nameresolve).
- As the workers start, each trainer discover their peers and initialize the distributed processes. Ranks are determined by the order of worker-index.

This process guarantees that all peers can be discovered.

##### Training
Syncing of gradients are done automatically. 

##### Stopping
As ddp reducing operation blocks all peers. Some peers may be waiting for others to sync gradients while others
workers have already exited. To prevent this from happening, upon receiving a stop signal, we determine a step count,
larger than the current step, as where peers should stop running.