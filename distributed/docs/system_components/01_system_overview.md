# System Overview 

### Scheduler
- Controller and workers are scheduled via scheduler. 
- Currently, supported are 
  - Local scheduler. (For testing only. Does not assign GPU devices properly.)
  - Slurm Scheduler. (Checkout api/config.py -> Scheduling) (Recommended for frl cluster)

### **Controller**

Controller functions like a handle to manage experiments.

1. Load experiment configuration.
2. Schedule workers.
3. Configure workers.
4. Stop the experiment when appropriate.

For detailed documentation, checkout [controller documentation](02_controller.md).

### Workers

- **worker_base.py**

  Base class of all workers. Common handles are implemented in class worker_base.Worker are:

  - configure
  - start
  - pause
  - stop

  Unless interrupted, workers will run a dead loop of  ```_poll()```, which is to be implemented by different Workers.


##### Threading
- Policy Worker has a main thread(cpu), an inference thread, and a responding thread(cpu).
- Trainer Worker has a main thread(cpu, where buffer resides), and a training thread(gpu).

### Inference Stream 

- Inference stream is implemented in ZMQ (https://zeromq.org/)
  - InferenceClient is responsible for gathering/batching requests from Actors and receiving responses from inference clients.
  - InferenceServer is responsible for receiving requests and distributing inference results.
  - To support synchronized request-reply, Inference Stream is implemented in DEALER-ROUTER pattern.
- Supported
  - IpInferenceStream(Client & Server)
  - NameResolvingInferenceStream(Client & Server) (Recommended)
  - InlineInferenceClient(Client)

### Sample Stream
- Sample Stream is implemented in ZMQ
  - PUSH-PULL pattern
  - Sample are batches two times before consuming
    - Upon sending, actors will batch the sample along the _time_ dimension
    - Upon consuming, buffer will batch the sample along the _batch_ dimension.
- Supported
  - Ip(Producer & Consumer)
  - NameResolving(Producer & Consumer) (Recommended)
  - RoundRobinNameResolving(Producer) will send data to consumers in round-robin manner.
  - BroadcastingNameResolving(Producer) will broadcast data to all consumers.
  - MultiAgentNameResolving(Producer) will batch all agents before sending. Unsafe when ring-size > 1.

### Parameter Database
- Parameter Database stores model parameters, and workers can push/get parameters via ParameterDBClient.
  - ParameterDB is currently implemented in NFS.
  - Metadata query is supported via MongoDB.
- Supported
  - FileSystemParameterDB
  - MetadataParameterDB

### NameResolve
- Workers exchange system level metadata via NameResolve, including addresses, ports, peers, etc.
- NameResolve is currently used in the following ways,
  - Workers save their listen port to name resolve for controller.
  - InferenceServer reveal their inbound address for clients.
  - SampleConsumer reveal their inbound address for producers.
  - Trainers reveal their identity so that they can find DDP peers.
- Supported
  - MemoryNameResolve(Testing Only. No inter-node communication.)
  - NFSNameResolve
  - RedisNameResolve(Recommended)