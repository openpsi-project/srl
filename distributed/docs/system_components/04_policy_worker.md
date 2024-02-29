# Policy Worker

_"The best policy is to declare victory and leave."_

### Initialization(Configuration)

To configure a policy worker, the following must be specified.
1. policy.
2. policy name.
3. policy identifier.
4. an inference-stream.

Note that _policy name_ is an identifier for workers to communicate, which differs from `policy.type_`. 
Such that if you are training two policies named "alice" and "bob" in the same experiment, 
they won't interfere with each other. You are free to choose your policy_name(s) in you experiment.
In many classic RL task, there is only one _policy name_ throughout the experiment.
In some external research works, _policy name_ corresponds to an `agent`.

On the other hand, `policy.type_` is a name-resolving registry, where user can choose form the implemented policies.

### Worker Logic
To optimize efficiency, policy worker runs multi-threadedly. 
1. Main-thread: receive rollout-requests, batch rollout-requests, and handle worker-level request.
2. Rollout-thread: runs policy.rollout on batched rollout-requests
3. Respond-thread: unpack the batched rollout responses and send to inference clients.

### Auto-batching
To minimize the inference latency, the main thread dynamically decides the batch-size of rollout.
The main-thread pushes data to rollout-thread through a queue of size 1. While the queue is full, 
the main thread accumulate requests.
Once the queue is cleared, the main-thread batches all the pending requests and put to the queue.

Read [this instruction](../user_guide/algorithms.md) on how to implement your own policy.

