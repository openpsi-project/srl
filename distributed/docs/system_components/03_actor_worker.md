# Actor Worker

_"Reality is merely a simulation of God who himself is the simulation of humanity emulated through a brief history of time."_

### Initialization(Configuration)
To configure an actor worker, one has to specify an _Environment_, _Inference Streams_, _Sample Streams_, and _AgentSpecs_.

Initialization process is:
1. Make inference clients and sample producers. The actor worker keeps a reference to these streams.
2. Make agents as specified by AgentSpec. Each agent is matched with a inference_client and an sample producer.
3. Create many environment targets with the specified _Environment_ and agents created in 2.

Upon reset of any environment target, the actor worker will flush the INLINE inference clients and MA sample_producers. 

<hr />

## EnvironmentRing(`_EnvRing`)
To optimize the usage of cpu, while waiting for the rollout response from a `inference_stream`, the actor worker may run 
simulation on another simulator. This requires an actor worker to hold multiple simulators. As the number of simulators 
needed to balance the usage between cpu/gpu, we propose a flexible `ring` data structure..

### Initialization
- Environment ring is composed of several identical _simulators_.
- All simulators together makes a (imaginary) circle, with the _Ring_'s `head` pointing to one of them.

### rotate
- If the `head` steps successfully, the ring `rotates` to the next simulator.

## EnvironmentTarget(`_EnvTarget`)
An environment target hold a simulator and does the following.
1. Distribute simulator results to agents.
2. Get new actions form the agents.
3. reset the simulators when appropriate.

## Agent
An Agent 
1. receives observations/ rewards from its environment target.
2. Append the new observation to its memory.
3. request for new action.
4. pass action to its environment target.
5. send sample when appropriate.

Refer to the initialization docstring of Agents or [AgentSpec configuration](../user_guide/config_your_experiment.md#AgentSpec) on how to config an Agent.

# What's Next
- [Policy Worker](04_policy_worker.md)
- [Trainer Worker](05_trainer_worker.md)
