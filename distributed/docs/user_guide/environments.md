# Environment

Our environment API as the following features:

1. Agents in an environment can have different observation shape, action space, etc;
2. Environment can be [asynchronous](#support-for-asynchronous-environments),

and the following limitations:

1. Environments have to be homogeneous within an execution, the agent_count cannot change during an execution;
2. Observation space of each agent cannot change(There is no dynamic inference stream matching.) 

## `StepResult`
Environments returns the result of each `reset` and `step` to the system as `StepResult(dataclass)`.   

```python
@dataclasses.dataclass
class StepResult:
    """Step result for a single agent. In multi-agent scenario, env.step() essentially returns
    List[StepResult].
    """
    obs: Dict
    reward: np.ndarray
    done: np.ndarray
    info: Dict
    truncate: np.ndarray = np.zeros(shape=(1,), dtype=np.uint8)
```

In short, ```obs``` and ```info``` should be (nested) dictionary of numpy arrays. For example:

```python
import api.environment


class Env(api.environment.Environment):
    # Lets say this is a single agent environment.
    ...
    def step(self, actions: List[Action]) -> List[StepResult]:
        # Take action in the actual simulator
        obs, reward, done, truncated, info = self.__env.step(actions[0])
        # Let's say your observation is returned as two lists of float numbers, with the first representing
        # the global state, and the second representing the local observation of the agent.
        return api.environment.StepResult(obs={"global": np.array(obs[0]),
                                               "local": np.array(obs[1])},
                                          reward=np.array([reward]),
                                          done=np.array([0], dtype=np.uint8),
                                          truncate=np.array([0], dtype=np.uint8),
                                          info=None,
                                          )
```
Info is allowed to be None.

# Using legacy policy(Optional)

The native ```actor-critic``` implemented in [legacy/algorithm/ppo/actor_critic_policies](../../../legacy/algorithm/ppo/actor_critic_policies/actor_critic_policy.py)
is a great place to start if you do not have a customized policy implemented. (Read the [doc](../../../legacy/algorithm/ppo/actor_critic_policies/readme.md))

For the example above, if "global" observation is an array of length 32, "local" observation is of length 27 and the 
action space is discrete with 6 actions. 
You could use the following policy configuration in your experiment:

```python
import api.config


class MyExp(api.config.Experiment):
    def scheduling_setup(self) -> ExperimentScheduling:
        ...
    def initial_setup(self) -> ExperimentConfig:
        ...
        trainer = api.config.Trainer(type_="mappo")
        customized_policy = api.config.Policy(type_="actor-critic",
                                              args=dict(obs_dim={"global": 32, "local": 27},
                                                        action_dim=6))
        pw = api.config.PolicyWorker(
            ...,
            policy=customized_policy,
            ...
        )
        
        tw = api.config.TrainerWorker(
            ...,
            trainer=trainer,
            policy=customized_policy,
            ...
        )
        return api.config.ExperimentConfig(...)

api.config.register_experiment("demo-experiment", MyExp)
```


## Support for Asynchronous Environments

In some multi-agent environments, e.g. Hanabi, only part of the agents make action at each step.
In such cases, the list of `StepResult`s environment returns can have None value. 
If the `StepResult` of some agent is None, the system won't generate action for it, 
but passing a `None` as the action for this agent in the next step. 

For example, if your implemented environment returns:
```python
env.step(actions)
>>> [None, step_result1, None, step_result2]
```

Agents at index 0 and 2 will not run inference for this step, as a result, the in the next `step` call 
to your environment, agent 0 and 2 will not have actions. Intuitively, the next `step` will look like:
```python
env.step(actions = [None, action1, None, action2])
```

In this case, you will be responsible for handling "None actions".
