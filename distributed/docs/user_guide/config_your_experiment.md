# Configuring Your Experiment

## Table of Contents:
- [Experiment configuration as a Python class](#Configuration)
- [Inference Stream](#inference-stream)
- [Sample Stream](#sample-stream)
- [AgentSpec](#agentspec)
- [Version Control](#version-control)
- [Metadata Parameter Database](#metadata-parameter-database)
- [Population-based Training](#population-based-training-pbt)
- [wandb configuration](#wandb-configuration)

<hr />

## Experiment configuration as a Python class

Currently, we only support configuration file as python class. 
Compared to the widely-adopted yaml-styled configuration, python has many advantages:

- Most RL users are familiar with python.
- It is easier to make complex configuration. For-loops, conditions, and even RNG can be helpful.
- With IDE, it is less likely to make configuration mistakes.

Some drawbacks are:

- It requires effort save configurations elegantly. We currently save them as python scripts.
- Sending it through the network requires serialization. We currently do it with pickle(unsafe).

An experiment configuration comes in two parts:

1. A scheduling config: Tells the scheduler how may workers of different types are requires, and where to schedule them.
2. Configuration of all workers.

Thus, our base class for experiment config looks like:

```python
# api/config.py
class Experiment:
    """Base class for defining the procedure of an experiment.
    """

    def scheduling_setup(self) -> ExperimentScheduling:
        """Returns the Scheduling of all workers."""
        raise NotImplementedError()

    def initial_setup(self) -> ExperimentConfig:
        """Returns a list of workers to create when a trial of the experiment is initialized."""
        raise NotImplementedError()
```

Now lets start with a simplest example: an Atari Experiment.

#### 1. SchedulingSetup
Let's say we want 
- 16 cpus to run the simulation, (16 actor workers, as we name it)
- 2 GPUs to do model inference, each with 4 models running in parallel(8 policy workers), and
- 1 GPU to do optimization (1 trainer worker).

A TaskGroup is a scheduling concept. It specifies a group of workers, e.g.
```python
TasksGroup(count=16, 
           scheduling=Scheduling(cpu=1, mem=1024)
           )
```
will submit 16 workers, each with 1 cpu and 1G of memory.

Putting everything in place, The scheduling for our experiment will be:

```python
from api.config import ExperimentScheduling, TasksGroup, Scheduling


scheduling = ExperimentScheduling(
            actors=TasksGroup(count=16, 
                              scheduling=Scheduling.actor_worker_default(cpu=1, mem=1024)),
            policies=TasksGroup(count=8,
                                scheduling=Scheduling.policy_worker_default(gpu=0.25, mem=1024 * 10)),
            trainers=TasksGroup(count=1,
                                scheduling=Scheduling.trainer_worker_default(cpu=4, mem=1024 * 80)),
        )
```

Not that with `gpus=0.25`, policy workers will share gpus resource up to 4pw/GPU. The default values
for different worker types can be found in `api/config.py`.

#### 2. ExperimentConfig

An Experiment Config describes the configuration of all workers.
```python
# ----------- some handy variables -------------
policy_name = "default"
policy = Policy(type_="atari_naive_rnn", args=dict(action_space=18,))
# ----------- some handy variables -------------


experiment = ExperimentConfig(
    actors=[
        ActorWorker(env=Environment(type_="atari",
                                    args=dict(game_name="PongNoFrameskip-v4")),
                    inference_streams=[policy_name],
                    sample_streams=[policy_name],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                        )
                    ]) for _ in range(16)
    ],
    policies=[
        PolicyWorker(
            policy_name=policy_name,
            inference_stream=policy_name,
            policy=policy,
        ) for _ in range(8)
    ],
    trainers=[
        TrainerWorker(
            policy_name=policy_name,
            trainer="mappo",
            policy=policy,
            sample_stream=policy_name,
        ) for _ in range(1)
    ],
)
```
For actor workers, we have configured them to run environment:`"atari"`. 
And we also specified the policy to be `"atari_naive_rnn"` for policy and trainer workers.
Workers are connected through [inference stream](#inference-stream) and [sample stream](#sample-stream).

#### 3. Putting together

Finally we put everything together into an Experiment:

```python
from api.config import *


scheduling = ExperimentScheduling(
            actors=TasksGroup(count=16, 
                              scheduling=Scheduling.actor_worker_default(cpu=1, mem=1024)),
            policies=TasksGroup(count=8,
                                scheduling=Scheduling.policy_worker_default(gpu=0.25, mem=1024 * 10)),
            trainers=TasksGroup(count=1,
                                scheduling=Scheduling.trainer_worker_default(cpu=4, mem=1024 * 80)),
        )


# ----------- some handy variables -------------
policy_name = "default"
policy = Policy(type_="atari_naive_rnn", args=dict(action_space=18,))
# ----------- some handy variables -------------


experiment = ExperimentConfig(
    actors=[
        ActorWorker(env=Environment(type_="atari",
                                    args=dict(game_name="PongNoFrameskip-v4")),
                    inference_streams=[policy_name],
                    sample_streams=[policy_name],
                    agent_specs=[
                        AgentSpec(
                            index_regex=".*",
                            inference_stream_idx=0,
                            sample_stream_idx=0,
                        )
                    ]) for _ in range(16)
    ],
    policies=[
        PolicyWorker(
            policy_name=policy_name,
            inference_stream=policy_name,
            policy=policy,
        ) for _ in range(8)
    ],
    trainers=[
        TrainerWorker(
            policy_name=policy_name,
            trainer="mappo",
            policy=policy,
            sample_stream=policy_name,
        ) for _ in range(1)
    ],
)


class AtariExperiment(Experiment):

    def scheduling_setup(self):
        return scheduling

    def initial_setup(self):
        return experiment


register_experiment("example-atari", AtariExperiment)
```

And registered it as `"example-atari"`.

***NOTE: The Experiment above is for pure demonstration purpoes. For running examples, 
we direct you to legacy/experiments.*** 


<hr />

## Inference Stream
Inference Streams are for sending rollout requests and receive rollout response.
Choose one that best describes your needs from the following:

- *[Remote Inference](#remote-inference)*: If you need to speed up rollout, do not care about the consistency
of policy version.
- *[Inline Inference](#inline-inference)*: If you need to control the policy version,
for example, if you require a consistent policy version for each episode.

Typically, remote inference is used for training, while inline for evaluation/ past policy sampling.

### Remote inference
For some neural network models, GPU-inference may be much faster. However, hosting neural networks
on every actor worker is infeasible due to limited GPU memory. 

To set up a remote inference stream 

1. In policy worker configurations, specify ```inference_stream="alice"```.
2. In actor worker configurations, [match your agent](#matching-agents-to-streams) to this stream.

Note that 

```python
inference_stream="alice"
``` 
is equivalent to

```python
from api.config import InferenceStream

inferece_stream = InferenceStream(type_=InferenceStream.Type.NAME,
                                  stream_name="alice")
```

### Inline inference
Policy(and its neural network) is hosted locally in inline inference stream. Therefore, inline inference stream 
has the client side, but not the server side.

A typical configuration of inline inference stream is as follows.

```python
from api.config import InferenceStream

eval_inference_stream = InferenceStream(type_=InferenceStream.Type.INLINE,
                                        stream_name="",  # this field required but has no effect.
                                        policy=some_policy,
                                        policy_name="some_policy_name",
                                        policy_identifier="evaluation",  # or mongo_db_query such as {"$match": ..}
                                        pull_interval_seconds=None)
```
By setting `pull_interval_seconds=None`, the inline inference stream won't pull parameters by itself. The parameter
will be updated by the actor worker when any of the environment reset. In other words, if your actor worker has 
`ring_size > 1`, the consistency of policy version within each episode cannot be guaranteed. See [here](#version-control)
for more discussions.

<hr />

## Sample Stream

Sample stream is analogous to a producer-consumer model. In a typical use case, the producer side are 
actors and the consumer side may be trainers or eval_manager. In some cases, we need buffer workers 
in between actors and trainers.

The system currently support the following sample streams.
- **simple name-resolving**: send samples to one of the consumer. Currently, workers determine the target consumer 
by their worker index. [`SampleStream.Type.NAME`]
- **round-robin**: send samples to all consumers in round-robin manner. [`SampleStream.Type.NAME_ROUND_ROBIN`]
- **multi-agent**: [_use only in actor workers._] the stream will batch trajectories form different agents before sending.
***Sample has to be unpacked by the consumer to get agent-wise trajectory.*** [`SampleStream.Type.NAME_MULTI_AGENT`]

<hr />

## AgentSpec
### Matching Agents to streams

AgentSpec is a configuration field of Actor Worker. It describes how each agent performs rollout and collect samples.

Firstly, note that the streams(inference/sample) come in lists in the configuration of actor workers. For example:

```python
from legacy.experiments import ActorWorker, AgentSpec

cfg = ActorWorker(inference_streams=["inf1", "inf2"],
                  sample_streams=["train", "eval"],
                  ...
```

So far, it is yet to decide which agents go to which streams. 

The example below matches all agents to inference 
stream `inf1` and sample stream `train`.
```python
                  ...
                  # Example 1
                  agent_specs=[AgentSpec(index_regex=f".*",
                                         inference_stream_idx=0,
                                         sample_stream_idx=0)],
                  ...
```
`index_regex` is a regular expression pattern, which will be matched with agent index(0, 1, ..., converted to string). 
`.*` will match everything. 
For more on regular expression, see [regular expression(python)](https://docs.python.org/3/library/re.html).


Note that the AgentSpec comes in a list. We can also match different agents to different Streams. The following 
example matches agents 0, 1, 2, 3 to `inf1` and `train`, and agents 4, 5, 6 to `inf2` and `eval`. 
```python
                  ...
                  # Example 2
                  agent_specs=[AgentSpec(index_regex="[0-3]",
                                         inference_stream_idx=0,
                                         sample_stream_idx=0),
                               AgentSpec(index_regex="[3-6]",
                                         inference_stream_idx=1,
                                         sample_stream_idx=1)
                               ],
                  ...
```
Although agent `3` is matched by both pattern, the latter won't overwrite the former. In other words, the system will 
respect the first matched AgentSpec.

### Additional specifications
- **deterministic_action**: whether action is selected deterministically(argmax),
    as oppose to stochastically(sample).
- **sample_steps**: length of the sample to be sent. Effective only if send_full_trajectory=False.
- **bootstrap_steps**: number of additional steps appended to each sent sample. Temporal-difference style value
    tracing benefits from bootstrapping.
- **send_after_done**: whether to send sample only if the episode is done.
- **send_full_trajectory**: send full trajectories instead of sample of fixed length. Mostly used when episodes
    are of fixed length, or when agent is running evaluation.
- **pad_trajectory**: pad the full trajectories to fixed length. Useful when full trajectories are needed but
    length of the environment episodes varies. Effective only when send_full_trajectory=True.
- **send_concise_info**: If True, each episode is contracted in one time step, with the first observation and the
    last episode-info.


# Version Control
If you want fixed parameter version for each trajectory sent to an SampleStream, follow the steps:
1. Set the InferenceStream type to Inline, specify `policy, policy_name, policy_identifier`
and set `pull_interval_seconds` to None.
2. Set the ring_size of the actor worker to be 1.

Read below for details.
- Parameter of remote inference stream(i.e. those connected to policy workers.) are controlled by Policy Workers. 
As policy workers usually serves many actor workers, possibly with ring_size > 1, they _cannot_ change the parameter
version based on the progress of any single environment. Instead, policy workers update parameter based on a frequency
sepcified in their configuration. As a result, the SampleBatch generated by a remote inference 
stream are highly likely to have varying parameter version.
- In some cases(e.g. evaluation, metadata updating, etc), a fixed sample version for each trajectory is beneficial. 
In this case, we should use Inline Inference Stream, which holds the neural network locally on cpu device. In addition,
We should set the `pull_interval_seconds` parameter of the inference stream to None, so that the `load_parameter`
function is only called by actor, upon reset. 
- Note that we can use a mixture of `inline` and `remote` inference streams in one Actor Worker. The parameter version
of the sample batch of different agents will follow different patterns. (stated above)
- If `ring_size > 1` OR `pull_interval_seconds is not None`, again, the samples generated by InlineInferenceStream will be varying. 

### Evaluation
Users may use eval_manager to log evaluation results. To config your experiment to run evaluation, follow the steps:
1. Add ActorWorker with configuration ```is_evaluation=True, send_full_trajectory=True```
2. Add EvalManager and specify a tag in its configuration. e.g. ```tag="my_evaluation"```
3. Add PolicyWorkers with config ```policy_identifier="my_evaluation"```
4. Connect you Actor Worker with above-mentioned PolicyWorker and EvalManager with new inference/sample stream. It is suggested that you use "eval_" as prefix. _Note that if the name of one stream is the prefix of another, name-resolving streams won't be resolved correctly._
5. If you want a fixed number of evaluations for each version, set ```eval_games_per_version=100``` in EvalManager config.
6. Check the previous section on how to use a customized wandb_name.

### Metadata Parameter Database
Users may utilize metadata DB to implement population algorithms, e.g. Prioritized Fictitious-Self-Play. Try config your
experiment according to the following steps. If you meet any errors, please raise an issue on Github.

- NOTE: Currently, Metadata DB keeps track of the metadata of each policy `version`, NOT `tag` (See below for more on version and tags).
We did not implement the metadata of `tag`, because tags are subject to change during the experiments and their metadata 
cannot be trusted.

Steps:
1. Change your parameter db from ```Type.FILESYSTEM``` to ```Type.METADATA```. Make sure that all workers use the same type.
2. Setup a sample stream for metadata. For example, in PFSP, if your main agent(team A) is playing against past versions(team B), then all agents on team B should be connected to this new sample stream.
3. Evaluation workers will automatically update the metadata of each version. The metadata is the episode-wise average of the custom "EpisodeInfo" of you environment. Feel free to add more fields if you feel the current EpisodeInfo is not enough.
4. (Important!) In the policy worker(Or InlineInferenceStream) of team B, set the ```policy_identifier``` to the Mongodb Query which 
leads to your desired parameter version. (see below for some common queries.) If you use a policy identifier that by passes 
the metadataDB (i.e. version("123") or tag("latest")), the Eval Manager will receive a sample whose version cannot be
found in the metadata DB, this will cause your EvalManager to raise an error.

6. Metadata will expire in 3 days after the last update.

Example MongoDB query:

Note that your query may return multiple entries, the system will sample one parameter from the query result, uniformly.
Query can be single entry (dict) or a sequence of operations (list).
Remember to add ```md.``` prefix to you data field name.

1. randomly pick one: ```{"$match": {}}```
2. win_rate > 0.8: ```{"$match": {"md.win_rate": {"$gt": 0.8}}``` # "$lt"
3. win_rate >= 0.8: ```{"$match": {"md.win_rate": {"$gte": 0.8}}``` # "$lte"
4. 0.8=> win_rate >= 0.6 and episode_return > 10: ```{"$match": {"md.win_rate": {"$gte": 0.6, "$lte" 0.8}, {"md.episode_return": "$gt": 10}}```
5. top 10 win_rate: ```[{"$sort": {"md.win_rate": -1}}, {"$limit": 10}]```
6. top 10 win_rate with episode_return greater than 10: ```[{"$match": {"md.episode_return": "$gt": 10}}, {"$sort": {"md.win_rate": -1}}, {"$limit": 10}]```
7. If nothing is returned by the query, eval manager will try to fallback on tag `latest`. If `latest` is not available,
ParameterDB raises error.

Troubleshooting:
1. check if you forgot the $ dollar sign before your operators.
2. check if you forgot md. before your field name.
3. check if you field name is consistent with that in environment.EpisodeInfo.

### About tags
If you dive deeper into the parameter DB, the terms `tag` and `version` may get confusing. In general, `version` is 
dense and updated frequently as training proceeds, whereas `tags` are for workers to have some _consensus_ on how to use
the parameter versions.
These designs are still subject to changes and this is how things work currently:
1. A parameter DB is uniquely identified by {experiment_name + trial_name + policy_name}
2. There is only one writer to each parameter DB: master of the trainer workers. The master trainer may `push without tag`
or `push with tag`. In both cases, the pushed parameter will be tagged as `latest`. And currently, this is considered as 
the `main policy version` that is being trained, and policy workers by default follows tag `latest`. 
3. `Push with tag` happens at a configurable frequency in trainer worker. See `push_frequency_seconds` and `push_frequency_steps` 
in TrainerWorker config. At this frequency, the trainer worker will attach a tag which is the system time.

- NOTE: Currently, latest is a tag that is maintained internally by the parameter DB. In other words, tag latest is 
automatically added to each push, and it is not considered a push with tag if no additional tags are specified.

4. If the master trainer pushes `without tag`, the pushed parameter will be garbage-collected when it is outdate. 
In other words, the parameter will be considered as "saved for future use" only if the master trainer `push with tag`.
5. In the case of parameter DB that supports metadata, the pushed parameter will enter the metadata-database _only if_ it
is pushed with tags. 
6. If a version is pushed `without tag` and tagged later (e.g. for evaluation purpose), its metadata cannot be tracked, and
its parameter will be kept in the database until its last tag is removed. During its life time, it can be retrieved by its tag, 
version, but not though metadata query.
7. In our current implementation, version is the number of backward pass to get the parameter, which is consistent with
the `policy_verison` attributed of a policy, `policy_version` of the RolloutResult, and the `policy_version` of a 
SampleBatch.


### Population-Based Training (PBT)
User may use population_manager to run PBT experiments. Generally speaking, a population is a set of different 
policies. In our design, a `population` is a list of different policy_name-s. Population manager keeps track 
of the information (e.g. episode return, diversity, etc) of all policies within the population. It uses 
population algorithm to process the information and send requests to change experiment setup.

#### Vanilla PBT
Vanilla PBT trains a population of policies in parallel. A policy is deemed `ready` when it has been trained 
for a minimum interval (e.g., 1k steps) and will go through the exploit-and-explore process.

* In `exploit` process, weaker policy with low return will copy the parameters and hyperparameters of stronger
policies.
* In `explore` process, weaker policy will slightly modify the copied hyperparameters to do some exploration.

In our implementation, we use `explore_configs`, which is a list of dict, to specify how to do exploration.
Possible exploration methods include `perturb` by a factor (e.g., 1.2 or 0.8) and `resample` from a prior 
distribution. For detailed example on how to set `explore_configs`, see config/atari.py - AtariVanillaPBTMiniExperiment.

Suppose you already have an experiment config to train a single policy, you can configure a `vanilla_pbt` 
experiment to train a population by the following steps:
1. Set your `population`, which is a list of policy_name, e.g., `population=["policy_0", "policy_1"]`.
2. Set `population_algorithm` as vanilla_pbt and config with proper args.
3. Set different initial_hyperparams for each policy in the population.
4. Add a `population_sample_stream` for population_manager get samples from evaluation actors.
5. For each policy in the population, configure like a single policy experiment, and
    1. Connect `population_sample_stream` to evaluation actors.
    2. Remeber to use different inference streams and sample streams for different policies, except the newly
    added `population_sample_stream`.
6. Add a `population_manager` configured with `population`, `population_algorithm`, etc.

For a full vanilla_pbt experiment configuration example, see config/atari.py - AtariVanillaPBTMiniExperiment.

## wandb configuration
(Updated 2022.01.03)

Experiments now support the following wandb_configurations:
- entity
- project
- group
- job_type
- name

By default, these values will be setup by the system with the following defaults:
- entity = None
- porject = experiment_name (-e)
- group = trial_name (-f)
- job_type = worker_type (actor/policy/trainer/eval_manager)
- name = policy_name or "unnamed" 

The worker configuration will also be passed as argument ```config``` to ```wandb.init```. Nested dataclasses are not
parsed by W&B. For example, currently trainer configuration in TrainerWorker cannot be used as filter. This is a 
known issue and will be resolved in the future.  A workaround is to add the values that you want to filter on to 
wandb_name. See below for configuration instructions.

You may specify your customized configuration in you experiment configuration. An example will be:

```python

from legacy import experiments

actor_worker_count = 10
actor_worker_configs = [experiments.ActorWorker(...,  # worker configuration
                                           worker_info=experiments.WorkerInformation(
                                               wandb_entity="your_entity",
                                               wandb_project="your_project",
                                               wandb_group="your_group",
                                               wandb_job_type="actor_worker",
                                               wandb_name=f"my_perfect_wandb_name_actor_{worker_index}")
                                           ) for worker_index in range(actor_worker_count)]
```

For a full experiment configuration example, see config/football.py - FootballExperiment

## Read Next
- [optimize your experiment](optimize_your_experiment.md)