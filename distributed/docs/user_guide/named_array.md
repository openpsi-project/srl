# NamedArray

*NamedArray is a key data structure in the system and used almost everywhere. 
This intro will get your started.*

*We did not come up with the idea. Read the inline docstrings for more details.*

*Debugging named array related problems may become troublesome. Make sure you read this doc carefully.*

*Implementation credits to Wei Fu. https://github.com/garrett4wade*


### Why
Named array extends numpy array in the 
following ways.
1. Each NamedArray aggregates multiple numpy arrays, possibly of different shapes.
2. Each numpy array is given a name, providing a user-friendly way of indexing to the corresponding data.
3. Named arrays can be nested.

All of these come in handy in reinforcement learning, where data of different shapes and nesting-relations
are passed around between system components.

### Creating and Indexing a NamedArray

Let's use gym api for an example.

```python
import numpy as np
from base.namedarray import NamedArray


# Suppose episode info contains `episode_length` and `episode_return`.
# Subclassing
class EpisodeInfo(NamedArray):
    def __init__(self,
                 episode_length: np.ndarray,
                 episode_return: np.ndarray):
        super(EpisodeInfo, self).__init__(episode_length=episode_length,
                                          episode_return=episode_return)


# Use NamedArray directly
my_sample_batch = NamedArray(obs=np.random.randn((10, 10)),
                             reward=np.random.random((10, 1)),
                             done=np.zeros((10, 1), dtype=np.uint8),
                             info=None
                             )
```

As shown, the syntax is just like for python dataclasses. However, the datatype is restricted to numpy.ndarray or other
NamedArray class.

`EpisodeInfo` has two fields: `episode_length` and `episode_return`. Let's see how you can assign and get their 
values.

```python
ei = EpisodeInfo(episode_length=np.full(shape=(10, 1), fill_value=10),
                episode_return=np.ones(shape=(10, 1))
                )

print(ei.episode_return)
print(ei.episode_length.shape)
ei5 = ei[:5]
print(ei5)
print(ei5.shape)
```

Not surprisingly, `ei[:5]` returns an `EpisodeInfo` instance. The `shape` attribute of a NamedArray prints the shape
of each field iteratively. Now let's try how things work out for nested data.

```python
msb = NamedArray(obs=np.random.random(size=(10, 3, 200, 200)),
                 reward=np.random.random(size=(10, 1)),
                 done=np.array([False] * 9 + [True]),
                 info=None)
print(msb.shape)
print(len(msb))
print(msb["obs"])
print(msb[:5].shape)
```
As shown, `msb["obs"]` is equivalent to `msb.obs` and indexing will apply to sub-NamedArray.


### Aggregation and Mapping

Image that we are running a gym environment and result for each step is wrapped in a `MiniSampleBatch`. 

For demonstration purpose, let's give each field a default value.

```python
from base.namedarray import recursive_aggregate


msb = NamedArray(obs=np.random.random(size=(10, 3, 200, 200)),
                 reward=np.random.random(size=(10, 1)),
                 done=np.array([False] * 9 + [True]),
                 info=None)


msb_list = [msb for _ in range(10)]

agg_msb = recursive_aggregate(xs=msb_list, 
                              aggregate_fn=np.stack)
print(agg_msb.__class__)
print(agg_msb.shape)
```

By using `recursive_aggregate`, np.stack is applied to each field, except for those with value None. The aggregation 
result is returned as a new instance of `MiniSampleBatch`.

In other occasions, we may want to apply some function to each field of a single NamedArray instance. For example,
when training, all numpy arrays must be converted to pytorch Tensors, or must be moved to GPU for gradient computation.

```python
import torch
from base.namedarray import recursive_apply

torch_msb = recursive_apply(x=agg_msb,
                            fn=lambda x: torch.from_numpy(x).to("cuda:0"))
```

# FAQ

1. As a algorithm developer, where should I use NamedArray?
   - In your environment, reset/step would return a list of `StepResult`. The `observation` and `episode_info` in each
     StepResult should be NamedArray. Reward and done are numpy arrays.
   - In your policy, the `analyze` and `rollout` methods, the sample and rollout requests are passed in as NamedArray.
     You will have to indexing through them to get your data.
   - In your trainer, when you implement the `step` method, the sample_batch is passed in as a NamedArray. For specific
     cases like data chunking, you will have to use `recursive_apply`.

2. What happens if I use `recursive_aggregate` on NamedArrays of different shapes, possibly with some Nones?
    - Firstly, the nesting structure, including names of each field, must match. Otherwise, it causes error.
    - If all shapes of a specific data field(not nesting NamedArray) matches except some `None`s. All `None`s 
      will be filled with numpy.zeros(shape={shape_of_the_first_array}).
    - If any two arrays in a specific data field differs in shape, the aggregation causes error. 
    - One common cause is that the environment returns values (e.g. observation) of inconsistent shapes.
