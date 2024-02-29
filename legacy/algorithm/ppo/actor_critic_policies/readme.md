# Actor Critic Policies
#### updated: (2022-04-05)

Implementation of actor-critic policy.
- Supported observation shapes are: flat, vector, pixels and voxels.
- Supported Action spaces are: Discrete, MultiDiscrete and Continuous.

### Arguments & Defaults
##### Common keyword config:
```python
class ActorCriticPolicy(SingleModelPytorchPolicy):

    def __init__(self,
                 obs_dim: Union[int, Dict[str, Union[int, Tuple[int]]]],
                 action_dim: Union[int, List[int]],
                 hidden_dim: int = 128,
                 state_dim: Optional[Union[int, Dict[str, Union[int, Tuple[int]]]]] = None,
                 value_dim: int = 1,
                 chunk_len: int = 10,
                 num_dense_layers: int = 2,
                 rnn_type: str = "gru",
                 cnn_layers: Dict[str, Tuple] = None,
                 use_maxpool: Dict[str, Tuple] = None,
                 num_rnn_layers: int = 1,
                 popart: bool = True,
                 activation: str = "relu",
                 layernorm: bool = True,
                 shared_backbone: bool = False,
                 continuous_action: bool = False,
                 auxiliary_head: bool = False,
                 seed=0,
                 **kwargs):
        """Actor critic style policy.
        Args:
            obs_dim: key-value pair of observation-shape. Passing int value is equivalent to {"obs": int}. Currently,
            Supported operations shapes are: int(MLP), Tuple[int, int](Conv1d), Tuple[int, int, int](Conv2d),
            Tuple[int, int, int, int](Conv3d).
            action_dim: action dimension. For discrete action, accepted types are int and list[int](Mulit-Discrete).
            For continuous action, accepted type is int.
            hidden_dim: hidden size of neural network. Observation/States are first mapped to this size, concatenated
            together, then passed through a mlp and possible a rnn.
            state_dim: Similar to obs_dim. If shared_backbone, state_dim has not effect. Overlaps are allowed between
            obse_dim and state_dim.
            value_dim: Size of state_value (same as the size of reward).
            chunk_len: RNN unroll length when training.
            num_dense_layers: number of dense layers between observation concatenation and rnn.
            rnn_type: "lstm" or "gru"
            cnn_layers: Key-value of user-specified convolution layers.
            use_maxpool: whether to use maxpool.
            Format is {"obs_key": [(output_channel, kernel_size, stride, padding), ...]}
            num_rnn_layers: Number of rnn layers.
            popart: Whether to use a popart head.
            activation: Supported are "relu" or "tanh".
            layernorm: Whether to use layer-norm.
            shared_backbone: Whether to use a separate backbone for critic.
            continuous_action: Whether to action space is continuous.
            auxiliary_head: Whether to use a auxiliary_head.
            seed: Seed of initial state.
            kwargs: Additional configuration passed to pytorch model. Currently supports "std_type" and "init_log_std",
            both used for continuous action.
        """
        ...
```
### Registration
```python
register("actor-critic", ActorCriticPolicy)
register("actor-critic-separate", functools.partial(ActorCriticPolicy, shared_backbone=False, auxiliary_head=False))
register("actor-critic-shared", functools.partial(ActorCriticPolicy, shared_backbone=True, auxiliary_head=False))
register("actor-critic-auxiliary", functools.partial(ActorCriticPolicy, shared_backbone=False, auxiliary_head=True))
register("gym_mujoco", functools.partial(ActorCriticPolicy, continuous_action=True))
register("actor-critic-separate-continuous-action", functools.partial(ActorCriticPolicy, continuous_action=True))
```