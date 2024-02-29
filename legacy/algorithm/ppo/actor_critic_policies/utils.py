from typing import Dict, Union, Tuple
import dataclasses
import torch.nn as nn

from legacy.algorithm import modules


@dataclasses.dataclass
class ActionIndex:
    start: int
    end: int


def get_action_indices(act_dims):
    """Convert action dimensions to indices. Results can be used to parse model forward results.
    Args:
        act_dims: dimensions of all value heads.
    Returns:
        action_indices (List[ActionIndex]): index of each action.
    Example:
         >>> indices = get_action_indices([3, 5, 7])
         >>> print(indices)
         [ActionIndex(start=0, end=3), ActionIndex(start=3, end=8), ActionIndex(start=8, end=15)]
    """
    curr_action_index = 0
    action_indices = []
    for dim in act_dims:
        action_indices.append(ActionIndex(curr_action_index, curr_action_index + dim))
        curr_action_index += dim
    return action_indices


def make_models_for_obs(obs_dim: Dict[str, Union[int, Tuple]], hidden_dim: int, activation,
                        cnn_layers: Dict[str, Tuple], use_maxpool: Dict[str, bool]):
    """Make models based on a dict of observation dimension.
    Args:
        obs_dim: Key-Value pair of observation_name and dimension of observation {"obs_name": obs_dim},
        hidden_dim: Embedding dimension of all observations.
        activation: nn.ReLU or nn.Tanh
        cnn_layers: Key-Value pair of observation_name and user-specified cnn-layers.
        use_maxpool: whether to use maxpool for each obs. effective only with convolutional nets.
    """
    obs_embd_dict = nn.ModuleDict()
    for k, v in obs_dim.items():
        if isinstance(v, int):
            obs_embd_dict.update({
                k: nn.Sequential(nn.LayerNorm([v]),
                                 modules.mlp([v, hidden_dim], activation=activation, layernorm=True))
            })
        elif len(v) in [2, 3, 4]:
            obs_embd_dict.update({
                k: nn.Sequential(
                    nn.LayerNorm(v),
                    modules.Convolution(v,
                                        cnn_layers=cnn_layers.get(k, None),
                                        use_maxpool=use_maxpool.get(k, False),
                                        activation=activation,
                                        hidden_size=hidden_dim,
                                        use_orthogonal=True))
            })
        else:
            raise NotImplementedError()
    return obs_embd_dict
