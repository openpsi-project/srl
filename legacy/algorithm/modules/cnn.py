from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn

import legacy.algorithm.modules.utils as utils


def cnn_output_dim(dimension, padding, dilation, kernel_size, stride):
    """Calculates the output height and width based on the input
    height and width to the convolution layer.
    ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
    """
    out_dimension = []
    for i in range(len(dimension)):
        out_dimension.append(
            int(
                np.floor(((dimension[i] + 2 * padding[i] - dilation[i] *
                           (kernel_size[i] - 1) - 1) / stride[i]) + 1)))
    if not all([d > 0 for d in out_dimension]):
        raise ValueError(f"CNN Dimension error, got {out_dimension} after convolution")
    return tuple(out_dimension)


def maxpool_output_dim(dimension, dilation, kernel_size, stride):
    """Calculates the output height and width based on the input
    height and width to the convolution layer.
    ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
    """
    out_dimension = []
    for i in range(len(dimension)):
        out_dimension.append(
            int(np.floor(((dimension[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1)))
    if not all([d > 0 for d in out_dimension]):
        raise ValueError(f"CNN Dimension error, got {out_dimension} after convolution")
    return tuple(out_dimension)


class Convolution(nn.Module):
    """A model that uses Conv2d and max-pooling to embed a image to vector.
    """

    def __init__(self, input_shape: Tuple, cnn_layers: Optional[List[Tuple[int, int, int, int, str]]],
                 use_maxpool: bool, activation, hidden_size: int, use_orthogonal: bool):
        """Initialization method of ImageToVector (auto-cnn).
        Args:
            input_shape: Shape of input image, channel first.
            cnn_layers: List of user-specified cnn layer configuration (out_channels, kernel_size, stride, padding).
            use_maxpool: Whether to use a maxpool2d of size 2 after each layer.
            activation: nn.Relu / nn.Tanh.
            hidden_size: output dimension.
            use_orthogonal: whether to use orthogonal initialization method.
        """
        super(Convolution, self).__init__()
        self.__input_shape = input_shape
        self.__input_dims = len(input_shape)
        self.__use_maxpool = use_maxpool
        self.__cnn_layers = cnn_layers
        self.activation_name = activation

        if self.__input_dims == 2:
            self.__cnn = nn.Conv1d
            self.__max_pool = nn.MaxPool1d
        elif self.__input_dims == 3:
            self.__cnn = nn.Conv2d
            self.__max_pool = nn.MaxPool2d
        elif self.__input_dims == 4:
            self.__cnn = nn.Conv3d
            self.__max_pool = nn.MaxPool3d

        self.__activation = activation

        self.__weights_init_gain = nn.init.calculate_gain(activation.__name__.lower())
        self.__weights_init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
        self.__bias_init_constant = 0
        self.__bias_init_method = nn.init.constant_

        self.__hidden_size = hidden_size
        self.__model = self._build_cnn_model()

    def __init_layer(self, layer: nn.Module):
        self.__weights_init_method(layer.weight.data, gain=self.__weights_init_gain)
        self.__bias_init_method(layer.bias.data, val=self.__bias_init_constant)
        return layer

    def __linear_layers(self, input_dim):
        linear_sizes = [input_dim]
        while linear_sizes[-1] > self.__hidden_size * 8:
            linear_sizes.append(linear_sizes[-1] // 2)
        linear_sizes.append(self.__hidden_size)
        return utils.mlp(linear_sizes)

    def _build_cnn_model(self):
        cnn_layers = []
        num_channels, *dimension = self.__input_shape
        if self.__cnn_layers is None:
            self.__cnn_layers = [(num_channels, 5, 1, 0, "zeros"), (num_channels * 2, 3, 1, 0, "zeros"),
                                 (num_channels, 3, 1, 0, "zeros")]
        for i, (out_channels, kernel_size, stride, padding, padding_mode) in enumerate(self.__cnn_layers):
            if self.__use_maxpool and i != len(self.__cnn_layers) - 1:
                # Add a maxpool layer if not yet the last layer.
                cnn_layers.append(self.__max_pool(2))
                dimension = maxpool_output_dim(dimension=dimension,
                                               dilation=np.array([1] * self.__input_dims, dtype=np.float32),
                                               kernel_size=np.array([2] * self.__input_dims,
                                                                    dtype=np.float32),
                                               stride=np.array([2] * self.__input_dims, dtype=np.float32))

            cnn_layers.append(
                self.__init_layer(
                    self.__cnn(in_channels=num_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               padding_mode=padding_mode)))
            dimension = cnn_output_dim(dimension=dimension,
                                       padding=np.array([padding] * self.__input_dims, dtype=np.float32),
                                       dilation=np.array([1] * self.__input_dims, dtype=np.float32),
                                       kernel_size=np.array([kernel_size] * self.__input_dims,
                                                            dtype=np.float32),
                                       stride=np.array([stride] * self.__input_dims, dtype=np.float32))

            cnn_layers.append(self.__activation())
            num_channels = out_channels
        cnn_layers.extend([nn.Flatten(), self.__linear_layers(input_dim=num_channels * np.prod(dimension))])

        return nn.Sequential(*cnn_layers)

    def forward(self, x):
        T, B = x.size()[:2]
        x = torch.flatten(x, start_dim=0, end_dim=1)
        cnn_x = self.__model(x)

        return cnn_x.reshape(T, B, -1)
