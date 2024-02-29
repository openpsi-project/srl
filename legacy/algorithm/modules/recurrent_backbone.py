import math
import torch.nn as nn

from legacy.algorithm import modules


class RecurrentBackbone(nn.Module):

    @property
    def feature_dim(self):
        return self.__feature_dim

    def __init__(
            self,
            obs_dim: int,
            dense_layers: int,
            hidden_dim: int,
            rnn_type: str,
            num_rnn_layers: int,
            dense_layer_gain: float = math.sqrt(2),
            activation='relu',
            layernorm=True,
    ):
        super(RecurrentBackbone, self).__init__()

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'elu':
            act_fn = nn.ELU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise NotImplementedError(f"Activation function {activation} not implemented.")

        self.__feature_dim = hidden_dim
        self.__rnn_type = rnn_type
        self.fc = modules.mlp([obs_dim, *([hidden_dim] * dense_layers)], act_fn, layernorm=layernorm)
        for k, p in self.fc.named_parameters():
            if 'weight' in k and len(p.data.shape) >= 2:
                # filter out layer norm weights
                nn.init.orthogonal_(p.data, gain=dense_layer_gain)
            if 'bias' in k:
                nn.init.zeros_(p.data)

        self.num_rnn_layers = num_rnn_layers
        if self.num_rnn_layers:
            self.rnn = modules.AutoResetRNN(hidden_dim,
                                            hidden_dim,
                                            num_layers=num_rnn_layers,
                                            rnn_type=self.__rnn_type)
            self.rnn_norm = nn.LayerNorm([hidden_dim])
            for k, p in self.rnn.named_parameters():
                if 'weight' in k and len(p.data.shape) >= 2:
                    # filter out layer norm weights
                    nn.init.orthogonal_(p.data)
                if 'bias' in k:
                    nn.init.zeros_(p.data)

    def forward(self, obs, hx, on_reset=None):
        features = self.fc(obs)
        if self.num_rnn_layers > 0:
            features, hx = self.rnn(features, hx, on_reset)
            features = self.rnn_norm(features)
        return features, hx
