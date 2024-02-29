import torch
import torch.nn as nn
import torch.nn.functional as F

from legacy.algorithm import modules


class QMixer(nn.Module):

    def __init__(
        self,
        num_agents,
        state_dim,
        hidden_dim,
        num_hypernet_layers,
        hypernet_hidden_dim,
        popart,
    ):
        super(QMixer, self).__init__()

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.hypernet_hidden_dim = hypernet_hidden_dim

        def init(module):
            nn.init.orthogonal_(module.weight.data, gain=0.01)
            return module

        self.state_norm = nn.LayerNorm([self.state_dim])

        if num_hypernet_layers == 1:
            self.hyper_w1 = init(nn.Linear(state_dim, num_agents * hidden_dim))
            self.hyper_w2 = init(nn.Linear(state_dim, hidden_dim))
        elif num_hypernet_layers == 2:
            self.hyper_w1 = nn.Sequential(init(nn.Linear(state_dim, hypernet_hidden_dim)), nn.ReLU(),
                                          init(nn.Linear(hypernet_hidden_dim, num_agents * hidden_dim)))
            self.hyper_w2 = nn.Sequential(init(nn.Linear(state_dim, hypernet_hidden_dim)), nn.ReLU(),
                                          init(nn.Linear(hypernet_hidden_dim, hidden_dim)))

        self.hyper_b1 = init(nn.Linear(state_dim, hidden_dim))
        self.hyper_b2 = nn.Sequential(init(nn.Linear(state_dim, hypernet_hidden_dim)), nn.ReLU(),
                                      init(nn.Linear(hypernet_hidden_dim, 1)))

        self.__popart = popart
        if popart:
            self.popart_head = modules.PopArtValueHead(1, 1)

    def forward(self, q_i, state):  # (T, B, *D)
        batch_size = q_i.size(1)
        state = state.view(-1, batch_size, self.state_dim).float()
        # state = self.state_norm(state)
        q_i = q_i.view(-1, batch_size, 1, self.num_agents)
        state = self.state_norm(state)

        w1 = torch.abs(self.hyper_w1(state))
        b1 = self.hyper_b1(state)
        w1 = w1.view(-1, batch_size, self.num_agents, self.hidden_dim)
        b1 = b1.view(-1, batch_size, 1, self.hidden_dim)
        hidden_layer = F.elu(torch.matmul(q_i, w1) + b1)

        w2 = torch.abs(self.hyper_w2(state))
        b2 = self.hyper_b2(state)
        w2 = w2.view(-1, batch_size, self.hidden_dim, 1)
        b2 = b2.view(-1, batch_size, 1, 1)
        out = torch.matmul(hidden_layer, w2) + b2
        q_tot = out.view(-1, batch_size, 1, 1)
        if self.__popart:
            q_tot = self.popart_head(q_tot)
        return q_tot


class VDNMixer(nn.Module):

    def __init__(self, num_agents, *args, **kwargs):
        super().__init__()
        self.num_agents = num_agents
        self.mixer = nn.Linear(num_agents, 1, bias=False)

    def forward(self, q_i, state):  # (T, B, *D)
        batch_size = q_i.size(1)
        q_i = q_i.view(-1, batch_size, self.num_agents)
        return self.mixer(q_i).unsqueeze(-1)


Mixer = {}
Mixer["qmix"] = QMixer
Mixer['vdn'] = VDNMixer
