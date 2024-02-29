from typing import Optional, Dict, List
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """
    Overview:
        Positional Embedding used in vanilla Transformer
    .. note::
        Adapted from https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py
    """

    def __init__(self, embedding_dim: int):
        """
        Arguments:
            - embedding_dim: (:obj:`int`): dimension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        inv_freq = 1 / (10000**(torch.arange(0.0, embedding_dim, 2.0) / embedding_dim))  # (embedding_dim / 2)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq: torch.Tensor):
        """
        Overview:
            Compute positional embedding
        Arguments:
            - pos_seq: (:obj:`torch.Tensor`): positional sequence,
             usually a 1D integer sequence as [seq_len-1, seq_len-2, ..., 1, 0],
        Returns:
            - pos_embedding: (:obj:`torch.Tensor`): positional embedding. Shape (seq_len, 1, embedding_dim)
        """
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        # For position embedding, the order of sin/cos is negligible.
        # This is because tokens are consumed by the matrix multiplication which is permutation-invariant.
        pos_embedding = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_embedding.unsqueeze(1)


class GRUGatingUnit(torch.nn.Module):
    """
    Overview:
        GRU Gating Unit used in GTrXL.
    """

    def __init__(self, input_dim: int, bg: float = 2.):
        """
        Arguments:
            - input_dim: (:obj:`int`): dimension of input.
            - bg (:obj:`bg`): gate bias. By setting bg > 0 we can explicitly initialize the gating mechanism to
            be close to the identity map. This can greatly improve the learning speed and stability since it
            initializes the agent close to a Markovian policy (ignore attention at the beginning).
        """
        super(GRUGatingUnit, self).__init__()
        self.Wr = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Overview:
            Compute output value with gating mechanism
        Arguments:
            - x: (:obj:`torch.Tensor`): first input.
            - y: (:obj:`torch.Tensor`): second input.
            x and y have same shape and last shape is input_dim.
        Returns:
            - g: (:obj:`torch.Tensor`): output of GRU. Same shape of x and y.
        """
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))  # element wise multiplication
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g  # x.shape == y.shape == g.shape


class Memory:

    def __init__(
        self,
        memory_len: int = 20,
        batch_size: int = 64,
        embedding_dim: int = 256,
        layer_num: int = 3,
    ) -> None:
        """
        Arguments:
            - memory_len (:obj:`int`): dimension of memory (how many past observations to use as memory)
            - batch_size (:obj:`int`): dimension of each batch
            - embedding_dim (:obj:`int`): dimension of embedding (dimension of a single observation after embedding)
            - layer_num (:obj:`int`): number of transformer layers
        """
        self.embedding_dim = embedding_dim
        self.bs = batch_size
        self.layer_num = layer_num
        self.memory_len = memory_len
        self.memory = torch.zeros(self.layer_num + 1,
                                  self.memory_len,
                                  self.bs,
                                  self.embedding_dim,
                                  dtype=torch.float)


class AttentionXL(torch.nn.Module):
    """
    Overview:
        Attention of TransformerXL.
    """

    def __init__(self, input_dim: int, head_dim: int, head_num: int, dropout: nn.Module) -> None:
        """Overview:
            Init AttentionXL.
        Arguments:
            - input_dim (:obj:`int`): dimension of input
            - head_dim (:obj:`int`): dimension of each head
            - head_num (:obj:`int`): number of heads for multihead attention
            - dropout (:obj:`nn.Module`): dropout function
        """
        super(AttentionXL, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.dropout = dropout
        self.attention_kv = nn.Linear(input_dim, head_dim * head_num * 2)  # key, value
        self.attention_q = nn.Linear(input_dim,
                                     head_dim * head_num)  # query (not computed with past hidden states)
        self.project = nn.Linear(head_dim * head_num, input_dim)  # project attention output back to input_dim
        self.project_pos = nn.Linear(input_dim, head_dim * head_num)  # project the positional embedding
        self.scale = 1 / (head_dim**0.5)  # for scaled dot product attention

    def _rel_shift(self, x: torch.Tensor, zero_upper: bool = False):
        """
        Overview:
            Relatively shift the attention score matrix.
        Example:
            a00 a01 a02      0 a00 a01 a02       0  a00 a01      a02  0  a10     a02  0   0
            a10 a11 a12  =>  0 a10 a11 a12  =>  a02  0  a10  =>  a11 a12  0  =>  a11 a12  0
            a20 a21 a22      0 a20 a21 a22      a11 a12  0       a20 a21 a22     a20 a21 a22
                                                a20 a21 a22
            1) Append one "column" of zeros to the left
            2) Reshape the matrix from [3 x 4] into [4 x 3]
            3) Remove the first "row"
            4) Mask out the upper triangle (optional)
        .. note::
            See the following material for better understanding:
                https://github.com/kimiyoung/transformer-xl/issues/8
                https://arxiv.org/pdf/1901.02860.pdf (Appendix B)
        Arguments:
            - x (:obj:`torch.Tensor`): input tensor of shape (cur_seq, full_seq, bs, head_num).
            - zero_upper (:obj:`bool`): if True set the upper-right triangle to zero.
        Returns:
            - x (:obj:`torch.Tensor`): input after relative shift. Shape (cur_seq, full_seq, bs, head_num).
        """
        x_padded = F.pad(x, [1, 0])  # step 1
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))  # step 2
        x = x_padded[:, :, 1:].view_as(x)  # step 3
        if zero_upper:
            ones = torch.ones((x.size(2), x.size(3))).unsqueeze(0).unsqueeze(0)
            x = x * torch.tril(ones.to(x.device), x.size(3) - x.size(2))  # step 4
        return x

    def forward(
        self,
        inputs: torch.Tensor,
        pos_embedding: torch.Tensor,
        full_input: torch.Tensor,
        u: torch.nn.Parameter,
        v: torch.nn.Parameter,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Overview:
            Compute AttentionXL.
        Arguments:
            - inputs (:obj:`torch.Tensor`): attention input of shape (cur_seq, bs, input_dim)
            - pos_embedding (:obj:`torch.Tensor`): positional embedding of shape (full_seq, 1, full_seq)
            - full_input (:obj:`torch.Tensor`): memory + input concatenation of shape (full_seq, bs, input_dim)
            - u (:obj:`torch.nn.Parameter`): content parameter of shape (head_num, head_dim)
            - v (:obj:`torch.nn.Parameter`): position parameter of shape (head_num, head_dim)
            - mask (:obj:`Optional[torch.Tensor]`): attention mask of shape (cur_seq, full_seq, 1)
            full_seq = prev_seq + cur_seq
        Returns:
            - output (:obj:`torch.Tensor`): attention output of shape (cur_seq, bs, input_dim)
        """
        bs, cur_seq, full_seq = inputs.shape[1], inputs.shape[0], full_input.shape[0]
        prev_seq = full_seq - cur_seq

        kv = self.attention_kv(full_input)
        key, value = torch.chunk(kv, 2, dim=-1)  # full_seq x bs x num_head*dim_head
        query = self.attention_q(inputs)  # cur_seq x bs x num_head*dim_head
        r = self.project_pos(pos_embedding)  # full_seq x 1 x num_head*dim_head

        key = key.view(full_seq, bs, self.head_num, self.head_dim)
        query = query.view(cur_seq, bs, self.head_num, self.head_dim)
        value = value.view(cur_seq + prev_seq, bs, self.head_num, self.head_dim)
        r = r.view(full_seq, self.head_num, self.head_dim)

        # (query + u) * key^T
        q_u = query + u
        content_attn = q_u.permute(1, 2, 0, 3) @ key.permute(1, 2, 3, 0)  # bs x head_num x cur_seq x full_seq

        # (query + v) * R^T
        q_v = query + v
        position_attn = q_v.permute(1, 2, 0, 3) @ r.permute(1, 2, 0)  # bs x head_num x cur_seq x full_seq
        position_attn = self._rel_shift(position_attn)

        attn = content_attn + position_attn  # bs x head_num x cur_seq x full_seq
        attn.mul_(self.scale)

        # fills float('-inf') where mask is True to let softmax ignore those positions.
        if mask is not None:
            mask = mask.permute(2, 0, 1).unsqueeze(1)  # 1 x 1 x cur_seq x full_seq
            # assert mask.shape[2:] == attn.shape[2:]  # check shape of mask
            attn = attn.masked_fill(mask, -float("inf")).type_as(attn)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # multiply softmax output by value
        attn_vec = attn @ value.permute(1, 2, 0, 3)
        attn_vec = attn_vec.permute(2, 0, 1, 3)

        attn_vec = attn_vec.contiguous().view(cur_seq, bs, self.head_num * self.head_dim)
        # cur_seq x bs x head_num * head_dim
        output = self.dropout(self.project(attn_vec))  # cur_seq x bs x input_dim
        return output


class GatedTransformerXLLayer(torch.nn.Module):
    """
    Overview:
        Attention layer of GTrXL
    """

    def __init__(self,
                 input_dim: int,
                 head_dim: int,
                 hidden_dim: int,
                 head_num: int,
                 mlp_num: int,
                 dropout: nn.Module,
                 activation: nn.Module,
                 gru_gating: bool = True,
                 gru_bias: float = 2.) -> None:
        """
        Arguments:
            - input_dim (:obj:`int`): dimension of input
            - head_dim (:obj:`int`): dimension of each head
            - hidden_dim (:obj:`int`): dimension of hidden layer in mlp
            - head_num (:obj:`int`): number of heads for multihead attention
            - mlp_num (:obj:`int`): number of mlp layers in attention layer
            - dropout (:obj:`nn.Module`): dropout
            - activation (:obj:`nn.Module`): activation function
            - gru_gating (:obj:`bool`): if False replace GRU gates with residual connections
            - gru_bias (:obj:`float`): GRU gate bias
        """
        super(GatedTransformerXLLayer, self).__init__()
        self.dropout = dropout
        self.gating = gru_gating
        if self.gating is True:
            self.gate1 = GRUGatingUnit(input_dim, gru_bias)
            self.gate2 = GRUGatingUnit(input_dim, gru_bias)
        self.attention = AttentionXL(
            input_dim,
            head_dim,
            head_num,
            dropout,
        )
        layers = []
        dims = [input_dim] + [hidden_dim] * (mlp_num - 1) + [input_dim]
        for i in range(mlp_num):
            layers.append(nn.Sequential(nn.Linear(dims[i], dims[i + 1]), activation))
            if i != mlp_num - 1:
                layers.append(self.dropout)
        layers.append(self.dropout)
        self.mlp = nn.Sequential(*layers)
        self.layernorm1 = nn.LayerNorm(input_dim)
        self.layernorm2 = nn.LayerNorm(input_dim)
        self.activation = activation

    def forward(
        self,
        inputs: torch.Tensor,
        pos_embedding: torch.Tensor,
        u: torch.nn.Parameter,
        v: torch.nn.Parameter,
        memory: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Overview:
            Compute forward pass of GTrXL layer.
        Arguments:
            - inputs (:obj:`torch.Tensor`): attention input of shape (cur_seq, bs, input_dim)
            - pos_embedding (:obj:`torch.Tensor`): positional embedding of shape (full_seq, 1, full_seq)
            - u (:obj:`torch.nn.Parameter`): content parameter of shape (head_num, head_dim)
            - v (:obj:`torch.nn.Parameter`): position parameter of shape (head_num, head_dim)
            - memory (:obj:`Optional[torch.Tensor]`): memory of shape (prev_seq, bs, input_dim)
            - mask (:obj:`Optional[torch.Tensor]`): attention mask of shape (cur_seq, full_seq, 1)
            full_seq = prev_seq + cur_seq
        Returns:
            - output (:obj:`torch.Tensor`): layer output of shape (cur_seq, bs, input_dim)
        """

        # concat memory with input across sequence dimension
        full_input = torch.cat([memory, inputs], dim=0)  # full_seq x bs x input_dim
        x1 = self.layernorm1(full_input)
        a1 = self.dropout(self.attention(inputs, pos_embedding, x1, u, v, mask=mask))
        a1 = self.activation(a1)  # RELU after attention
        o1 = self.gate1(inputs, a1) if self.gating else inputs + a1
        x2 = self.layernorm2(o1)
        m2 = self.dropout(self.mlp(x2))
        o2 = self.gate2(o1, m2) if self.gating else o1 + m2
        return o2


class GTrXL(nn.Module):
    """
    Overview:
        GTrXL Transformer implementation.
    .. note::
        For details refer to Stabilizing Transformer for Reinforcement Learning: https://arxiv.org/abs/1910.06764
    """

    def __init__(self,
                 input_dim: int,
                 head_dim: int = 128,
                 embedding_dim: int = 256,
                 head_num: int = 2,
                 mlp_num: int = 2,
                 layer_num: int = 3,
                 dropout_ratio: float = 0.,
                 activation: nn.Module = nn.ReLU(),
                 gru_gating: bool = True,
                 gru_bias: float = 2.,
                 use_embedding_layer: bool = True,
                 batch_first: bool = False) -> None:
        """Overview:
            Init GTrXL Model
        Arguments:
            - input_dim (:obj:`int`): dimension of input (dimension of a single observation)
            - head_dim (:obj:`int`): dimension of each head
            - hidden_dim (:obj:`int`): dimension of hidden layer in mlp
            - embedding_dim (:obj:`int`): dimension of embedding (dimension of a single observation after embedding)
            - head_num (:obj:`int`): number of heads for multihead attention
            - mlp_num (:obj:`int`): number of mlp layers in attention layer
            - layer_num (:obj:`int`): number of transformer layers
            - dropout_ratio (:obj:`float`): dropout ratio
            - activation (:obj:`nn.Module`): activation function
            - gru_gating (:obj:`bool`): if False replace GRU gates with residual connections
            - gru_bias (:obj:`float`): GRU gate bias
            - use_embedding_layer (:obj:`bool`): default True. If False, don't use input embedding layer.
        """
        super(GTrXL, self).__init__()
        assert embedding_dim % 2 == 0, 'embedding_dim={} should be even'.format(input_dim)
        self.batch_first = batch_first
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        if isinstance(input_dim, list):
            input_dim = np.prod(input_dim)
        self.use_embedding_layer = use_embedding_layer
        if use_embedding_layer:
            self.embedding = nn.Sequential(nn.Linear(input_dim, embedding_dim), activation)
        self.activation = activation
        self.pos_embedding = PositionalEmbedding(embedding_dim)
        layers = []
        dims = [embedding_dim] + [embedding_dim] * layer_num
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        for i in range(layer_num):
            layers.append(
                GatedTransformerXLLayer(dims[i], head_dim, embedding_dim, head_num, mlp_num, self.dropout,
                                        self.activation, gru_gating, gru_bias))
        self.layers = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim
        # u and v are the parameters to compute global content bias and global positional bias
        self.u, self.v = (
            torch.nn.Parameter(torch.zeros(self.head_num, self.head_dim)),
            torch.nn.Parameter(torch.zeros(self.head_num, self.head_dim)),
        )
        self.att_mask = {
        }  # create an attention mask for each different seq_len, in this way we don't need to create a
        # new one each time we call the forward method
        self.pos_embedding_dict = {}  # create a pos embedding for each different seq_len

    def forward(
            self,
            x: torch.Tensor,
            memory: torch.Tensor,  # (num_layers, memory_len, bs, hidden_size)
    ) -> Dict[str, torch.Tensor]:
        """
        Overview:
            GTrXL forward pass.
        Arguments:
            - x (:obj:`torch.Tensor`): input tensor. Shape (seq_len, bs, input_size).
            - batch_first (:obj:`bool`): if the input data has shape (bs, seq_len, input_size), set this param to \
                ``True`` in order to transpose along the first and second dimension and obtain shape \
                (seq_len, bs, input_size). This param doesn't affects the output memory.
            - return_mem (:obj:`bool`): if this param is False, return only the output tensor without dict.
        Returns:
            - x (:obj:`Dict[str, torch.Tensor]`): dict containing transformer output of shape \
             (seq_len, bs, embedding_size) and memory of shape (layer_num, seq_len, bs, embedding_size)
        """
        if self.batch_first:
            x = torch.transpose(x, 1, 0)  # bs x cur_seq x input_dim -> cur_seq x bs x input_dim
        cur_seq, bs = x.shape[:2]

        if self.use_embedding_layer:
            x = self.dropout(self.embedding(x))
        prev_seq = memory.shape[1]
        full_seq = cur_seq + prev_seq

        if cur_seq in self.att_mask.keys():
            attn_mask = self.att_mask[cur_seq]
        else:
            attn_mask = (
                torch.triu(
                    torch.ones((cur_seq, full_seq)),
                    diagonal=1 + prev_seq,  # fixed in train, eval, collect
                ).bool().unsqueeze(-1).to(x.device))  # cur_seq x full_seq x 1
            self.att_mask[cur_seq] = attn_mask

        if cur_seq in self.pos_embedding_dict.keys():
            pos_embedding = self.pos_embedding_dict[cur_seq]
        else:
            pos_ips = torch.arange(full_seq - 1, -1, -1.0, dtype=torch.float)  # full_seq
            pos_embedding = self.pos_embedding(pos_ips.to(x.device))
            self.pos_embedding_dict[cur_seq] = pos_embedding
        pos_embedding = self.dropout(pos_embedding)  # full_seq x 1 x embedding_dim

        hidden_state = [x.detach()]
        out = x
        for i in range(self.layer_num):
            layer = self.layers[i]
            out = layer(
                out,
                pos_embedding,
                self.u,
                self.v,
                mask=attn_mask,
                memory=memory[i],  # (layer_num+1) x memory_len x batch_size x embedding_dim
            )  # cur_seq x bs x embedding_dim
            hidden_state.append(out.detach())

        out = self.dropout(out)
        seq_len = hidden_state[0].shape[0]
        memory_ = torch.cat([memory, torch.stack(hidden_state)], dim=1)[:, seq_len:]

        if self.batch_first:
            out = torch.transpose(out, 1, 0)  # cur_seq x bs x embedding_dim -> bs x cur_seq x embedding_dim

        return out, memory_
