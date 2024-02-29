import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CatSelfEmbedding(nn.Module):

    def __init__(self, self_dim, others_shape_dict, d_embedding, use_orthogonal=True):
        super(CatSelfEmbedding, self).__init__()
        self.self_dim = self_dim
        self.others_shape_dict = others_shape_dict
        self.d_embedding = d_embedding

        def get_layer(input_dim, output_dim):
            linear = nn.Linear(input_dim, output_dim)
            nn.init.xavier_uniform_(linear.weight.data)
            return nn.Sequential(linear, nn.ReLU(inplace=True))

        self.others_keys = sorted(self.others_shape_dict.keys())
        self.self_embedding = get_layer(self_dim, d_embedding)
        for k in self.others_keys:
            if 'mask' not in k:
                setattr(self, k + '_fc', get_layer(others_shape_dict[k][-1] + self_dim, d_embedding))

    def forward(self, self_vec, **inputs):
        other_embeddings = []
        self_embedding = self.self_embedding(self_vec)
        self_vec_ = self_vec.unsqueeze(-2)
        for k, x in inputs.items():
            assert k in self.others_keys
            expand_shape = [-1 for _ in range(len(x.shape))]
            expand_shape[-2] = x.shape[-2]
            x_ = torch.cat([self_vec_.expand(*expand_shape), x], -1)
            other_embeddings.append(getattr(self, k + '_fc')(x_))

        other_embeddings = torch.cat(other_embeddings, dim=-2)
        return self_embedding, other_embeddings


def ScaledDotProductAttention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(-2).unsqueeze(-2)
        scores = scores - (1 - mask) * 1e10
    # in case of overflow
    scores = scores - scores.max(dim=-1, keepdim=True)[0]
    scores = F.softmax(scores, dim=-1)
    if mask is not None:
        # for stablity
        scores = scores * mask

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, input_dim, heads, d_head, dropout=0.0, use_orthogonal=True):
        super(MultiHeadSelfAttention, self).__init__()

        self.d_model = d_head * heads
        self.d_head = d_head
        self.h = heads

        self.pre_norm = nn.LayerNorm(input_dim)
        self.q_linear = nn.Linear(input_dim, self.d_model)
        nn.init.normal_(self.q_linear.weight.data, std=math.sqrt(0.125 / input_dim))
        self.k_linear = nn.Linear(input_dim, self.d_model)
        nn.init.normal_(self.k_linear.weight.data, std=math.sqrt(0.125 / input_dim))

        self.v_linear = nn.Linear(input_dim, self.d_model)
        nn.init.normal_(self.v_linear.weight.data, std=math.sqrt(0.125 / input_dim))

        # self.attn_dropout = nn.Dropout(dropout)
        self.attn_dropout = None

    def forward(self, x, mask, use_ckpt=False):
        x = self.pre_norm(x)
        # perform linear operation and split into h heads
        k = self.k_linear(x).view(*x.shape[:-1], self.h, self.d_head).transpose(-2, -3)
        q = self.q_linear(x).view(*x.shape[:-1], self.h, self.d_head).transpose(-2, -3)
        v = self.v_linear(x).view(*x.shape[:-1], self.h, self.d_head).transpose(-2, -3)

        # calculate attention
        scores = ScaledDotProductAttention(q, k, v, self.d_head, mask, self.attn_dropout)

        # concatenate heads and put through final linear layer
        return scores.transpose(-2, -3).contiguous().view(*x.shape[:-1], self.d_model)


class ResidualMultiHeadSelfAttention(nn.Module):

    def __init__(self, input_dim, heads, d_head, dropout=0.0, use_orthogonal=True):
        super(ResidualMultiHeadSelfAttention, self).__init__()
        self.d_model = heads * d_head
        self.attn = MultiHeadSelfAttention(input_dim, heads, d_head, dropout, use_orthogonal)

        post_linear = nn.Linear(self.d_model, self.d_model)
        nn.init.normal_(post_linear.weight.data, std=math.sqrt(0.125 / self.d_model))
        self.dense = post_linear
        self.residual_norm = nn.LayerNorm(self.d_model)
        # self.dropout_after_attn = nn.Dropout(dropout)
        self.dropout_after_attn = None

    def forward(self, x, mask, use_ckpt=False):
        scores = self.dense(self.attn(x, mask, use_ckpt))
        if self.dropout_after_attn is not None:
            scores = self.dropout_after_attn(scores)
        return self.residual_norm(x + scores)


def masked_avg_pooling(scores, mask=None):
    if mask is None:
        return scores.mean(-2)
    else:
        assert mask.shape[-1] == scores.shape[-2]
        masked_scores = scores * mask.unsqueeze(-1)
        return masked_scores.sum(-2) / (mask.sum(-1, keepdim=True) + 1e-5)
