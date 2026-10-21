import math

import torch
from torch import nn


class ChunkedCrossAttention(nn.Module):
    component_id = "chunked-cross-attention"

    def __init__(self, d_model=32, num_heads=4, dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if min(d_model, num_heads) <= 0 or d_model % num_heads:
            raise ValueError("invalid attention dimensions")
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def _attend(self, query, key):
        batch, steps, width = query.shape
        heads = self.num_heads
        head_dim = width // heads
        q = self.q_proj(query).view(batch, steps, heads, head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch, key.shape[1], heads, head_dim).transpose(1, 2)
        v = self.v_proj(key).view(batch, key.shape[1], heads, head_dim).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
        mixed = torch.softmax(scores, dim=-1) @ v
        return self.o_proj(mixed.transpose(1, 2).contiguous().view(batch, steps, width))

    def forward(self, hidden, retrieved=None):
        hidden = torch.as_tensor(hidden)
        if retrieved is None:
            retrieved = hidden.roll(shifts=1, dims=1)
        else:
            retrieved = torch.as_tensor(retrieved)
        if hidden.ndim != 4 or retrieved.ndim != 4 or hidden.shape[0] != retrieved.shape[0] or hidden.shape[1] != retrieved.shape[1]:
            raise ValueError("hidden and retrieved must be (batch, chunks, length, d_model)")
        if hidden.shape[-1] != self.d_model or retrieved.shape[-1] != self.d_model:
            raise ValueError("width must equal d_model")
        out = hidden.clone()
        for index in range(1, hidden.shape[1]):
            out[:, index] = hidden[:, index] + self._attend(hidden[:, index], retrieved[:, index - 1])
        return out
