import math

import torch
from torch import nn


class LowRankAttention(nn.Module):
    component_id = "linformer"
    def __init__(self, max_seq_len=4, rank=2):
        super().__init__()
        if not isinstance(max_seq_len, int) or isinstance(max_seq_len, bool) or max_seq_len < 1:
            raise ValueError("max_seq_len must be a positive integer")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= max_seq_len:
            raise ValueError("rank must be an integer between 1 and max_seq_len")
        self.max_seq_len = max_seq_len
        self.rank = rank
        self.key_projection = nn.Parameter(torch.empty(rank, max_seq_len))
        self.value_projection = nn.Parameter(torch.empty(rank, max_seq_len))
        nn.init.xavier_uniform_(self.key_projection)
        nn.init.xavier_uniform_(self.value_projection)

    def forward(self, query, key, value):
        if query.ndim < 2 or key.ndim != query.ndim or value.ndim != key.ndim:
            raise ValueError("query, key, and value must have the same rank of at least two")
        if query.shape[:-2] != key.shape[:-2] or key.shape[:-2] != value.shape[:-2]:
            raise ValueError("query, key, and value leading dimensions must match")
        if query.shape[-1] != key.shape[-1]:
            raise ValueError("query and key feature dimensions must match")
        if key.shape[-2] != value.shape[-2]:
            raise ValueError("key and value sequence lengths must match")
        sequence_length = key.shape[-2]
        if sequence_length > self.max_seq_len:
            raise ValueError("key/value sequence length exceeds max_seq_len")
        low_key = torch.einsum("rn,...nd->...rd", self.key_projection[:, :sequence_length], key)
        low_value = torch.einsum("rn,...nd->...rd", self.value_projection[:, :sequence_length], value)
        weights = torch.softmax(query @ low_key.transpose(-1, -2) / math.sqrt(query.shape[-1]), dim=-1)
        return weights @ low_value
