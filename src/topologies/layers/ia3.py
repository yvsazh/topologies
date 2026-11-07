import math

import torch
from torch import nn


class IA3(nn.Module):
    component_id = "ia3"
    PRESETS = {
        "tiny": {"key_dim": 4, "value_dim": 4, "ffn_dim": 8},
        "small": {"key_dim": 16, "value_dim": 16, "ffn_dim": 32},
        "base": {"key_dim": 64, "value_dim": 64, "ffn_dim": 128},
    }

    def __init__(self, key_dim, value_dim, ffn_dim):
        super().__init__()
        for name, dimension in (("key_dim", key_dim), ("value_dim", value_dim), ("ffn_dim", ffn_dim)):
            if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.ffn_dim = ffn_dim
        self.key_scale = nn.Parameter(torch.ones(key_dim))
        self.value_scale = nn.Parameter(torch.ones(value_dim))
        self.ffn_scale = nn.Parameter(torch.ones(ffn_dim))

    @staticmethod
    def _validate_attention_tensor(name, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim < 2:
            raise ValueError(f"{name} must have rank at least two")
        if any(size <= 0 for size in tensor.shape):
            raise ValueError(f"{name} dimensions must be positive")

    def attend(self, queries, keys, values):
        self._validate_attention_tensor("queries", queries)
        self._validate_attention_tensor("keys", keys)
        self._validate_attention_tensor("values", values)
        if queries.shape[-1] != self.key_dim or keys.shape[-1] != self.key_dim:
            raise ValueError(f"queries and keys must have key feature dimension {self.key_dim}")
        if values.shape[-1] != self.value_dim:
            raise ValueError(f"values must have value feature dimension {self.value_dim}")
        if keys.shape[-2] != values.shape[-2]:
            raise ValueError("keys and values must have the same sequence length")
        try:
            torch.broadcast_shapes(queries.shape[:-2], keys.shape[:-2], values.shape[:-2])
        except (RuntimeError, ValueError):
            raise ValueError("queries, keys, and values must have broadcastable batch dimensions") from None
        scaled_keys = keys * self.key_scale
        logits = torch.matmul(queries, scaled_keys.transpose(-2, -1)) / math.sqrt(self.key_dim)
        weights = torch.softmax(logits, dim=-1)
        return torch.matmul(weights, values * self.value_scale)

    def scale_hidden(self, hidden):
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("hidden must be a torch.Tensor")
        if hidden.ndim < 1:
            raise ValueError("hidden must have at least one dimension")
        if hidden.shape[-1] != self.ffn_dim:
            raise ValueError(f"hidden final dimension must be {self.ffn_dim}")
        return hidden * self.ffn_scale

    def forward(self, queries, keys, values, hidden):
        attention = self.attend(queries, keys, values)
        scaled_hidden = self.scale_hidden(hidden)
        if attention.shape[:-1] != scaled_hidden.shape[:-1]:
            raise ValueError("attention and hidden outputs must have matching leading dimensions")
        return torch.cat((attention, scaled_hidden), dim=-1)
