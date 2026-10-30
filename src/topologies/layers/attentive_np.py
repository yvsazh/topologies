import math

import torch
from torch import nn


class AttentiveNP(nn.Module):
    component_id = "attentive-np"

    def __init__(self, *args, **kwargs):
        super().__init__()
        if args or kwargs:
            raise ValueError("AttentiveNP accepts no constructor arguments")

    def forward(self, queries, keys, values):
        tensors = (queries, keys, values)
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError("queries, keys, and values must be torch tensors")
        if any(tensor.ndim != 3 for tensor in tensors):
            raise ValueError("queries, keys, and values must have shape [B, sequence, feature]")
        if any(dimension == 0 for tensor in tensors for dimension in tensor.shape):
            raise ValueError("attention tensors must have nonempty batch, sequence, and feature dimensions")
        if queries.shape[0] != keys.shape[0] or keys.shape[0] != values.shape[0]:
            raise ValueError("queries, keys, and values must have matching batch sizes")
        if queries.shape[-1] != keys.shape[-1]:
            raise ValueError("query and key feature widths must match")
        if keys.shape[1] != values.shape[1]:
            raise ValueError("key and value sequence lengths must match")
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise ValueError("queries, keys, and values must use floating-point dtypes")
        if any(tensor.dtype != queries.dtype for tensor in tensors[1:]):
            raise ValueError("queries, keys, and values must use the same dtype")
        if any(tensor.device != queries.device for tensor in tensors[1:]):
            raise ValueError("queries, keys, and values must be on the same device")

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(queries.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, values)
