import math

import torch
from torch import nn


class InformerQuerySparsity(nn.Module):
    component_id = "informer"

    def __init__(self, head_dim=8):
        super().__init__()
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        self.head_dim = head_dim

    def forward(self, query, keys):
        if query.ndim < 2 or keys.ndim < 2 or query.shape[-1] != self.head_dim or keys.shape[-1] != self.head_dim:
            raise ValueError("query and keys must have matching [..., sequence, head_dim] dimensions")
        if keys.shape[-2] == 0:
            raise ValueError("keys must contain at least one key")
        scaled_scores = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        return scaled_scores.max(dim=-1).values - scaled_scores.mean(dim=-1)
