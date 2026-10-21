import torch
from torch import nn


def galerkin_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape (batch, heads, time, features)")
    if query.shape[:2] != key.shape[:2] or query.shape[2] != key.shape[2]:
        raise ValueError("query and key batch, head, and sequence dimensions must match")
    if key.shape[:3] != value.shape[:3]:
        raise ValueError("key and value batch, head, and sequence dimensions must match")
    return query @ (key.transpose(-1, -2) @ value) / key.shape[-2]


class GalerkinAttention(nn.Module):
    component_id = "galerkin-transformer"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return galerkin_attention(query, key, value)
