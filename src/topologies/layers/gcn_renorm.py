import torch
from torch import nn


class GCNRenorm(nn.Module):
    component_id = "gcn-renorm"

    def __init__(self):
        super().__init__()

    def forward(self, adjacency, features=None):
        if not isinstance(adjacency, torch.Tensor):
            raise TypeError("adjacency must be a torch.Tensor")
        if adjacency.ndim < 2:
            raise ValueError("adjacency must have at least two dimensions")
        if adjacency.shape[-2] != adjacency.shape[-1]:
            raise ValueError("adjacency must be square")
        if adjacency.shape[-1] == 0:
            raise ValueError("adjacency must contain at least one node")
        if not torch.is_floating_point(adjacency):
            raise ValueError("adjacency must have a real floating-point dtype")
        if not torch.isfinite(adjacency).all():
            raise ValueError("adjacency values must be finite")
        if (adjacency < 0).any():
            raise ValueError("adjacency values must be nonnegative")
        if not torch.allclose(adjacency, adjacency.transpose(-1, -2), rtol=1e-5, atol=1e-8):
            raise ValueError("adjacency must be symmetric")

        if features is not None:
            if not isinstance(features, torch.Tensor):
                raise TypeError("features must be a torch.Tensor")
            if features.ndim != adjacency.ndim:
                raise ValueError("features must have matching adjacency batch dimensions")
            if features.shape[-2] != adjacency.shape[-1]:
                raise ValueError("features must have the same node dimension as adjacency")
            if features.shape[:-2] != adjacency.shape[:-2]:
                raise ValueError("features must have matching adjacency batch dimensions")
            if not torch.is_floating_point(features):
                raise ValueError("features must have a real floating-point dtype")
            if features.dtype != adjacency.dtype or features.device != adjacency.device:
                raise ValueError("features and adjacency must have matching dtype and device")

        identity = torch.eye(adjacency.shape[-1], dtype=adjacency.dtype, device=adjacency.device)
        with_self_loops = adjacency + identity
        inverse_sqrt_degree = with_self_loops.sum(dim=-1).rsqrt()
        normalized = (
            inverse_sqrt_degree.unsqueeze(-1)
            * with_self_loops
            * inverse_sqrt_degree.unsqueeze(-2)
        )
        if features is None:
            return normalized
        return normalized @ features
