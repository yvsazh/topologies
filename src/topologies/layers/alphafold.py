import torch
from torch import nn


class IPAPointDistanceBias(nn.Module):
    component_id = "alphafold"

    def __init__(self, num_heads=2, initial_gamma=1.0):
        super().__init__()
        if num_heads <= 0 or initial_gamma <= 0:
            raise ValueError("num_heads and initial_gamma must be positive")
        self.num_heads = num_heads
        gamma = torch.full((num_heads,), float(initial_gamma))
        self.raw_gamma = nn.Parameter(torch.log(torch.expm1(gamma)))

    def forward(self, query_points, key_points):
        if query_points.ndim != 5 or key_points.ndim != 5:
            raise ValueError("query_points and key_points must have shape (batch, length, heads, points, 3)")
        if query_points.shape[0] != key_points.shape[0]:
            raise ValueError("query and key point batches must match")
        if query_points.shape[2:] != key_points.shape[2:] or query_points.shape[-1] != 3:
            raise ValueError("query and key heads, point counts, and xyz dimensions must match")

        difference = query_points[:, :, None] - key_points[:, None, :]
        squared_distances = difference.square().sum(dim=-1).sum(dim=-1)
        gamma = torch.nn.functional.softplus(self.raw_gamma)
        return -0.5 * gamma.view(1, 1, 1, self.num_heads) * squared_distances
