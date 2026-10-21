import torch
from torch import nn


class LowRankGradient(nn.Module):
    component_id = "galore"
    def __init__(self, rank):
        super().__init__()
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError("rank must be a positive integer")
        self.rank = rank

    def forward(self, gradient):
        if gradient.ndim < 2:
            raise ValueError("gradient must have at least two dimensions")
        u, _, _ = torch.linalg.svd(gradient, full_matrices=False)
        basis = u[..., :, : min(self.rank, u.shape[-1])]
        return basis @ (basis.transpose(-1, -2) @ gradient)
