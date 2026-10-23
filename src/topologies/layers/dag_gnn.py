import torch
from torch import nn


class DAGGNN(nn.Module):
    component_id = "dag-gnn"

    def __init__(self, num_variables=2):
        super().__init__()
        if isinstance(num_variables, bool) or not isinstance(num_variables, int) or num_variables <= 0:
            raise ValueError("num_variables must be a positive integer")
        self.num_variables = num_variables

    def forward(self, noise, adjacency):
        if not isinstance(noise, torch.Tensor):
            raise TypeError("noise must be a tensor with shape [batch, num_variables]")
        if noise.ndim != 2:
            raise ValueError("noise must have shape [batch, num_variables]")
        if not isinstance(adjacency, torch.Tensor):
            raise TypeError("adjacency must be a square tensor")
        if adjacency.ndim == 2:
            is_square = adjacency.shape[0] == adjacency.shape[1]
        elif adjacency.ndim == 3:
            is_square = adjacency.shape[1] == adjacency.shape[2]
        else:
            is_square = False
        if not is_square:
            raise ValueError("adjacency must be a square tensor")
        if adjacency.shape[-2:] != (self.num_variables, self.num_variables):
            raise ValueError("adjacency dimensions must match num_variables")
        if noise.shape[1] != self.num_variables:
            raise ValueError("noise feature count must match num_variables")
        if adjacency.ndim == 3 and adjacency.shape[0] not in (1, noise.shape[0]):
            raise ValueError("adjacency batch must be 1 or match noise batch")
        if not noise.is_floating_point() or not adjacency.is_floating_point():
            raise ValueError("noise and adjacency must use floating-point dtypes")
        if noise.dtype != adjacency.dtype or noise.device != adjacency.device:
            raise ValueError("noise and adjacency must have matching dtype and device")
        identity = torch.eye(
            self.num_variables, dtype=adjacency.dtype, device=adjacency.device
        )
        return torch.linalg.solve(
            (identity - adjacency).transpose(-2, -1), noise.unsqueeze(-1)
        ).squeeze(-1)
