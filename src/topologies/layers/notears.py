import torch
from torch import nn


class NOTEARS(nn.Module):
    component_id = "notears"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if not torch.is_tensor(x):
            raise ValueError("W must be a torch.Tensor.")
        if x.ndim < 2:
            raise ValueError("W must have at least two dimensions with square matrices.")
        if x.shape[-2] != x.shape[-1]:
            raise ValueError("W must be square.")
        if x.shape[-1] == 0:
            raise ValueError("W must have at least one node.")
        if not torch.is_floating_point(x):
            raise ValueError("W must contain real floating-point values.")
        exponential = torch.matrix_exp(x * x)
        return exponential.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - x.shape[-1]
