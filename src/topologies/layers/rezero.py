from numbers import Integral

import torch
from torch import nn


class ReZero(nn.Module):
    component_id = "rezero"
    def __init__(self, dim=4):
        super().__init__()
        if isinstance(dim, bool) or not isinstance(dim, Integral) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = int(dim)
        self.branch = nn.Linear(self.dim, self.dim)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim == 0 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have last dimension {self.dim}")
        return x + self.alpha * self.branch(x)
