import torch
from torch import nn


class SkipInit(nn.Module):
    component_id = "skipinit"
    def __init__(self, dim=4):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.linear = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        if x.ndim == 0 or x.shape[-1] != self.dim:
            raise ValueError("input's last dimension must equal dim")
        return x + self.alpha * self.linear(x)
