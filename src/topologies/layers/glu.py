import torch
from torch import nn


class GLU(nn.Module):
    component_id = "glu"

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise ValueError("GLU does not accept constructor arguments")
        super().__init__()

    def forward(self, x):
        x = torch.as_tensor(x)
        if x.ndim == 0:
            raise ValueError("input must have a last dimension")
        if x.shape[-1] <= 0:
            raise ValueError("last dimension must be a positive even integer")
        if x.shape[-1] % 2:
            raise ValueError("last dimension must be even")
        value, gate = x.chunk(2, dim=-1)
        return value * torch.sigmoid(gate)
