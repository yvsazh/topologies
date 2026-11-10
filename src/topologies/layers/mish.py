import torch
from torch import nn
from torch.nn import functional as F


class Mish(nn.Module):
    component_id = "mish"

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise ValueError("Mish accepts no constructor arguments")
        super().__init__()

    def forward(self, x):
        if not torch.is_tensor(x):
            raise ValueError("Mish input must be a torch.Tensor")
        if not x.is_floating_point():
            raise ValueError("Mish input must have a floating-point dtype")
        return x * torch.tanh(F.softplus(x, beta=1, threshold=20))
