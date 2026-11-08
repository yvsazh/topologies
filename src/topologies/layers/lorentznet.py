import torch
from torch import nn


class LorentzNet(nn.Module):
    component_id = "lorentznet"

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise ValueError("LorentzNet accepts no constructor arguments")
        super().__init__()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("beta must be a torch.Tensor")
        if not x.is_floating_point():
            raise ValueError("beta must have a floating-point dtype")
        if not torch.isfinite(x).all():
            raise ValueError("beta values must be finite and satisfy 0 <= beta < 1")
        if ((x < 0) | (x >= 1)).any():
            raise ValueError("beta values must satisfy 0 <= beta < 1")
        return torch.rsqrt(1 - x.square())
