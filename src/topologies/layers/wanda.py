import torch
from torch import nn


class Wanda(nn.Module):
    component_id = "wanda"

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise ValueError("Wanda takes no constructor arguments")
        super().__init__()

    def forward(self, weight, activation_norm):
        if not isinstance(weight, torch.Tensor):
            raise TypeError("weight must be a torch.Tensor")
        if not isinstance(activation_norm, torch.Tensor):
            raise TypeError("activation_norm must be a torch.Tensor")
        if weight.ndim != 2:
            raise ValueError("weight must be a two-dimensional matrix")
        if any(size == 0 for size in weight.shape):
            raise ValueError("weight dimensions must be non-empty")
        if activation_norm.device != weight.device:
            raise ValueError("weight and activation_norm must be on the same device")
        if activation_norm.ndim == 1:
            if activation_norm.shape[0] != weight.shape[1]:
                raise ValueError("activation_norm vector length must match weight input features")
        elif activation_norm.ndim == 2:
            if activation_norm.shape != weight.shape:
                raise ValueError("two-dimensional activation_norm must match the weight shape")
        else:
            raise ValueError("activation_norm must be a vector or match the weight shape")
        return weight.abs() * activation_norm
