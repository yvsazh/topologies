import torch
from torch import nn


class SELU(nn.Module):
    component_id = "selu"

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise ValueError("SELU accepts no constructor arguments")
        super().__init__()

    def forward(self, x):
        if not torch.is_tensor(x):
            raise ValueError("SELU input must be a torch.Tensor")
        if not x.is_floating_point():
            raise ValueError("SELU input must have a floating-point dtype")
        positive = 1.0507009873554805 * x
        negative = 1.0507009873554805 * 1.6732632423543772 * torch.expm1(x.clamp_max(0))
        return torch.where(x > 0, positive, negative)
