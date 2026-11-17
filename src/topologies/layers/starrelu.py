import math
import numbers

import torch
from torch import nn


def _is_finite_real(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, numbers.Real)
        and math.isfinite(float(value))
    )


class StarReLU(nn.Module):
    component_id = "starrelu"

    def __init__(self, scale=1.0, bias=0.0, *args, **kwargs):
        if args:
            raise ValueError("StarReLU accepts only scale and bias")
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise ValueError(f"StarReLU received unexpected arguments: {names}")
        for name, value in (("scale", scale), ("bias", bias)):
            if not _is_finite_real(value):
                raise ValueError(f"{name} must be a finite real number")
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.bias = nn.Parameter(torch.tensor(float(bias)))

    def forward(self, x):
        if not torch.is_tensor(x):
            raise ValueError("StarReLU input must be a torch.Tensor")
        if not x.is_floating_point():
            raise ValueError("StarReLU input must have a floating-point dtype")
        return self.scale * torch.relu(x).square() + self.bias
