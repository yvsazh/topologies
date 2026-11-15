import math

import torch
from torch import nn


def _invalid_constructor_value(name):
    return ValueError(f"{name} must be a finite number")


class R2D2(nn.Module):
    component_id = "r2d2"

    def __init__(self, regularization: float, bias: float, scale: float = 10.0):
        super().__init__()
        for name, value in (("regularization", regularization), ("bias", bias), ("scale", scale)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _invalid_constructor_value(name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        inverse_softplus = regularization if regularization > 20 else math.log(
            math.expm1(regularization)
        )
        self.raw_regularization = nn.Parameter(torch.tensor(inverse_softplus))
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.bias = nn.Parameter(torch.tensor(float(bias)))

    @property
    def regularization(self):
        value = nn.functional.softplus(self.raw_regularization)
        return value + torch.finfo(value.dtype).tiny

    def forward(self, support_x, support_y, query_x):
        for name, value in (("support_x", support_x), ("support_y", support_y), ("query_x", query_x)):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.ndim != 3:
                raise ValueError(f"{name} must have shape [batch, sequence, features]")
            if not value.is_floating_point():
                raise ValueError(f"{name} must have a floating-point dtype")
        batch_size, support_size, embedding_size = support_x.shape
        if batch_size == 0 or support_size == 0 or embedding_size == 0:
            raise ValueError("support_x dimensions must be non-empty")
        if support_y.shape[0] != batch_size:
            raise ValueError("support_x and support_y batch dimensions must match")
        if support_y.shape[1] != support_size:
            raise ValueError("support_x and support_y support dimensions must match")
        if support_y.shape[2] == 0:
            raise ValueError("support_y output dimension must be non-empty")
        if query_x.shape[0] != batch_size:
            raise ValueError("support_x and query_x batch dimensions must match")
        if query_x.shape[1] == 0:
            raise ValueError("query_x query dimension must be non-empty")
        if query_x.shape[2] != embedding_size:
            raise ValueError("support_x and query_x embedding dimensions must match")
        tensors = (support_x, support_y, query_x)
        if any(value.device != support_x.device for value in tensors[1:]):
            raise ValueError("support_x, support_y, and query_x must be on the same device")
        if any(value.dtype != support_x.dtype for value in tensors[1:]):
            raise ValueError("support_x, support_y, and query_x must have the same dtype")
        identity = torch.eye(support_size, device=support_x.device, dtype=support_x.dtype)
        regularization = self.regularization.to(device=support_x.device, dtype=support_x.dtype)
        system = support_x @ support_x.transpose(-1, -2) + regularization * identity
        solved = torch.linalg.solve(system, support_y)
        weights = support_x.transpose(-1, -2) @ solved
        scale = self.scale.to(device=query_x.device, dtype=query_x.dtype)
        bias = self.bias.to(device=query_x.device, dtype=query_x.dtype)
        return scale * (query_x @ weights) + bias
