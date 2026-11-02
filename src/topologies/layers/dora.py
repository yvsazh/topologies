import math

import torch
from torch import nn


class DoRALinear(nn.Module):
    component_id = "dora"
    PRESETS = {
        "tiny": {"in_features": 2, "out_features": 1, "rank": 1, "bias": False},
        "small": {"in_features": 16, "out_features": 8, "rank": 2, "bias": False},
    }

    def __init__(self, in_features, out_features, rank=1, bias=False, base_weight=None):
        super().__init__()
        if type(in_features) is not int or in_features <= 0:
            raise ValueError("in_features must be a positive integer")
        if type(out_features) is not int or out_features <= 0:
            raise ValueError("out_features must be a positive integer")
        if type(rank) is not int or rank <= 0 or rank > min(in_features, out_features):
            raise ValueError("rank must be a positive integer no greater than min(in_features, out_features)")
        if type(bias) is not bool:
            raise ValueError("bias must be a boolean")
        if base_weight is not None:
            if not isinstance(base_weight, torch.Tensor):
                raise ValueError("base_weight must be a floating-point tensor")
            if tuple(base_weight.shape) != (out_features, in_features):
                raise ValueError("base_weight must have shape [out_features, in_features]")
            if not base_weight.is_floating_point() or not torch.isfinite(base_weight).all():
                raise ValueError("base_weight must contain only finite floating-point values")
            if (torch.linalg.vector_norm(base_weight, dim=1) == 0).any():
                raise ValueError("each base_weight row must have nonzero norm")
            factory = {"device": base_weight.device, "dtype": base_weight.dtype}
            weight = base_weight.detach().clone()
        else:
            factory = {}
            weight = torch.empty(out_features, in_features, **factory)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.base_weight = nn.Parameter(weight, requires_grad=False)
        self.A = nn.Parameter(torch.empty(rank, in_features, **factory))
        self.B = nn.Parameter(torch.zeros(out_features, rank, **factory))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.m = nn.Parameter(torch.linalg.vector_norm(self.base_weight, dim=1).detach().clone())
        self.bias = nn.Parameter(torch.zeros(out_features, **factory)) if bias else None

    @property
    def effective_weight(self):
        direction = self.base_weight + self.B @ self.A
        normalizer = torch.linalg.vector_norm(direction, dim=1, keepdim=True).detach()
        return self.m.unsqueeze(1) * direction / normalizer

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim == 0:
            raise ValueError("input must be a tensor with a final feature dimension")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"input final dimension must equal in_features ({self.in_features})")
        if not x.is_floating_point():
            raise ValueError("input must be a floating-point tensor")
        return torch.nn.functional.linear(x, self.effective_weight, self.bias)
