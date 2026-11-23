import math

import torch
from torch import nn


class VREx(nn.Module):
    component_id = "vrex"

    def __init__(self, beta):
        super().__init__()
        try:
            beta = float(beta)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("beta must be a finite non-negative real number")
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be a finite non-negative real number")
        self.beta = beta

    def forward(self, risks):
        if (not isinstance(risks, torch.Tensor) or risks.ndim not in (1, 2)
                or risks.shape[-1] == 0 or (risks.ndim == 2 and risks.shape[0] == 0)):
            raise ValueError("risks must be a non-empty rank-1 tensor or batch of rank-1 tensors")
        if not risks.is_floating_point():
            raise ValueError("risks must be a floating-point tensor")
        return risks.sum(dim=-1) + self.beta * risks.var(dim=-1, unbiased=False)
