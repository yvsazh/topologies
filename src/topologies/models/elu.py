import math

import torch
from torch import nn


class ELU(nn.Module):
    architecture_id = "elu"
    PRESETS = {"tiny": {"alpha": 1.0}, "small": {"alpha": 1.0}}

    def __init__(self, alpha: float = 1.0):
        if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
                or not math.isfinite(float(alpha)) or float(alpha) <= 0):
            raise ValueError("alpha must be a positive finite real number")
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x):
        if not torch.is_tensor(x):
            raise ValueError("ELU input must be a torch.Tensor")
        if not x.is_floating_point():
            raise ValueError("ELU input must have a floating-point dtype")
        negative = self.alpha * torch.expm1(x.clamp_max(0))
        return torch.where(x > 0, x, negative)
