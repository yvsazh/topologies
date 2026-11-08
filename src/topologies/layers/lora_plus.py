import math
import sys

import torch
from torch import nn
from torch.nn import functional as F


class LoRAPlus(nn.Module):
    component_id = "lora-plus"
    PRESETS = {
        "tiny": {"in_features": 4, "out_features": 3, "rank": 1, "alpha": 1.0, "ratio": 8.0},
        "small": {"in_features": 32, "out_features": 32, "rank": 4, "alpha": 8.0, "ratio": 8.0},
        "base": {"in_features": 128, "out_features": 128, "rank": 8, "alpha": 16.0, "ratio": 8.0},
    }

    def __init__(self, in_features, out_features, rank, alpha=1.0, ratio=8.0):
        super().__init__()
        for name, value in (("in_features", in_features), ("out_features", out_features), ("rank", rank)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if rank > min(in_features, out_features):
            raise ValueError("rank must not exceed min(in_features, out_features)")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or alpha <= 0
            or (isinstance(alpha, int) and alpha > sys.float_info.max)
        ):
            raise ValueError("alpha must be a positive finite number")
        if not math.isfinite(alpha):
            raise ValueError("alpha must be a positive finite number")
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or ratio <= 1
            or (isinstance(ratio, int) and ratio > sys.float_info.max)
        ):
            raise ValueError("ratio must be a finite number greater than 1")
        if not math.isfinite(ratio):
            raise ValueError("ratio must be a finite number greater than 1")
        alpha = float(alpha)
        ratio = float(ratio)
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.ratio = ratio
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.A, mean=0.0, std=1.0 / math.sqrt(in_features))

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim == 0 or not x.is_floating_point():
            raise ValueError("x must be a floating-point tensor with a final feature dimension")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"x must have a final dimension of {self.in_features}")
        base = F.linear(x, self.weight)
        update = F.linear(F.linear(x, self.A), self.B)
        return base + (self.alpha / self.rank) * update

    def optimizer_groups(self, learning_rate):
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or learning_rate <= 0
            or (isinstance(learning_rate, int) and learning_rate > sys.float_info.max)
        ):
            raise ValueError("learning_rate must be a positive finite number")
        if not math.isfinite(learning_rate):
            raise ValueError("learning_rate must be a positive finite number")
        learning_rate = float(learning_rate)
        return [
            {"params": [self.A], "lr": learning_rate},
            {"params": [self.B], "lr": learning_rate * self.ratio},
        ]
