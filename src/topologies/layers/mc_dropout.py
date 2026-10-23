import math

import torch
from torch import nn


class MCDropout(nn.Module):
    component_id = "mc-dropout"
    PRESETS = {"tiny": {"keep_prob": 0.9}, "small": {"keep_prob": 0.8}}

    def __init__(self, keep_prob):
        super().__init__()
        try:
            probability = float(keep_prob)
        except (OverflowError, TypeError, ValueError):
            probability = math.inf
        if (
            isinstance(keep_prob, bool)
            or not isinstance(keep_prob, (int, float))
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise ValueError("keep_prob must be a finite number in [0, 1]")
        self.keep_prob = probability

    def forward(self, x):
        if self.keep_prob == 1.0:
            return x
        mask = torch.rand(x.shape, device=x.device) < self.keep_prob
        return x * mask.to(dtype=x.dtype)
