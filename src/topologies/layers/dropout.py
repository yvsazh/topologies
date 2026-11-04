import math

import torch
from torch import nn


class Dropout(nn.Module):
    component_id = "dropout"

    def __init__(self, p=0.5):
        super().__init__()
        if (
            isinstance(p, bool)
            or not isinstance(p, (int, float))
            or p < 0
            or p > 1
            or not math.isfinite(p)
        ):
            raise ValueError("p must be a finite number in [0, 1]")
        self.p = float(p)

    def forward(self, x):
        if not self.training:
            return x * (1.0 - self.p)
        if self.p == 0.0:
            return x
        if self.p == 1.0:
            return torch.zeros_like(x)
        keep_probabilities = torch.full_like(x, 1.0 - self.p)
        return x * torch.bernoulli(keep_probabilities)
