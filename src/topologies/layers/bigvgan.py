import math

import torch
from torch import nn


class SnakeActivation(nn.Module):
    component_id = "bigvgan"

    def __init__(self, channels, alpha=1.0):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha == 0.0:
            raise ValueError("alpha must be finite and non-zero")
        self.channels = channels
        self.alpha = nn.Parameter(torch.full((1, channels, 1), float(alpha)))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError("x must have shape (batch, channels, length)")
        return x + torch.sin(self.alpha * x).square() / self.alpha
