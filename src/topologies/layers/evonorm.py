import math

import torch
from torch import nn


class EvoNormS0(nn.Module):
    component_id = "evonorm"

    def __init__(self, channels, groups=32, eps=1e-5):
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if not isinstance(groups, int) or isinstance(groups, bool) or groups <= 0:
            raise ValueError("groups must be a positive integer")
        if channels % groups != 0:
            raise ValueError("groups must divide channels")
        if not isinstance(eps, (int, float)) or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a positive finite number")
        self.channels = channels
        self.groups = groups
        self.eps = float(eps)
        self.v = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("x must have rank 4 in NCHW format")
        if x.shape[1] != self.channels:
            raise ValueError("input channels must match the configured channels")
        if any(size <= 0 for size in (x.shape[0], x.shape[2], x.shape[3])):
            raise ValueError("input batch and spatial dimensions must be positive")
        batch, _, height, width = x.shape
        grouped = x.reshape(batch, self.groups, self.channels // self.groups, height, width)
        variance = grouped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        denominator = torch.sqrt(variance + self.eps).expand_as(grouped).reshape_as(x)
        return x * torch.sigmoid(self.v * x) / denominator * self.gamma + self.beta
