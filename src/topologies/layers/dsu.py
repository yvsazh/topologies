import math

import torch
from torch import nn


class DSU(nn.Module):
    component_id = "dsu"

    def __init__(self, p=0.5):
        super().__init__()
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            error = ValueError("p must be a finite number in [0, 1]")
            raise error
        try:
            finite = math.isfinite(float(p))
        except OverflowError:
            finite = False
        if not finite or not 0 <= p <= 1:
            raise ValueError("p must be a finite number in [0, 1]")
        self.p = float(p)

    @staticmethod
    def _sqrt_nonnegative(value):
        positive = value > 0
        safe = torch.where(positive, value, torch.ones_like(value))
        return torch.where(positive, safe.sqrt(), torch.zeros_like(value))

    def forward(self, x):
        if not torch.is_tensor(x) or x.ndim != 4:
            raise ValueError("DSU expects a tensor with NCHW shape (B, C, H, W)")
        if not x.is_floating_point():
            raise ValueError("DSU expects a floating-point input tensor")
        if any(size < 1 for size in x.shape):
            raise ValueError("DSU input dimensions must be non-empty")
        if not self.training or self.p == 0:
            return x
        if self.p < 1 and torch.rand((), device=x.device) >= self.p:
            return x
        mean = x.mean(dim=(2, 3), keepdim=True)
        variance = (x - mean).square().mean(dim=(2, 3), keepdim=True)
        std = self._sqrt_nonnegative(variance)
        mean_centered = mean - mean.mean(dim=0, keepdim=True)
        std_centered = std - std.mean(dim=0, keepdim=True)
        mean_scope = self._sqrt_nonnegative(mean_centered.square().mean(dim=0, keepdim=True))
        std_scope = self._sqrt_nonnegative(std_centered.square().mean(dim=0, keepdim=True))
        beta = mean + torch.randn_like(mean) * mean_scope
        gamma = std + torch.randn_like(std) * std_scope
        nonzero_std = std > 0
        safe_std = torch.where(nonzero_std, std, torch.ones_like(std))
        normalized = torch.where(nonzero_std, (x - mean) / safe_std, torch.zeros_like(x))
        return gamma * normalized + beta
