import math

import torch
from torch import nn


class MixStyle(nn.Module):
    component_id = "mixstyle"

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        for name, value in (("p", p), ("alpha", alpha), ("eps", eps)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                error = ValueError(f"{name} must be a finite number")
                raise error
            try:
                finite = math.isfinite(float(value))
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x):
        if not torch.is_tensor(x) or x.ndim != 4:
            raise ValueError("MixStyle expects a tensor with shape (B, C, H, W)")
        if not x.is_floating_point():
            raise ValueError("MixStyle expects a floating-point input tensor")
        if any(size < 1 for size in x.shape):
            raise ValueError("MixStyle input dimensions must be non-empty")
        if not self.training or self.p == 0 or x.shape[0] < 2:
            return x
        if torch.rand((), device=x.device) >= self.p:
            return x
        mean = x.mean(dim=(2, 3), keepdim=True).detach()
        variance = x.detach().var(dim=(2, 3), keepdim=True, unbiased=False)
        std = (variance + self.eps).sqrt().detach()
        normalized = (x - mean) / std
        concentration = torch.tensor(self.alpha, device=x.device, dtype=torch.float32)
        coefficient = torch.distributions.Beta(concentration, concentration).sample((x.shape[0],))
        coefficient = coefficient.to(dtype=x.dtype).view(-1, 1, 1, 1)
        permutation = torch.randperm(x.shape[0], device=x.device)
        mixed_mean = coefficient * mean + (1 - coefficient) * mean[permutation]
        mixed_std = coefficient * std + (1 - coefficient) * std[permutation]
        return normalized * mixed_std + mixed_mean
