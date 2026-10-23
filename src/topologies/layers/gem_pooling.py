import math
import numbers

import torch
from torch import nn


class GeMPooling(nn.Module):
    component_id = "gem-pooling"
    PRESETS = {
        "tiny": {"p": 3.0, "eps": 1e-6, "learnable": True},
        "small": {"p": [3.0] * 8, "eps": 1e-6, "learnable": True},
        "base": {"p": [3.0] * 512, "eps": 1e-6, "learnable": True},
    }

    def __init__(self, p=3.0, eps=1e-6, learnable=True):
        super().__init__()
        if isinstance(eps, bool) or not isinstance(eps, numbers.Real) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a finite positive number")
        if not isinstance(learnable, bool):
            error_type = ValueError
            raise error_type("learnable must be a bool")
        if torch.is_tensor(p):
            if p.ndim > 1 or p.numel() == 0 or p.dtype == torch.bool or p.is_complex():
                raise ValueError("p must be a positive scalar or one-dimensional sequence")
            values = p.detach().to(device="cpu", dtype=torch.get_default_dtype()).reshape(-1)
        elif isinstance(p, numbers.Real) and not isinstance(p, bool):
            values = torch.tensor([p], dtype=torch.get_default_dtype())
        elif isinstance(p, (list, tuple)) and p and all(
            isinstance(value, numbers.Real) and not isinstance(value, bool) for value in p
        ):
            values = torch.tensor(p, dtype=torch.get_default_dtype())
        else:
            raise ValueError("p must be a positive scalar or one-dimensional sequence")
        if not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
            raise ValueError("p values must be finite and positive")
        if learnable:
            self.p = nn.Parameter(values)
        else:
            self.register_buffer("p", values)
        self.eps = float(eps)

    def forward(self, x):
        if not torch.is_tensor(x) or x.ndim != 4:
            raise ValueError("x must be a 4D [B,C,H,W] tensor")
        if any(size < 1 for size in x.shape):
            raise ValueError("x must have nonempty batch, channel, and spatial dimensions")
        if self.p.numel() != 1 and self.p.numel() != x.shape[1]:
            raise ValueError("p must have one value per input channels")
        x = x.clamp_min(self.eps)
        p = self.p.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        log_mean_power = torch.logsumexp(x.log() * p, dim=(-2, -1), keepdim=True)
        log_mean_power = log_mean_power - math.log(x.shape[-2] * x.shape[-1])
        return (log_mean_power / p).exp()
