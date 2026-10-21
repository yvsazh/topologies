import torch
from torch import nn
import math


def depth_mup(values, depth):
    values = torch.as_tensor(values, dtype=torch.float32)
    depth = torch.as_tensor(depth, dtype=torch.float32)
    return values * torch.rsqrt(depth)


class DepthMup(nn.Module):
    component_id = "depth-mu-p"
    PRESETS = {
        "tiny": {"n_mels": 8, "mode": "scale"},
        "base": {"n_mels": 16, "mode": "hold"},
    }

    def __init__(self, n_mels=8, mode="scale", dropout=0.0, vocab_size=8, context_length=2, depth=4.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"scale", "hold"}:
            raise ValueError("unknown mode")
        if n_mels <= 0 or n_mels % 2:
            raise ValueError("invalid width")
        if vocab_size <= 0 or context_length <= 0:
            raise ValueError("invalid sequence size")
        if not math.isfinite(depth) or depth <= 0:
            raise ValueError("depth must be finite and positive")
        self.n_mels = n_mels
        self.mode = mode
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.depth = float(depth)

    def forward(self, values):
        values = torch.as_tensor(values, dtype=torch.float32)
        if values.ndim != 3 or values.shape[-1] != self.n_mels:
            raise ValueError("values must have shape (batch, time, width)")
        if values.shape[-2] < 2:
            raise ValueError("time must be at least 2")
        if self.mode == "hold":
            return values
        return values * (self.depth ** -0.5)
