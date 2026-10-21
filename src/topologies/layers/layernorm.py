import math

import torch


class LayerNorm(torch.nn.Module):
    component_id = "layernorm"
    def __init__(self, eps=1e-5):
        super().__init__()
        if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a finite positive number")
        self.eps = float(eps)

    def forward(self, x):
        if x.ndim == 0 or x.shape[-1] == 0:
            raise ValueError("input must have a non-empty last axis")
        mean = x.mean(dim=-1, keepdim=True)
        deviations = x - mean
        variance = deviations.square().mean(dim=-1, keepdim=True)
        return deviations / torch.sqrt(variance + self.eps)
