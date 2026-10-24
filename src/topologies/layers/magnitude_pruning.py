import math
import numbers

import torch
from torch import nn


class MagnitudePruning(nn.Module):
    component_id = "magnitude-pruning"

    def __init__(self, quality):
        super().__init__()
        try:
            if isinstance(quality, bool) or not isinstance(quality, numbers.Real):
                raise TypeError
            quality = float(quality)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("quality must be a finite non-negative real number") from error
        if not math.isfinite(quality) or quality < 0:
            raise ValueError("quality must be a finite non-negative real number")
        self.quality = quality
        self.register_buffer("mask", None)

    def forward(self, weights):
        if not isinstance(weights, torch.Tensor):
            raise TypeError("weights must be a floating-point tensor")
        if weights.ndim < 1 or weights.numel() == 0:
            raise ValueError("weights must be a non-empty tensor with rank at least one")
        if not weights.is_floating_point():
            raise ValueError("weights must be a floating-point tensor")
        if self.mask is None or self.mask.shape != weights.shape:
            with torch.no_grad():
                threshold = self.quality * weights.detach().std(correction=0)
                self.mask = weights.detach().abs() >= threshold
        mask = self.mask.to(device=weights.device, dtype=weights.dtype)
        return weights * mask
