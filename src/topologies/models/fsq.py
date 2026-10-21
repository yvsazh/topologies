import torch
from torch import nn


class FSQ(nn.Module):
    def __init__(self, levels, bound=1.0):
        super().__init__()
        if not levels or any(not isinstance(level, int) or level < 2 for level in levels):
            raise ValueError("levels must contain integers of at least 2")
        if bound <= 0:
            raise ValueError("bound must be positive")
        self.levels = tuple(levels)
        self.bound = float(bound)

    def forward(self, values):
        if values.shape[-1] != len(self.levels):
            raise ValueError("last input dimension must match levels")
        levels = values.new_tensor(self.levels)
        clipped = values.clamp(-self.bound, self.bound)
        indices = ((clipped + self.bound) * (levels - 1) / (2 * self.bound)).round()
        quantized = indices * (2 * self.bound) / (levels - 1) - self.bound
        return values + (quantized - values).detach()
