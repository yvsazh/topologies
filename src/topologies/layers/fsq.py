import torch
from torch import nn


class FSQ(nn.Module):
    component_id = "fsq"

    def __init__(self, levels):
        super().__init__()
        if not levels or any(
            not isinstance(level, int) or isinstance(level, bool) or level < 2
            for level in levels
        ):
            raise ValueError("levels must contain integers of at least 2")
        self.levels = tuple(levels)

    def forward(self, values):
        if values.shape[-1] != len(self.levels):
            raise ValueError("last input dimension must match levels")
        levels = values.new_tensor(self.levels)
        bounded = torch.tanh(values)
        indices = ((bounded + 1.0) * (levels - 1) / 2.0).round()
        quantized = 2.0 * indices / (levels - 1) - 1.0
        return bounded + (quantized - bounded).detach()
