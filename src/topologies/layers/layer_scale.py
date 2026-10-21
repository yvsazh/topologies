import math

import torch
from torch import nn


class LayerScale(nn.Module):
    component_id = "layer-scale"
    def __init__(self, dim=4, init_value=1e-5):
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if not isinstance(init_value, (int, float)) or not math.isfinite(init_value):
            raise ValueError("init_value must be a finite number")
        self.gamma = nn.Parameter(torch.full((dim,), float(init_value)))

    def forward(self, x, branch_output):
        if x.shape != branch_output.shape or x.shape[-1] != self.gamma.numel():
            raise ValueError("x and branch_output must have matching shapes and final dim")
        return x + self.gamma * branch_output
