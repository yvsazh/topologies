import math
from numbers import Real

import torch
from torch import nn


class FIRE(nn.Module):
    component_id = "fire"

    def __init__(self, num_heads=12, mlp_width=32, init_c=0.1, init_L=512.0, eps=1e-6):
        super().__init__()
        if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if isinstance(mlp_width, bool) or not isinstance(mlp_width, int) or mlp_width <= 0:
            raise ValueError("mlp_width must be a positive integer")
        for name, value in (("init_c", init_c), ("init_L", init_L), ("eps", eps)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite scalar")

        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, num_heads),
        )
        self.c = nn.Parameter(torch.tensor(float(init_c)))
        self.init_L = nn.Parameter(torch.tensor(float(init_L)), requires_grad=False)
        self.L_multiplier = nn.Parameter(torch.tensor(1.0))
        self.eps = float(eps)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("x must be rank 4 with shape [batch, heads, sequence, features]")
        if x.size(1) != self.num_heads:
            raise ValueError(f"x head dimension must equal num_heads ({self.num_heads})")
        seq_length = x.size(2)
        if seq_length <= 0:
            raise ValueError("x sequence length must be positive")

        positions = torch.arange(seq_length, dtype=torch.float, device=x.device)
        rel_distance = positions[:, None] - positions[None, :]
        threshold = torch.abs(self.L_multiplier * self.init_L)
        pos_normalizer = torch.maximum(positions, threshold)[:, None]
        rel_distance = torch.log(torch.abs(self.c * rel_distance) + 1)
        pos_normalizer = torch.log(torch.abs(self.c * pos_normalizer) + 1) + self.eps
        normalized_distance = rel_distance / pos_normalizer
        fire_bias = self.mlp(normalized_distance.unsqueeze(-1))
        return fire_bias.unsqueeze(0).permute(0, 3, 1, 2).expand(x.size(0), -1, -1, -1)
