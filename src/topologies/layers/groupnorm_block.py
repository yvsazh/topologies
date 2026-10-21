

from __future__ import annotations

import torch
import torch.nn as nn


class GroupNormBlock(nn.Module):


    component_id = "groupnorm-block"

    def __init__(self, channels: int = 16, groups: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))
