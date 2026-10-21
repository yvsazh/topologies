

from __future__ import annotations

from torch import nn


class SiLUActivation(nn.Module):
    component_id = "silu-activation"

    def __init__(self) -> None:
        super().__init__()
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x)
