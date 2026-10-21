

from __future__ import annotations

import torch
import torch.nn as nn


class GeGLUFFN(nn.Module):


    component_id = "geglu-ffn"

    def __init__(self, dim: int = 32, hidden: int = 64) -> None:
        super().__init__()
        self.w = nn.Linear(dim, hidden, bias=False)
        self.v = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.gelu(self.w(x)) * self.v(x))
