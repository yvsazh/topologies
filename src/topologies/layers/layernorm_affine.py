

from __future__ import annotations

import torch.nn as nn


class LayerNormAffine(nn.Module):


    component_id = "layernorm-affine"

    def __init__(self, dim: int = 32, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)

    def forward(self, x):
        return self.norm(x)
