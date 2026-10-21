

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class QKNormAttention(nn.Module):
    component_id = "qk-norm-attention"

    def __init__(
        self,
        dim: int = 32,
        heads: int = 4,
        reference_length_97_5: int | None = None,
        scale_init: float | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads != 0:
            raise ValueError("dim and heads must be positive and dim must be divisible by heads")
        if reference_length_97_5 is not None and reference_length_97_5 < 2:
            raise ValueError("reference_length_97_5 must be at least 2")
        if scale_init is None:
            if reference_length_97_5 is None:
                raise ValueError(
                    "provide the training-set 97.5th-percentile sequence length "
                    "or an explicit scale_init"
                )
            length = reference_length_97_5
            scale_init = math.log2(length * length - length)
        if not math.isfinite(scale_init) or scale_init <= 0:
            raise ValueError("scale_init must be finite and positive")
        self.heads = heads
        self.hd = dim // heads
        self.reference_length_97_5 = reference_length_97_5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.heads * self.hd:
            raise ValueError("x must have shape (batch, sequence, dim)")
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def sp(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.heads, self.hd).transpose(1, 2)

        q = F.normalize(sp(q), p=2, dim=-1)
        k = F.normalize(sp(k), p=2, dim=-1)
        v = sp(v)
        s = self.scale * (q @ k.transpose(-2, -1))
        a = torch.softmax(s, dim=-1) @ v
        return self.out(a.transpose(1, 2).contiguous().view(B, T, D))
