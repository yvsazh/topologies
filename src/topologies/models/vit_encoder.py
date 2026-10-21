

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GlobalTokens(nn.Module):
    def __init__(self, dim: int, slots: int) -> None:
        super().__init__()
        base_grid = math.isqrt(slots - 1)
        if base_grid * base_grid != slots - 1:
            raise ValueError("slots must contain one class token and a square patch grid")
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, slots, dim))
        self.base_grid = base_grid

    def forward(self, patches, grid_shape):
        batch, _, width = patches.shape
        patch_positions = self.pos[:, 1:].reshape(
            1, self.base_grid, self.base_grid, width
        ).permute(0, 3, 1, 2)
        if tuple(grid_shape) != (self.base_grid, self.base_grid):
            patch_positions = F.interpolate(
                patch_positions, size=grid_shape, mode="bicubic", align_corners=False
            )
        patch_positions = patch_positions.flatten(2).transpose(1, 2)
        positions = torch.cat((self.pos[:, :1], patch_positions), dim=1)
        tokens = torch.cat((self.cls.expand(batch, -1, -1), patches), dim=1)
        return tokens + positions


class ViTEncoder(nn.Module):


    architecture_id = "vit-encoder"

    def __init__(
        self, classes: int = 10, dim: int = 32, patch: int = 4, layers: int = 2, heads: int = 4
    ) -> None:
        super().__init__()
        if min(classes, dim, patch, layers, heads) <= 0:
            raise ValueError("classes, dim, patch, layers, and heads must be positive")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.patch = nn.Conv2d(3, dim, patch, stride=patch)
        self.tokens = _GlobalTokens(dim, 17)
        self.layers = nn.ModuleList([_E(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_grid = self.patch(x)
        h = patch_grid.flatten(2).transpose(1, 2)
        h = self.tokens(h, patch_grid.shape[-2:])
        for l in self.layers:
            h = l(h)
        return self.head(self.norm(h)[:, 0, :])


class _E(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.ao = nn.Linear(dim, dim)
        self.f1 = nn.Linear(dim, dim * 2)
        self.f2 = nn.Linear(dim * 2, dim)
        self.heads = heads
        self.hd = dim // heads

    def forward(self, x):
        B, T, D = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)

        def sp(t):
            return t.view(B, T, self.heads, self.hd).transpose(1, 2)

        q, k, v = sp(q), sp(k), sp(v)
        a = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.hd), dim=-1) @ v
        x = x + self.ao(a.transpose(1, 2).contiguous().view(B, T, D))
        return x + self.f2(torch.nn.functional.gelu(self.f1(self.n2(x))))
