

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sliding_window_mask(length: int, window: int, device=None) -> torch.Tensor:
    if length < 0 or window < 1:
        raise ValueError("length must be nonnegative and window positive")
    positions = torch.arange(length, device=device)
    query = positions[:, None]
    key = positions[None, :]
    allowed = (key <= query) & ((query - key) < window)
    return ~allowed


class MistralDecoder(nn.Module):
    architecture_id = "mistral-decoder"

    def __init__(
        self, vocab: int = 64, dim: int = 32, layers: int = 2, heads: int = 4,
        kv_heads: int = 1, window: int = 4, rope_theta: float = 1_000_000.0,
    ) -> None:
        super().__init__()
        if min(vocab, dim, layers, heads, kv_heads, window) <= 0 or dim % heads or rope_theta <= 0:
            raise ValueError("dimensions and window must be positive and dim divisible by heads")
        if heads % kv_heads or (dim // heads) % 2:
            raise ValueError("heads must divide into kv_heads and head dimension must be even")
        self.tok = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            [_MBlock(dim, heads, kv_heads, window, rope_theta) for _ in range(layers)]
        )
        self.norm = _RMS(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


class _RMS(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt() * self.w


class _MBlock(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, window: int,
                 rope_theta: float = 1_000_000.0) -> None:
        super().__init__()
        self.n1 = _RMS(dim)
        self.n2 = _RMS(dim)
        self.h = heads
        self.kv = kv_heads
        self.hd = dim // heads
        self.groups = heads // kv_heads
        self.window = window
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, kv_heads * self.hd, bias=False)
        self.v = nn.Linear(dim, kv_heads * self.hd, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        hidden = 7 * dim // 2
        self.w = nn.Linear(dim, hidden, bias=False)
        self.v2 = nn.Linear(dim, hidden, bias=False)
        self.p = nn.Linear(hidden, dim, bias=False)
        inv = 1.0 / (rope_theta ** (torch.arange(0, self.hd, 2).float() / self.hd))
        self.register_buffer("inv", inv, persistent=False)

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.shape
        pairs = x.reshape(batch, heads, length, self.hd // 2, 2)
        positions = torch.arange(length, device=x.device, dtype=self.inv.dtype)
        angles = torch.outer(positions, self.inv).to(dtype=x.dtype)
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]
        even, odd = pairs.unbind(dim=-1)
        return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        h = self.n1(x)
        q = self._rope(self.q(h).view(batch, length, self.h, self.hd).transpose(1, 2))
        k = self._rope(self.k(h).view(batch, length, self.kv, self.hd).transpose(1, 2))
        v = self.v(h).view(batch, length, self.kv, self.hd).transpose(1, 2)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.hd)
        disallowed = sliding_window_mask(length, self.window, device=x.device)
        scores = scores.masked_fill(disallowed.view(1, 1, length, length), -torch.inf)
        attended = torch.softmax(scores, dim=-1) @ v
        x = x + self.o(attended.transpose(1, 2).contiguous().view(batch, length, dim))
        h = self.n2(x)
        return x + self.p(torch.nn.functional.silu(self.w(h)) * self.v2(h))
