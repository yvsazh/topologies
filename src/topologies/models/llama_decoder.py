

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LLaMADecoder(nn.Module):
    architecture_id = "llama-decoder"

    def __init__(self, vocab: int = 64, dim: int = 32, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        if min(vocab, dim, layers, heads) <= 0:
            raise ValueError("vocab, dim, layers, and heads must be positive")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if (dim // heads) % 2:
            raise ValueError("head dimension must be even for rotary embeddings")
        self.tok = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([_LBlock(dim, heads) for _ in range(layers)])
        self.norm = _RMS(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


class _RMS(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt() * self.w


class _LBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.n1 = _RMS(dim)
        self.n2 = _RMS(dim)
        self.h = heads
        self.hd = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        hidden = 8 * dim // 3
        self.w = nn.Linear(dim, hidden, bias=False)
        self.v2 = nn.Linear(dim, hidden, bias=False)
        self.p = nn.Linear(hidden, dim, bias=False)
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.hd, 2).float() / self.hd))
        self.register_buffer("inv", inv, persistent=False)

    def _rope(self, t: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = t.shape
        pairs = t.reshape(batch, heads, length, self.hd // 2, 2)
        positions = torch.arange(length, device=t.device, dtype=self.inv.dtype)
        angles = torch.outer(positions, self.inv).to(dtype=t.dtype)
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]
        even, odd = pairs.unbind(dim=-1)
        return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        h = self.n1(x)
        q = self._rope(self.q(h).view(batch, length, self.h, self.hd).transpose(1, 2))
        k = self._rope(self.k(h).view(batch, length, self.h, self.hd).transpose(1, 2))
        v = self.v(h).view(batch, length, self.h, self.hd).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.hd)
        causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal.view(1, 1, length, length), -torch.inf)
        attended = torch.softmax(scores, dim=-1) @ v
        x = x + self.o(attended.transpose(1, 2).contiguous().view(batch, length, dim))
        h = self.n2(x)
        return x + self.p(torch.nn.functional.silu(self.w(h)) * self.v2(h))
