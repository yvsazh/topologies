

from __future__ import annotations

import math

import torch
import torch.nn as nn


class GPTDecoderLM(nn.Module):
    architecture_id = "gpt-decoder-lm"

    def __init__(
        self,
        vocab: int = 64,
        dim: int = 32,
        layers: int = 2,
        heads: int = 4,
        max_positions: int = 32,
    ) -> None:
        super().__init__()
        if min(vocab, dim, layers, heads, max_positions) <= 0 or dim % heads:
            raise ValueError("dimensions must be positive and dim must be divisible by heads")
        self.max_positions = max_positions
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_positions, dim)
        self.blocks = nn.ModuleList([_CausalBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] > self.max_positions:
            raise ValueError(f"tokens must have shape (batch, sequence <= {self.max_positions})")
        T = tokens.shape[1]
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.tok(tokens) + self.pos(pos)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


class _CausalBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.ao = nn.Linear(dim, dim)
        self.f1 = nn.Linear(dim, dim * 4)
        self.f2 = nn.Linear(dim * 4, dim)
        self.heads = heads
        self.hd = dim // heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.n1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)

        def sp(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.heads, self.hd).transpose(1, 2)

        q, k, v = sp(q), sp(k), sp(v)
        s = q @ k.transpose(-2, -1) / math.sqrt(self.hd)
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
        s = s.masked_fill(~causal, -1e9)
        a = torch.softmax(s, dim=-1) @ v
        x = x + self.ao(a.transpose(1, 2).contiguous().view(B, T, D))
        h = self.n2(x)
        return x + self.f2(torch.nn.functional.gelu(self.f1(h), approximate="tanh"))
