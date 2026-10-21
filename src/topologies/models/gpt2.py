from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        sequence_order = torch.arange(length, device=x.device)
        causal = sequence_order[None, :] <= sequence_order[:, None]
        allowed = causal[None, None] & valid[:, None, None, :]
        allowed = allowed | (~valid[:, None, :, None] & torch.eye(length, dtype=torch.bool, device=x.device)[None, None])
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.attn_dropout, training=self.training)
        result = (weights @ v).transpose(1, 2).reshape(batch, length, width)
        return self.resid_dropout(self.proj(result)) * valid.unsqueeze(-1)


class GPT2Block(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(width, eps=1e-5)
        self.attn = CausalSelfAttention(width, heads, dropout)
        self.ln_2 = nn.LayerNorm(width, eps=1e-5)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"), nn.Linear(4 * width, width), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, valid: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), valid, position_ids)
        x = x + self.mlp(self.ln_2(x)) * valid.unsqueeze(-1)
        return x * valid.unsqueeze(-1)


class GPT2(nn.Module):
    architecture_id = "gpt-2"
    PRESETS = {name: {"preset": name} for name in ("tiny", "small", "base")}
    BUDGET_KNOBS = {"d_model": range(64, 257, 8)}

    def __init__(self, vocab_size: int, context_length: int, d_model: int = 768, num_layers: int = 12, num_heads: int = 12, dropout: float = 0.0, pad_token_id: int | None = None, preset: str | None = None) -> None:
        super().__init__()
        if preset is not None:
            geometries = {"tiny": (64, 2, 2), "small": (256, 4, 4), "base": (768, 12, 12)}
            if preset not in geometries:
                raise ValueError("preset must be tiny, small, or base")
            d_model, num_layers, num_heads = geometries[preset]
        for name, value in (("vocab_size", vocab_size), ("context_length", context_length), ("d_model", d_model), ("num_layers", num_layers), ("num_heads", num_heads)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if pad_token_id is not None and not 0 <= pad_token_id < vocab_size:
            raise ValueError("pad_token_id must be within vocab_size")
        self.vocab_size, self.context_length, self.d_model = vocab_size, context_length, d_model
        self.pad_token_id = pad_token_id
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(context_length, d_model)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList([GPT2Block(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        for block in self.h:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))
            nn.init.normal_(block.mlp[2].weight, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, tokens: torch.Tensor, mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be a LongTensor shaped [batch, time]")
        batch, length = tokens.shape
        if not 0 < length <= self.context_length:
            raise ValueError("sequence length must be in [1, context_length]")
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.vocab_size):
            raise ValueError("token IDs must be within vocab_size")
        valid = torch.ones_like(tokens, dtype=torch.bool) if mask is None else mask.bool()
        if valid.shape != tokens.shape or valid.device != tokens.device:
            raise ValueError("mask must match token shape and device")
        if position_ids is None:
            position_ids = (valid.long().cumsum(-1) - 1).clamp_min(0)
        if position_ids.shape != tokens.shape or position_ids.device != tokens.device or position_ids.dtype != torch.long:
            raise ValueError("position_ids must be a LongTensor matching tokens")
        if position_ids.numel() and (position_ids.min() < 0 or position_ids.max() >= self.context_length):
            raise ValueError("position_ids must be in [0, context_length)")
        x = self.drop(self.wte(tokens) + self.wpe(position_ids)) * valid.unsqueeze(-1)
        for block in self.h:
            x = block(x, valid, position_ids)
        return self.ln_f(x) * valid.unsqueeze(-1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.lm_head(self.forward_features(tokens, mask, position_ids))


GPT2LM = GPT2
