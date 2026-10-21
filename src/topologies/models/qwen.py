from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def _rope(x: torch.Tensor, positions: torch.Tensor, theta: float, factor: float) -> torch.Tensor:
    half = x.shape[-1] // 2
    inv = theta ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    phase = positions.float()[..., None] * inv / factor
    cos, sin = phase.cos().to(x.dtype)[:, None], phase.sin().to(x.dtype)[:, None]
    left, right = x[..., :half], x[..., half:]
    return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)


class QwenAttention(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int, head_dim: int, bias: bool, theta: float, rope_factor: float, qk_norm: bool, eps: float) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.theta, self.rope_factor = theta, rope_factor
        self.q_proj = nn.Linear(width, heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(width, kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(width, kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(heads * head_dim, width, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, valid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        b, t, width = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = _rope(q, positions, self.theta, self.rope_factor), _rope(k, positions, self.theta, self.rope_factor)
        if self.heads != self.kv_heads:
            repeat = self.heads // self.kv_heads
            k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        causal = torch.ones((t, t), dtype=torch.bool, device=x.device).tril()
        allowed = causal[None, None] & valid[:, None, None, :]
        allowed = allowed | (~allowed.any(-1, keepdim=True) & causal[None, None])
        probs = F.softmax(scores.masked_fill(~allowed, torch.finfo(scores.dtype).min).float(), -1).to(scores.dtype)
        out = (probs @ v).transpose(1, 2).reshape(b, t, self.heads * self.head_dim)
        return self.o_proj(out) * valid.unsqueeze(-1)


class QwenBlock(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int, head_dim: int, hidden: int, eps: float, bias: bool, theta: float, factor: float, qk_norm: bool) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(width, eps)
        self.attention = QwenAttention(width, heads, kv_heads, head_dim, bias, theta, factor, qk_norm, eps)
        self.ffn_norm = RMSNorm(width, eps)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = (x + self.attention(self.attn_norm(x), valid, positions)) * valid.unsqueeze(-1)
        return (x + self.down(F.silu(self.gate(self.ffn_norm(x))) * self.up(self.ffn_norm(x)))) * valid.unsqueeze(-1)


class Qwen(nn.Module):
    architecture_id = "qwen"
    PRESETS = {"tiny": {"preset": "tiny"}, "small": {"preset": "small"}, "base": {"preset": "base"},
               "qwen2-0.5b": {"size": "qwen2-0.5b"}, "qwen2.5-0.5b": {"size": "qwen2.5-0.5b"},
               "qwen3-0.6b": {"size": "qwen3-0.6b"}}
    BUDGET_KNOBS = {"d_model": range(16, 257, 8)}
    CONFIGS = {
        "qwen2-0.5b": {"layers": 24, "heads": 14, "kv_heads": 2, "head_dim": 64, "hidden": 896, "intermediate": 4864, "context": 131072, "vocab": 151936, "bias": True, "theta": 1000000.0, "factor": 1.0, "qk_norm": False, "eps": 1e-6},
        "qwen2.5-0.5b": {"layers": 24, "heads": 14, "kv_heads": 2, "head_dim": 64, "hidden": 896, "intermediate": 4864, "context": 32768, "vocab": 151936, "bias": True, "theta": 1000000.0, "factor": 1.0, "qk_norm": False, "eps": 1e-6},
        "qwen3-0.6b": {"layers": 28, "heads": 16, "kv_heads": 8, "head_dim": 128, "hidden": 1024, "intermediate": 3072, "context": 40960, "vocab": 151936, "bias": False, "theta": 1000000.0, "factor": 1.0, "qk_norm": True, "eps": 1e-6},
    }

    def __init__(self, vocab_size: int = 32000, context_length: int = 256, d_model: int = 128, num_layers: int = 4, num_heads: int = 4, num_kv_heads: int = 2, intermediate_size: int = 352, attention_bias: bool = False, rope_theta: float = 1000000.0, rope_factor: float = 1.0, qk_norm: bool = False, rms_norm_eps: float = 1e-6, dropout: float = 0.0, pad_token_id: int | None = None, preset: str | None = None, size: str | None = None, head_dim: int | None = None) -> None:
        super().__init__()
        if preset is not None:
            if preset not in {"tiny", "small", "base"}:
                raise ValueError("preset must be tiny, small, or base")
            d_model, num_layers, num_heads, num_kv_heads, intermediate_size = {"tiny": (32, 2, 4, 2, 88), "small": (128, 4, 4, 2, 352), "base": (256, 6, 8, 4, 704)}[preset]
            head_dim = d_model // num_heads
        if size is not None:
            if size not in self.CONFIGS:
                raise ValueError(f"unsupported native size {size!r}")
            config = self.CONFIGS[size]
            vocab_size, d_model, head_dim = config["vocab"], config["hidden"], config["head_dim"]
            num_layers, num_heads, num_kv_heads, intermediate_size = config["layers"], config["heads"], config["kv_heads"], config["intermediate"]
            context_length, attention_bias, rope_theta, rope_factor, qk_norm, rms_norm_eps = config["context"], config["bias"], config["theta"], config["factor"], config["qk_norm"], config["eps"]
        vals = (vocab_size, context_length, d_model, num_layers, num_heads, num_kv_heads, intermediate_size)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in vals):
            raise ValueError("model dimensions must be positive integers")
        head_dim = d_model // num_heads if head_dim is None else head_dim
        if num_heads % num_kv_heads or head_dim % 2:
            raise ValueError("heads must divide KV heads and head_dim be even")
        if rope_theta <= 0 or rope_factor <= 0 or rms_norm_eps <= 0 or not 0 <= dropout < 1:
            raise ValueError("RoPE/norm scales must be positive and dropout in [0,1)")
        if pad_token_id is not None and not 0 <= pad_token_id < vocab_size:
            raise ValueError("pad_token_id must be in vocabulary")
        self.vocab_size, self.context_length, self.d_model, self.pad_token_id = vocab_size, context_length, d_model, pad_token_id
        self.attention_bias, self.qk_norm, self.rope_theta, self.rope_factor = attention_bias, qk_norm, rope_theta, rope_factor
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([QwenBlock(d_model, num_heads, num_kv_heads, head_dim, intermediate_size, rms_norm_eps, attention_bias, rope_theta, rope_factor, qk_norm) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model, rms_norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward_features(self, tokens: torch.Tensor, mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.dtype != torch.long or not 0 < tokens.shape[1] <= self.context_length:
            raise ValueError("tokens must be LongTensor [batch,time] within context_length")
        valid = torch.ones_like(tokens, dtype=torch.bool) if mask is None else mask.bool()
        if valid.shape != tokens.shape or valid.device != tokens.device:
            raise ValueError("mask must match token shape and device")
        if position_ids is None:
            position_ids = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        if position_ids.shape != tokens.shape or position_ids.dtype != torch.long or position_ids.device != tokens.device:
            raise ValueError("position_ids must match tokens")
        x = self.dropout(self.embedding(tokens)) * valid.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, valid, position_ids)
        return self.norm(x) * valid.unsqueeze(-1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.lm_head(self.forward_features(tokens, mask, position_ids))
