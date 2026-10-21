from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


REFERENCE_CONFIGS = {
    "llama1-7b": dict(version="llama1", d_model=4096, num_layers=32, num_heads=32, num_kv_heads=32, intermediate_size=11008, context_length=2048, vocab_size=32000, rope_theta=10000.0, rms_norm_eps=1e-6),
    "llama1-13b": dict(version="llama1", d_model=5120, num_layers=40, num_heads=40, num_kv_heads=40, intermediate_size=13824, context_length=2048, vocab_size=32000, rope_theta=10000.0, rms_norm_eps=1e-6),
    "llama1-30b": dict(version="llama1", d_model=6656, num_layers=60, num_heads=52, num_kv_heads=52, intermediate_size=17920, context_length=2048, vocab_size=32000, rope_theta=10000.0, rms_norm_eps=1e-6),
    "llama1-65b": dict(version="llama1", d_model=8192, num_layers=80, num_heads=64, num_kv_heads=64, intermediate_size=22016, context_length=2048, vocab_size=32000, rope_theta=10000.0, rms_norm_eps=1e-6),
    "llama2-7b": dict(version="llama2", d_model=4096, num_layers=32, num_heads=32, num_kv_heads=32, intermediate_size=11008, context_length=4096, vocab_size=32000, rope_theta=10000.0, rms_norm_eps=1e-5),
    "llama2-13b": dict(version="llama2", d_model=5120, num_layers=40, num_heads=40, num_kv_heads=40, intermediate_size=13824, context_length=4096, vocab_size=32000, rope_theta=10000.0, rms_norm_eps=1e-5),
    "llama2-70b": dict(version="llama2", d_model=8192, num_layers=80, num_heads=64, num_kv_heads=8, intermediate_size=28672, context_length=4096, vocab_size=32000, rope_theta=10000.0, rms_norm_eps=1e-5),
    "llama3-8b": dict(version="llama3", d_model=4096, num_layers=32, num_heads=32, num_kv_heads=8, intermediate_size=14336, context_length=8192, vocab_size=128256, rope_theta=500000.0, rms_norm_eps=1e-5),
    "llama3-70b": dict(version="llama3", d_model=8192, num_layers=80, num_heads=64, num_kv_heads=8, intermediate_size=28672, context_length=8192, vocab_size=128256, rope_theta=500000.0, rms_norm_eps=1e-5),
}


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def apply_rotary(x: torch.Tensor, position_ids: torch.Tensor, theta: float) -> torch.Tensor:
    dim = x.shape[-1]
    inv_freq = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    phase = position_ids.float()[..., None] * inv_freq
    cos, sin = phase.cos().to(x.dtype)[:, None], phase.sin().to(x.dtype)[:, None]
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, hidden, bias=False)
        self.up_proj = nn.Linear(width, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int, theta: float) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim, self.theta = heads, kv_heads, width // heads, theta
        self.q_proj = nn.Linear(width, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(width, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(width, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch, time, width = x.shape
        q = self.q_proj(x).view(batch, time, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary(q, position_ids, self.theta), apply_rotary(k, position_ids, self.theta)
        if self.kv_heads != self.heads:
            copies = self.heads // self.kv_heads
            k, v = k.repeat_interleave(copies, dim=1), v.repeat_interleave(copies, dim=1)
        order = torch.arange(time, device=x.device)
        allowed = (order[None, :] <= order[:, None])[None, None] & valid[:, None, None, :]
        allowed = allowed | (~valid[:, None, :, None] & torch.eye(time, device=x.device, dtype=torch.bool)[None, None])
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        out = (probs @ v).transpose(1, 2).reshape(batch, time, width)
        return self.o_proj(out) * valid.unsqueeze(-1)


class LlamaBlock(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int, hidden: int, eps: float, theta: float) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(width, eps)
        self.attention = LlamaAttention(width, heads, kv_heads, theta)
        self.ffn_norm = RMSNorm(width, eps)
        self.feed_forward = SwiGLU(width, hidden)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = (x + self.attention(self.attention_norm(x), valid, positions)) * valid.unsqueeze(-1)
        x = (x + self.feed_forward(self.ffn_norm(x))) * valid.unsqueeze(-1)
        return x


class Llama(nn.Module):
    architecture_id = "llama"
    PRESETS = {"tiny": {"preset": "tiny"}, "small": {"preset": "small"}, "base": {"preset": "base"}}
    PRESETS.update({key: {"size": key} for key in REFERENCE_CONFIGS})
    BUDGET_KNOBS = {"d_model": range(16, 257, 8)}

    def __init__(self, vocab_size: int, context_length: int, d_model: int = 256, num_layers: int = 4, num_heads: int = 4, num_kv_heads: int | None = None, intermediate_size: int | None = None, version: str = "llama3", rope_theta: float | None = None, rms_norm_eps: float | None = None, dropout: float = 0.0, pad_token_id: int | None = None, preset: str | None = None, size: str | None = None) -> None:
        super().__init__()
        if preset is not None:
            if preset not in {"tiny", "small", "base"}:
                raise ValueError("preset must be tiny, small, or base")
            d_model, num_layers, num_heads, num_kv_heads, intermediate_size = {"tiny": (32, 2, 4, 2, 88), "small": (128, 4, 4, 2, 352), "base": (256, 6, 8, 4, 704)}[preset]
        if size is not None:
            if size not in REFERENCE_CONFIGS:
                raise ValueError("unknown native size preset")
            native = REFERENCE_CONFIGS[size]
            version = native["version"]
            d_model, num_layers, num_heads, num_kv_heads = native["d_model"], native["num_layers"], native["num_heads"], native["num_kv_heads"]
            intermediate_size, context_length, vocab_size = native["intermediate_size"], native["context_length"], native["vocab_size"]
            rope_theta, rms_norm_eps = native["rope_theta"], native["rms_norm_eps"]
        if version not in {"llama1", "llama2", "llama3"}:
            raise ValueError("version must be llama1, llama2, or llama3")
        for name, value in (("vocab_size", vocab_size), ("context_length", context_length), ("d_model", d_model), ("num_layers", num_layers), ("num_heads", num_heads)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        intermediate_size = 4 * d_model if intermediate_size is None else intermediate_size
        if d_model % num_heads or num_heads % num_kv_heads or d_model // num_heads % 2:
            raise ValueError("head geometry requires divisible query/KV heads and an even head dimension")
        if intermediate_size <= 0 or not 0 <= dropout < 1:
            raise ValueError("intermediate_size must be positive and dropout in [0, 1)")
        if pad_token_id is not None and not 0 <= pad_token_id < vocab_size:
            raise ValueError("pad_token_id must be within vocab_size")
        eps = rms_norm_eps if rms_norm_eps is not None else (1e-6 if version == "llama1" else 1e-5)
        theta = rope_theta if rope_theta is not None else (500000.0 if version == "llama3" else 10000.0)
        if eps <= 0 or theta <= 0:
            raise ValueError("rms_norm_eps and rope_theta must be positive")
        self.vocab_size, self.context_length, self.d_model, self.pad_token_id = vocab_size, context_length, d_model, pad_token_id
        self.version, self.num_heads, self.num_kv_heads = version, num_heads, num_kv_heads
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([LlamaBlock(d_model, num_heads, num_kv_heads, intermediate_size, eps, theta) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model, eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_features(self, tokens: torch.Tensor, mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be LongTensor [batch, time]")
        batch, time = tokens.shape
        if batch <= 0 or not 0 < time <= self.context_length:
            raise ValueError("tokens require positive batch and time within context_length")
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.vocab_size):
            raise ValueError("token IDs outside vocabulary")
        valid = torch.ones_like(tokens, dtype=torch.bool) if mask is None else mask.to(dtype=torch.bool)
        if valid.shape != tokens.shape or valid.device != tokens.device:
            raise ValueError("mask must match tokens shape and device")
        if position_ids is None:
            position_ids = (valid.long().cumsum(-1) - 1).clamp_min(0)
        if position_ids.shape != tokens.shape or position_ids.dtype != torch.long or position_ids.device != tokens.device:
            raise ValueError("position_ids must be LongTensor matching tokens")
        if position_ids.numel() and (position_ids.min() < 0 or position_ids.max() >= self.context_length):
            raise ValueError("position_ids outside context_length")
        x = self.dropout(self.token_embedding(tokens)) * valid.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, valid, position_ids)
        return self.norm(x) * valid.unsqueeze(-1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.lm_head(self.forward_features(tokens, mask, position_ids))
