from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def apply_rotary(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    dim = x.shape[-1]
    inv_freq = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    phase = positions.float()[..., None] * inv_freq
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


class MistralAttention(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int, theta: float, window: int, chunk_size: int) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, width // heads
        self.theta, self.window, self.chunk_size = theta, window, chunk_size
        self.q_proj = nn.Linear(width, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(width, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(width, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch, time, width = x.shape
        q = self.q_proj(x).view(batch, time, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary(q, positions, self.theta), apply_rotary(k, positions, self.theta)
        pieces = []
        for start in range(0, time, self.chunk_size):
            end = min(time, start + self.chunk_size)
            key_start = max(0, start - self.window)
            qpart = q[:, :, start:end]
            kpart, vpart = k[:, :, key_start:end], v[:, :, key_start:end]
            qpos = torch.arange(start, end, device=x.device)
            kpos = torch.arange(key_start, end, device=x.device)
            allowed = (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] >= qpos[:, None] - self.window)
            allowed = allowed[None, None] & valid[:, None, None, key_start:end]
            empty = ~allowed.any(-1, keepdim=True)
            allowed = allowed | (empty & (kpos[None, None, None, :] == qpos[None, None, :, None]))
            if self.heads != self.kv_heads:
                factor = self.heads // self.kv_heads
                kpart = kpart.repeat_interleave(factor, dim=1)
                vpart = vpart.repeat_interleave(factor, dim=1)
            scores = qpart @ kpart.transpose(-2, -1) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            probs = F.softmax(scores.float(), dim=-1).to(scores.dtype)
            pieces.append((probs @ vpart).transpose(1, 2).reshape(batch, end - start, width))
        return self.o_proj(torch.cat(pieces, dim=1)) * valid.unsqueeze(-1)


class MistralBlock(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int, hidden: int, eps: float, theta: float, window: int, chunk_size: int) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(width, eps)
        self.attention = MistralAttention(width, heads, kv_heads, theta, window, chunk_size)
        self.ffn_norm = RMSNorm(width, eps)
        self.feed_forward = SwiGLU(width, hidden)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = (x + self.attention(self.attention_norm(x), valid, positions)) * valid.unsqueeze(-1)
        return (x + self.feed_forward(self.ffn_norm(x))) * valid.unsqueeze(-1)


class Mistral(nn.Module):
    architecture_id = "mistral"
    PRESETS = {"tiny": {"preset": "tiny"}, "small": {"preset": "small"}, "base": {"preset": "base"}, "mistral-7b": {"size": "mistral-7b"}}
    BUDGET_KNOBS = {"d_model": range(16, 257, 8)}

    def __init__(self, vocab_size: int = 32000, context_length: int = 32768, d_model: int = 256, num_layers: int = 4, num_heads: int = 4, num_kv_heads: int = 2, intermediate_size: int = 704, sliding_window: int = 4096, rope_theta: float = 10000.0, rms_norm_eps: float = 1e-5, dropout: float = 0.0, pad_token_id: int | None = None, chunk_size: int = 128, preset: str | None = None, size: str | None = None) -> None:
        super().__init__()
        if preset is not None:
            if preset not in {"tiny", "small", "base"}:
                raise ValueError("preset must be tiny, small, or base")
            d_model, num_layers, num_heads, num_kv_heads, intermediate_size = {"tiny": (32, 2, 4, 2, 88), "small": (128, 4, 4, 2, 352), "base": (256, 6, 8, 4, 704)}[preset]
        if size is not None:
            if size != "mistral-7b":
                raise ValueError("size must be mistral-7b")
            vocab_size, context_length, d_model, num_layers = 32000, 32768, 4096, 32
            num_heads, num_kv_heads, intermediate_size = 32, 8, 14336
            sliding_window, rope_theta, rms_norm_eps = 4096, 10000.0, 1e-5
        vals = (vocab_size, context_length, d_model, num_layers, num_heads, num_kv_heads, intermediate_size, sliding_window, chunk_size)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in vals):
            raise ValueError("model dimensions, sliding_window, and chunk_size must be positive integers")
        if d_model % num_heads or num_heads % num_kv_heads or (d_model // num_heads) % 2:
            raise ValueError("d_model must divide heads, heads must divide into KV heads, and head dimension must be even")
        if not 0 <= dropout < 1 or rope_theta <= 0 or rms_norm_eps <= 0:
            raise ValueError("dropout must be in [0,1), rope_theta and rms_norm_eps positive")
        if pad_token_id is not None and not 0 <= pad_token_id < vocab_size:
            raise ValueError("pad_token_id must be within vocab_size")
        self.vocab_size, self.context_length, self.d_model, self.pad_token_id = vocab_size, context_length, d_model, pad_token_id
        self.sliding_window, self.num_heads, self.num_kv_heads = sliding_window, num_heads, num_kv_heads
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([MistralBlock(d_model, num_heads, num_kv_heads, intermediate_size, rms_norm_eps, rope_theta, sliding_window, chunk_size) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model, rms_norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_features(self, tokens: torch.Tensor, mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be LongTensor [batch,time]")
        batch, time = tokens.shape
        if batch <= 0 or not 0 < time <= self.context_length:
            raise ValueError("tokens require positive batch and time within context_length")
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.vocab_size):
            raise ValueError("token IDs outside vocabulary")
        valid = torch.ones_like(tokens, dtype=torch.bool) if mask is None else mask.to(dtype=torch.bool)
        if valid.shape != tokens.shape or valid.device != tokens.device:
            raise ValueError("mask must match tokens shape and device")
        if position_ids is None:
            position_ids = torch.arange(time, device=tokens.device).expand(batch, -1)
        if position_ids.shape != tokens.shape or position_ids.dtype != torch.long or position_ids.device != tokens.device or (position_ids.numel() and (position_ids.min() < 0 or position_ids.max() >= self.context_length)):
            raise ValueError("position_ids must be LongTensor matching tokens and within context_length")
        x = self.dropout(self.token_embedding(tokens)) * valid.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, valid, position_ids)
        return self.norm(x) * valid.unsqueeze(-1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.lm_head(self.forward_features(tokens, mask, position_ids))
