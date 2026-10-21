"""Sequence mixers for causal language models.

A mixer is ``forward(x, mask=None, cache=None) -> [B, T, D]``. ``mask`` is
``[B, T]`` with 1 for real tokens. ``cache`` is the object returned by
``init_cache``; passing it appends the new keys and values and attends over
the whole prefix. The training path (``cache is None`` and no padding) uses
scaled-dot-product attention with ``is_causal=True`` and does not sync the host.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from topologies.build.validation import ConfigError, check_divisible, check_positive
from topologies.nn.layers import RMSNorm


class MixerCache:
    """KV cache for one attention layer. ``offset`` is the number of cached keys."""

    def __init__(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.offset: int = 0


_MIXERS: dict[str, Any] = {}


def register_mixer(name: str, factory: Any) -> None:
    """Register ``factory(dim, *, heads, head_dim, max_seq, layer_idx, **kw)``."""
    _MIXERS[str(name).lower()] = factory


def build_mixer(spec: str, dim: int, **ctx: Any) -> nn.Module:
    """Build a registered mixer. ``spec`` is a name such as ``"attention"``."""
    key = str(spec).lower()
    if key not in _MIXERS:
        known = ", ".join(sorted(_MIXERS)) or "(none)"
        raise ConfigError(f"Unknown mixer {spec!r}. Known: {known}.", owner="build_mixer")
    return _MIXERS[key](dim, **ctx)


class Attention(nn.Module):
    """Causal multi-head attention with SDPA, optional GQA, RoPE, and a KV cache.

    References: Vaswani et al. 2017; Su et al. 2021 (RoPE); Shazeer 2019 (GQA/MQA).
    """

    causal = True

    def __init__(
        self,
        dim: int,
        heads: int,
        *,
        kv_heads: int | None = None,
        head_dim: int | None = None,
        max_seq: int = 2048,
        pos: str = "rope",
        rope_base: float = 10_000.0,
        rope_fraction: float = 1.0,
        qk_norm: bool = False,
        logit_softcap: float | None = None,
        bias: bool = False,
        dropout: float = 0.0,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        del layer_idx
        check_positive("Attention", dim=dim, heads=heads, max_seq=max_seq)
        self.heads = int(heads)
        self.kv_heads = int(heads if kv_heads is None else kv_heads)
        if self.heads % self.kv_heads:
            raise ConfigError(
                f"heads ({self.heads}) must be divisible by kv_heads ({self.kv_heads}).",
                owner="Attention",
            )
        if head_dim is None:
            check_divisible(("dim", dim), ("heads", self.heads), owner="Attention")
            head_dim = dim // self.heads
        self.head_dim = int(head_dim)
        self.max_seq = int(max_seq)
        self.dropout = float(dropout)
        self.logit_softcap = None if logit_softcap is None else float(logit_softcap)
        self.q = nn.Linear(dim, self.heads * self.head_dim, bias=bias)
        self.k = nn.Linear(dim, self.kv_heads * self.head_dim, bias=bias)
        self.v = nn.Linear(dim, self.kv_heads * self.head_dim, bias=bias)
        self.out = nn.Linear(self.heads * self.head_dim, dim, bias=bias)
        if pos not in ("rope", "none", "alibi"):
            raise ConfigError(f"pos must be rope, none, or alibi; got {pos!r}.", owner="Attention")
        self.pos = pos
        rope_dim = int(self.head_dim * float(rope_fraction))
        rope_dim -= rope_dim % 2
        self.rope_dim = rope_dim if pos == "rope" else 0
        self.rope = _RopeCache(self.rope_dim, max_seq, rope_base) if self.rope_dim else None
        if pos == "alibi":
            slopes = _alibi_slopes(self.heads)
            self.register_buffer("alibi_slopes", slopes, persistent=False)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None

    def init_cache(self, batch_size: int, *, max_seq: int | None = None,
                   device: torch.device | None = None, dtype: torch.dtype | None = None
                   ) -> MixerCache:
        del batch_size, max_seq, device, dtype
        return MixerCache()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                cache: MixerCache | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q = self._shape(self.q(x), self.heads)
        k = self._shape(self.k(x), self.kv_heads)
        v = self._shape(self.v(x), self.kv_heads)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        offset = 0 if cache is None else cache.offset
        if self.rope is not None and self.rope_dim:
            q = self._rope(q, offset)
            k = self._rope(k, offset)
        if cache is not None and cache.k is not None:
            k = torch.cat([cache.k, k], dim=2)
            v = torch.cat([cache.v, v], dim=2)
        if cache is not None:
            cache.k = k
            cache.v = v
            cache.offset = int(k.shape[2])
        k_len = k.shape[2]
        if self.kv_heads != self.heads:
            rep = self.heads // self.kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        drop = self.dropout if self.training else 0.0
        fast = self.logit_softcap is None and self.pos != "alibi"
        if fast and mask is None and cache is None:
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=t > 1)
        elif fast and mask is None and t == 1:
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=False)
        else:
            bias = _attention_bias(mask, t, k_len, offset, q.device, q.dtype)
            if self.pos == "alibi":
                bias = bias + _alibi_bias(self.alibi_slopes.to(q.device), t, k_len, offset, q.dtype)
            attn = _attend(q, k, v, bias, self.logit_softcap, drop, self.training)
        flat = attn.transpose(1, 2).contiguous().view(b, t, self.heads * self.head_dim)
        return self.out(flat)

    def _shape(self, value: torch.Tensor, heads: int) -> torch.Tensor:
        b, t, _ = value.shape
        return value.view(b, t, heads, self.head_dim).transpose(1, 2)

    def _rope(self, value: torch.Tensor, offset: int) -> torch.Tensor:
        if self.rope is None:
            return value
        rotated = self.rope.apply(value[..., :self.rope_dim], offset)
        if self.rope_dim == value.shape[-1]:
            return rotated
        return torch.cat([rotated, value[..., self.rope_dim:]], dim=-1)


class _RopeCache(nn.Module):
    """Precomputed RoPE cos/sin up to ``max_seq``. Longer inputs use ``inv_freq``."""

    def __init__(self, dim: int, max_seq: int, base: float) -> None:
        super().__init__()
        self.dim = dim
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        pos = torch.arange(max(max_seq, 1), dtype=torch.float32)
        emb = torch.cat([torch.outer(pos, inv), torch.outer(pos, inv)], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def apply(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        length = x.shape[-2]
        end = offset + length
        if end <= self.cos.shape[0]:
            cos, sin = self.cos[offset:end], self.sin[offset:end]
        else:
            pos = torch.arange(offset, end, device=x.device, dtype=torch.float32)
            freq = torch.outer(pos, self.inv_freq.to(x.device))
            emb = torch.cat([freq, freq], dim=-1)
            cos, sin = emb.cos(), emb.sin()
        half = self.dim // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos.to(dtype=x.dtype) + rotated * sin.to(dtype=x.dtype)


def _alibi_slopes(heads: int) -> torch.Tensor:
    """ALiBi head slopes (Press, Smith, Lewis 2022)."""
    ratio = 2 ** (-8 / heads)
    return torch.tensor([ratio ** (i + 1) for i in range(heads)], dtype=torch.float32)


def _alibi_bias(slopes: torch.Tensor, q_len: int, k_len: int, offset: int,
                dtype: torch.dtype) -> torch.Tensor:
    q_pos = torch.arange(offset, offset + q_len, device=slopes.device)
    k_pos = torch.arange(k_len, device=slopes.device)
    dist = k_pos.view(1, k_len) - q_pos.view(q_len, 1)  # <= 0 inside the causal region
    return (slopes.view(1, -1, 1, 1) * dist.to(dtype)).view(1, slopes.shape[0], q_len, k_len)


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
           softcap: float | None, dropout: float, training: bool) -> torch.Tensor:
    logits = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    logits = logits + bias
    weights = torch.softmax(logits, dim=-1)
    if training and dropout:
        weights = F.dropout(weights, dropout)
    return torch.matmul(weights, v)


def _attention_bias(mask: torch.Tensor | None, q_len: int, k_len: int, offset: int,
                    device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    q_pos = torch.arange(offset, offset + q_len, device=device)
    k_pos = torch.arange(k_len, device=device)
    allowed = k_pos.view(1, 1, 1, k_len) <= q_pos.view(1, 1, q_len, 1)
    if mask is not None:
        key_mask = mask
        if key_mask.shape[-1] != k_len:
            # The caller passes the mask of the new tokens. Cached keys stay visible.
            prefix = torch.ones(
                key_mask.shape[0], k_len - key_mask.shape[-1],
                device=key_mask.device, dtype=key_mask.dtype,
            )
            key_mask = torch.cat([prefix, key_mask], dim=-1)
        allowed = allowed & key_mask.bool().view(key_mask.shape[0], 1, 1, k_len)
    bias = torch.zeros(allowed.shape[0] if mask is not None else 1, 1, q_len, k_len,
                       device=device, dtype=dtype)
    return bias.masked_fill(~allowed, torch.finfo(dtype).min)


def _attention_factory(dim: int, *, heads: int, head_dim: int | None = None,
                       max_seq: int = 2048, layer_idx: int = 0, **kw: Any) -> Attention:
    return Attention(
        dim, heads, head_dim=head_dim, max_seq=max_seq, layer_idx=layer_idx, **kw
    )


register_mixer("attention", _attention_factory)
