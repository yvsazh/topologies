from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def relative_position_bucket(relative_position: torch.Tensor, num_buckets: int,
                             max_distance: int = 128, bidirectional: bool = True) -> torch.Tensor:
    buckets = 0
    if bidirectional:
        num_buckets //= 2
        buckets = buckets + (relative_position > 0).long() * num_buckets
        distance = relative_position.abs()
    else:
        distance = (-relative_position).clamp_min(0)
    max_exact = num_buckets // 2
    is_small = distance < max_exact
    large = max_exact + (torch.log(distance.float().clamp_min(max_exact) / max_exact) /
                         math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
    return buckets + torch.where(is_small, distance, large.clamp_max(num_buckets - 1))


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim
        self.q = nn.Linear(dim, inner, bias=False)
        self.k = nn.Linear(dim, inner, bias=False)
        self.v = nn.Linear(dim, inner, bias=False)
        self.o = nn.Linear(inner, dim, bias=False)
        self.dropout = dropout

    def forward(self, query: torch.Tensor, key: torch.Tensor, key_mask: torch.Tensor,
                causal: bool = False, bias: nn.Embedding | None = None,
                bidirectional: bool = True) -> torch.Tensor:
        b, qlen, _ = query.shape
        klen = key.shape[1]
        split = lambda t: t.view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q(query)), split(self.k(key)), split(self.v(key))
        scores = q @ k.transpose(-2, -1)
        qi = torch.arange(qlen, device=query.device)[:, None]
        ki = torch.arange(klen, device=query.device)[None, :]
        if bias is not None:
            buckets = relative_position_bucket(ki - qi, bias.num_embeddings, bidirectional=bidirectional)
            scores = scores + bias(buckets).permute(2, 0, 1).unsqueeze(0)
        valid = key_mask[:, None, None, :].expand(b, self.heads, qlen, klen)
        if causal:
            valid = valid & (ki <= qi)[None, None]
        probs = torch.softmax(scores.masked_fill(~valid, torch.finfo(scores.dtype).min), -1)
        out = (F.dropout(probs, self.dropout, self.training) @ v).transpose(1, 2).contiguous()
        return self.o(out.view(b, qlen, -1))

    def reset_t5(self, d_model: int) -> None:
        nn.init.normal_(self.q.weight, std=(d_model * self.head_dim) ** -0.5)
        nn.init.normal_(self.k.weight, std=d_model ** -0.5)
        nn.init.normal_(self.v.weight, std=d_model ** -0.5)
        nn.init.normal_(self.o.weight, std=(self.heads * self.head_dim) ** -0.5)


class _FF(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.wi_0 = nn.Linear(dim, hidden, bias=False)
        self.wi_1 = nn.Linear(dim, hidden, bias=False)
        self.wo = nn.Linear(hidden, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(F.dropout(F.gelu(self.wi_0(x), approximate="tanh") * self.wi_1(x), self.dropout, self.training))


class _EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, hidden: int, dropout: float, eps: float) -> None:
        super().__init__()
        self.norm1, self.norm2 = _RMSNorm(dim, eps), _RMSNorm(dim, eps)
        self.attn, self.ff, self.dropout = _Attention(dim, heads, head_dim, dropout), _FF(dim, hidden, dropout), dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor, bias: nn.Embedding) -> torch.Tensor:
        y = self.norm1(x)
        x = x + F.dropout(self.attn(y, y, mask, bias=bias), self.dropout, self.training)
        return x + F.dropout(self.ff(self.norm2(x)), self.dropout, self.training)


class _DecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, hidden: int, dropout: float, eps: float) -> None:
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (_RMSNorm(dim, eps) for _ in range(3))
        self.self_attn = _Attention(dim, heads, head_dim, dropout)
        self.cross_attn = _Attention(dim, heads, head_dim, dropout)
        self.ff, self.dropout = _FF(dim, hidden, dropout), dropout

    def forward(self, x: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor,
                memory_mask: torch.Tensor, bias: nn.Embedding) -> torch.Tensor:
        y = self.norm1(x)
        x = x + F.dropout(self.self_attn(y, y, mask, causal=True, bias=bias,
                                         bidirectional=False), self.dropout, self.training)
        x = x + F.dropout(self.cross_attn(self.norm2(x), memory, memory_mask), self.dropout, self.training)
        return x + F.dropout(self.ff(self.norm3(x)), self.dropout, self.training)


class ByT5(nn.Module):
    architecture_id = "byt5"
    PRESETS = {
        "tiny": {"d_model": 64, "num_layers": 2, "num_decoder_layers": 1, "num_heads": 4, "d_kv": 16, "d_ff": 128},
        "small": {"d_model": 1472, "num_layers": 12, "num_decoder_layers": 4, "num_heads": 6, "d_kv": 64, "d_ff": 3584, "dropout": 0.1},
        "base": {"d_model": 1536, "num_layers": 18, "num_decoder_layers": 6, "num_heads": 12, "d_kv": 64, "d_ff": 3968, "dropout": 0.1},
    }
    BUDGET_KNOBS = {"d_model": range(32, 257, 16)}

    def __init__(self, vocab_size: int = 384, context_length: int = 2048, d_model: int = 128,
                 num_layers: int = 4, num_decoder_layers: int = 2, num_heads: int = 4,
                 d_kv: int = 32, d_ff: int = 512, dropout: float = 0.0,
                 pad_token_id: int = 0, eos_token_id: int = 1,
                 relative_attention_num_buckets: int = 32, layer_norm_epsilon: float = 1e-6,
                 tie_word_embeddings: bool = False, initializer_factor: float = 1.0) -> None:
        super().__init__()
        if min(vocab_size, context_length, d_model, num_layers, num_decoder_layers, num_heads, d_kv, d_ff) <= 0:
            raise ValueError("vocabulary, context, model, layer, head, and feed-forward dimensions must be positive")
        if vocab_size < 384 or not 0 <= pad_token_id < vocab_size or not 0 <= eos_token_id < vocab_size:
            raise ValueError("vocab_size must include ByT5 controls and token IDs must be in range")
        if relative_attention_num_buckets < 4 or relative_attention_num_buckets % 2:
            raise ValueError("relative_attention_num_buckets must be even and at least four")
        if layer_norm_epsilon <= 0 or initializer_factor <= 0 or not 0 <= dropout < 1:
            raise ValueError("epsilon and initializer_factor must be positive and dropout in [0, 1)")
        self.vocab_size, self.context_length, self.d_model = vocab_size, context_length, d_model
        self.pad_token_id, self.eos_token_id, self.tie_word_embeddings = pad_token_id, eos_token_id, tie_word_embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_bias = nn.Embedding(relative_attention_num_buckets, num_heads)
        self.decoder_bias = nn.Embedding(relative_attention_num_buckets, num_heads)
        self.encoder = nn.ModuleList([_EncoderLayer(d_model, num_heads, d_kv, d_ff, dropout, layer_norm_epsilon) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([_DecoderLayer(d_model, num_heads, d_kv, d_ff, dropout, layer_norm_epsilon) for _ in range(num_decoder_layers)])
        self.encoder_norm, self.decoder_norm = _RMSNorm(d_model, layer_norm_epsilon), _RMSNorm(d_model, layer_norm_epsilon)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout, self.initializer_factor = dropout, initializer_factor
        self.reset_parameters()

    def reset_parameters(self) -> None:
        f = self.initializer_factor
        for name, p in self.named_parameters():
            if name.endswith("weight") and p.ndim == 2 and "norm" not in name:
                if "embedding.weight" in name or "encoder_bias.weight" in name or "decoder_bias.weight" in name:
                    nn.init.normal_(p, std=f)
                elif ".ff.wo.weight" in name:
                    nn.init.normal_(p, std=f * (p.shape[1] ** -0.5))
                else:
                    nn.init.normal_(p, std=f * (self.d_model ** -0.5))
        for bias in (self.encoder_bias, self.decoder_bias):
            nn.init.normal_(bias.weight, std=f)
        for layer in self.encoder:
            layer.attn.reset_t5(self.d_model)
        for layer in self.decoder:
            layer.self_attn.reset_t5(self.d_model)
            layer.cross_attn.reset_t5(self.d_model)
        if self.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

    def encode(self, src: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if src.ndim != 2 or src.size(1) > self.context_length:
            raise ValueError("src must be [batch, sequence] within context_length")
        mask = src.ne(self.pad_token_id) if mask is None else mask.bool()
        if mask.shape != src.shape or torch.any(~mask.any(-1)):
            raise ValueError("each source requires a nonempty matching mask")
        x = F.dropout(self.embedding(src), self.dropout, self.training) * mask.unsqueeze(-1)
        for layer in self.encoder:
            x = layer(x, mask, self.encoder_bias) * mask.unsqueeze(-1)
        return self.encoder_norm(x) * mask.unsqueeze(-1), mask

    def forward_features(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(input_ids, mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        memory, memory_mask = self.encode(src, mask)
        if tgt.ndim != 2 or tgt.size(1) > self.context_length or tgt.size(1) == 0:
            raise ValueError("tgt must be a nonempty [batch, sequence] within context_length")
        target_mask = tgt.ne(self.pad_token_id)
        target_mask[:, 0] = True
        x = F.dropout(self.embedding(tgt), self.dropout, self.training)
        for layer in self.decoder:
            x = layer(x, memory, target_mask, memory_mask, self.decoder_bias) * target_mask.unsqueeze(-1)
        x = self.decoder_norm(x)
        if self.tie_word_embeddings:
            x = x * (self.d_model ** -0.5)
        return F.linear(x, self.lm_head.weight)


def encode_utf8(text: str, append_eos: bool = True, unknown_token_id: int = 2) -> list[int]:
    ids = [3 + byte for byte in text.encode("utf-8")]
    if append_eos:
        ids.append(1)
    return ids
