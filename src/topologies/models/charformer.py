from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientSubwordBlock(nn.Module):
    def __init__(self, dim: int, max_block_size: int = 4, stride: int = 4,
                 score_calibration: bool = False) -> None:
        super().__init__()
        if dim <= 0 or max_block_size <= 0 or stride <= 0:
            raise ValueError("dim, max_block_size and stride must be positive")
        self.max_block_size = max_block_size
        self.stride = stride
        self.score_calibration = score_calibration
        self.scorer = nn.Linear(dim, 1, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, dim = x.shape
        if mask is None:
            mask = torch.ones(batch, length, dtype=torch.bool, device=x.device)
        if mask.shape != (batch, length) or mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean tensor with shape [batch, length]")
        blocks = []
        validity = []
        for size in range(1, self.max_block_size + 1):
            right = (-length) % size
            values = F.pad(x.transpose(1, 2), (0, right)).transpose(1, 2)
            valid = F.pad(mask, (0, right), value=False)
            count = valid.reshape(batch, -1, size).sum(-1).clamp_min(1)
            pooled = (values * valid.unsqueeze(-1)).reshape(batch, -1, size, dim).sum(2) / count.unsqueeze(-1)
            pooled = pooled.repeat_interleave(size, dim=1)[:, :length]
            exists = valid.reshape(batch, -1, size).any(-1).repeat_interleave(size, dim=1)[:, :length]
            blocks.append(pooled)
            validity.append(exists)
        candidates = torch.stack(blocks, dim=2)
        allowed = torch.stack(validity, dim=2) & mask.unsqueeze(-1)
        logits = self.scorer(candidates).squeeze(-1)
        logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1) * allowed
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        if self.score_calibration:
            consensus = torch.matmul(weights, weights.transpose(1, 2))
            consensus = consensus.masked_fill(~mask[:, None, :], torch.finfo(consensus.dtype).min)
            consensus = torch.softmax(consensus, dim=-1) * mask[:, :, None]
            consensus = consensus / consensus.sum(-1, keepdim=True).clamp_min(
                torch.finfo(consensus.dtype).tiny)
            weights = torch.matmul(consensus, weights)
        mixed = (candidates * weights.unsqueeze(-1)).sum(2) * mask.unsqueeze(-1)
        pad = (-length) % self.stride
        down_mask = F.pad(mask, (0, pad), value=False)
        down_count = down_mask.reshape(batch, -1, self.stride).sum(-1)
        down_values = F.pad(mixed, (0, 0, 0, pad)).reshape(batch, -1, self.stride, dim)
        reduced = (down_values * down_mask.reshape(batch, -1, self.stride, 1)).sum(2)
        reduced = reduced / down_count.clamp_min(1).unsqueeze(-1)
        down_valid = down_count.gt(0)
        return reduced * down_valid.unsqueeze(-1), down_valid


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, relative_buckets: int = 32, relative_max: int = 128) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.width = dim // heads
        self.relative_buckets = relative_buckets
        self.relative_max = relative_max
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def _bucket(self, relative: torch.Tensor, causal: bool) -> torch.Tensor:
        buckets = self.relative_buckets
        if causal:
            distance = (-relative).clamp_min(0)
            half = buckets
            exact = half // 2
            large = exact + (torch.log(distance.float().clamp_min(exact) / exact) /
                             math.log(self.relative_max / exact) * (half - exact)).long()
            return torch.where(distance < exact, distance, large.clamp_max(half - 1))
        half = buckets // 2
        sign = (relative > 0).long() * half
        distance = relative.abs()
        exact = half // 2
        large = exact + (torch.log(distance.float().clamp_min(exact) / exact) /
                         math.log(self.relative_max / exact) * (half - exact)).long()
        return sign + torch.where(distance < exact, distance, large.clamp_max(half - 1))

    def forward(self, query: torch.Tensor, key: torch.Tensor, key_mask: torch.Tensor,
                causal: bool = False, relative_bias: nn.Embedding | None = None) -> torch.Tensor:
        b, qlen, dim = query.shape
        klen = key.shape[1]
        split = lambda t: t.view(b, -1, self.heads, self.width).transpose(1, 2)
        q, k, v = split(self.q(query)), split(self.k(key)), split(self.v(key))
        scores = torch.matmul(q, k.transpose(-2, -1))
        qi = torch.arange(qlen, device=query.device)[:, None]
        ki = torch.arange(klen, device=query.device)[None, :]
        relative = ki - qi
        if relative_bias is not None:
            buckets = self._bucket(relative, causal)
            scores = scores + relative_bias(buckets).permute(2, 0, 1).unsqueeze(0)
        valid = key_mask[:, None, None, :].expand(b, self.heads, qlen, klen)
        if causal:
            valid = valid & (ki <= qi)[None, None]
        probs = torch.softmax(scores.masked_fill(~valid, torch.finfo(scores.dtype).min), dim=-1)
        probs = F.dropout(probs, self.dropout, self.training)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(b, qlen, dim)
        return self.o(out)

    def initialize_t5(self, d_model: int) -> None:
        nn.init.normal_(self.q.weight, mean=0.0, std=(d_model * self.width) ** -0.5)
        nn.init.normal_(self.k.weight, mean=0.0, std=d_model ** -0.5)
        nn.init.normal_(self.v.weight, mean=0.0, std=d_model ** -0.5)
        nn.init.normal_(self.o.weight, mean=0.0, std=(self.heads * self.width) ** -0.5)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().square().mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps).to(x.dtype) * self.weight


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float, variant: str) -> None:
        super().__init__()
        if variant not in {"relu", "gated-gelu"}:
            raise ValueError("feed_forward must be 'relu' or 'gated-gelu'")
        self.variant = variant
        self.wi = nn.Linear(dim, hidden, bias=False)
        self.wi_gate = nn.Linear(dim, hidden, bias=False) if variant == "gated-gelu" else None
        self.wo = nn.Linear(hidden, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.wi(x)
        hidden = F.relu(hidden) if self.variant == "relu" else F.gelu(hidden) * self.wi_gate(x)
        return self.wo(F.dropout(hidden, self.dropout, self.training))


class _EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int, dropout: float, variant: str) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(dim)
        self.norm2 = _RMSNorm(dim)
        self.attn = _Attention(dim, heads, dropout)
        self.ff = _FeedForward(dim, hidden, dropout, variant)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                relative_bias: nn.Embedding) -> torch.Tensor:
        normalized = self.norm1(x)
        x = x + F.dropout(self.attn(normalized, normalized, mask,
                                    relative_bias=relative_bias), self.dropout, self.training)
        x = x + F.dropout(self.ff(self.norm2(x)), self.dropout, self.training)
        return x * mask.unsqueeze(-1)


class _DecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int, dropout: float, variant: str) -> None:
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (_RMSNorm(dim) for _ in range(3))
        self.self_attn = _Attention(dim, heads, dropout)
        self.cross_attn = _Attention(dim, heads, dropout)
        self.ff = _FeedForward(dim, hidden, dropout, variant)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor,
                memory_mask: torch.Tensor, relative_bias: nn.Embedding) -> torch.Tensor:
        norm = self.norm1(x)
        x = x + F.dropout(self.self_attn(norm, norm, mask, causal=True,
                                         relative_bias=relative_bias), self.dropout, self.training)
        x = x + F.dropout(self.cross_attn(self.norm2(x), memory, memory_mask), self.dropout, self.training)
        x = x + F.dropout(self.ff(self.norm3(x)), self.dropout, self.training)
        return x * mask.unsqueeze(-1)


class Charformer(nn.Module):
    architecture_id = "charformer"
    PRESETS = {
        "tiny": {"d_model": 64, "num_layers": 2, "num_heads": 4, "intermediate_size": 128},
        "small": {"d_model": 256, "num_layers": 6, "num_heads": 8, "intermediate_size": 1024},
        "base": {"d_model": 768, "num_layers": 12, "num_heads": 12, "intermediate_size": 3072},
    }
    BUDGET_KNOBS = {"d_model": range(32, 193, 16)}

    def __init__(self, vocab_size: int = 258, context_length: int = 2048, d_model: int = 128,
                 num_layers: int = 4, num_heads: int = 4, intermediate_size: int = 512,
                 max_block_size: int = 4, downsample_rate: int = 4, dropout: float = 0.0,
                 feed_forward: str = "relu", pad_token_id: int = 0,
                 score_calibration: bool = False) -> None:
        super().__init__()
        if vocab_size < 3 or context_length <= 0 or d_model <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError("vocab_size, context_length, dimensions, layers and heads must be positive")
        if d_model % num_heads or intermediate_size <= 0 or not 0 <= dropout < 1:
            raise ValueError("invalid attention geometry, intermediate_size or dropout")
        self.vocab_size, self.context_length, self.pad_token_id = vocab_size, context_length, pad_token_id
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.byte_position_conv = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)
        self.gbst = GradientSubwordBlock(d_model, max_block_size, downsample_rate,
                                         score_calibration=score_calibration)
        self.encoder_relative_bias = nn.Embedding(32, num_heads)
        self.decoder_relative_bias = nn.Embedding(32, num_heads)
        self.encoder = nn.ModuleList([_EncoderLayer(d_model, num_heads, intermediate_size, dropout, feed_forward)
                                      for _ in range(num_layers)])
        self.decoder = nn.ModuleList([_DecoderLayer(d_model, num_heads, intermediate_size, dropout, feed_forward)
                                      for _ in range(num_layers)])
        self.encoder_norm = _RMSNorm(d_model)
        self.decoder_norm = _RMSNorm(d_model)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        nn.init.normal_(self.encoder_relative_bias.weight, std=self.d_model ** -0.5)
        nn.init.normal_(self.decoder_relative_bias.weight, std=self.d_model ** -0.5)
        for layer in self.encoder:
            layer.attn.initialize_t5(self.d_model)
        for layer in self.decoder:
            layer.self_attn.initialize_t5(self.d_model)
            layer.cross_attn.initialize_t5(self.d_model)
        nn.init.zeros_(self.output_bias)

    def encode(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() != 2 or input_ids.size(1) > self.context_length:
            raise ValueError("input_ids must be [batch, sequence] within context_length")
        mask = input_ids.ne(self.pad_token_id) if mask is None else mask.bool()
        if mask.shape != input_ids.shape or torch.any(~mask.any(1)):
            raise ValueError("each input requires a nonempty matching mask")
        embedded = self.embedding(input_ids)
        positional = self.byte_position_conv(embedded.transpose(1, 2)).transpose(1, 2)
        embedded = (embedded + F.gelu(positional)) * mask.unsqueeze(-1)
        x, down_mask = self.gbst(embedded, mask)
        x = F.dropout(x, self.dropout, self.training)
        for layer in self.encoder:
            x = layer(x, down_mask, self.encoder_relative_bias)
        return self.encoder_norm(x), down_mask

    def forward_features(self, input_ids: torch.Tensor,
                         mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(input_ids, mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        memory, memory_mask = self.encode(src)
        if tgt.dim() != 2 or tgt.size(1) > self.context_length:
            raise ValueError("tgt must be [batch, sequence] within context_length")
        target_mask = tgt.ne(self.pad_token_id)
        target_mask[:, 0] = True
        if torch.any(~target_mask.any(1)):
            raise ValueError("each decoder input requires a nonempty matching mask")
        x = F.dropout(self.embedding(tgt), self.dropout, self.training)
        for layer in self.decoder:
            x = layer(x, memory, target_mask, memory_mask, self.decoder_relative_bias)
        return F.linear(self.decoder_norm(x), self.embedding.weight, self.output_bias)


def byte_encode(text: str) -> list[int]:
    return [2 + value for value in text.encode("utf-8")]
