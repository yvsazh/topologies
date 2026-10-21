from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(width)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def apply_rotary(x, positions, theta):
    dim = x.shape[-1]
    inv = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    phase = positions.float()[..., None] * inv
    cos, sin = phase.cos().to(x.dtype)[:, None], phase.sin().to(x.dtype)[:, None]
    return torch.stack((x[..., ::2] * cos - x[..., 1::2] * sin,
                        x[..., ::2] * sin + x[..., 1::2] * cos), -1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(width, hidden, bias=False), nn.Linear(width, hidden, bias=False), nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Top2Router(nn.Module):
    def __init__(self, width, num_experts=8):
        super().__init__()
        if width <= 0 or num_experts < 2:
            raise ValueError("width positive and at least two experts required")
        self.num_experts = num_experts
        self.gate = nn.Linear(width, num_experts, bias=False)

    def forward(self, x, mask=None):
        logits = self.gate(x).float()
        probs = logits.softmax(-1)
        indices = logits.topk(2, dim=-1).indices
        weights = probs.gather(-1, indices)
        weights = weights / weights.sum(-1, keepdim=True)
        valid = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device) if mask is None else mask.bool()
        weights = weights * valid.unsqueeze(-1)
        indices = indices.masked_fill(~valid.unsqueeze(-1), -1)
        p = probs.reshape(-1, self.num_experts)
        i, v = indices.reshape(-1, 2), valid.reshape(-1)
        if bool(v.any()):
            freq = F.one_hot(i[v], self.num_experts).float().sum(1).mean(0) / 2
            aux = self.num_experts * (p[v].mean(0) * freq).sum()
        else:
            aux = logits.sum() * 0
        return indices, weights, aux, logits


class SparseMoE(nn.Module):
    def __init__(self, width, hidden, experts):
        super().__init__()
        self.router = Top2Router(width, experts)
        self.experts = nn.ModuleList([SwiGLU(width, hidden) for _ in range(experts)])

    def forward(self, x, mask):
        indices, weights, aux, logits = self.router(x, mask)
        flat, ids, gates = x.reshape(-1, x.shape[-1]), indices.reshape(-1, 2), weights.reshape(-1, 2)
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            locations = (ids == expert_id).nonzero(as_tuple=False)
            if locations.numel():
                token_ids, slots = locations[:, 0], locations[:, 1]
                values = expert(flat.index_select(0, token_ids)) * gates[token_ids, slots].to(flat.dtype).unsqueeze(-1)
                output = output.index_add(0, token_ids, values)
        return output.view_as(x), aux, logits


class Attention(nn.Module):
    def __init__(self, width, heads, kv_heads, theta):
        super().__init__()
        self.heads, self.kv_heads, self.dim, self.theta = heads, kv_heads, width // heads, theta
        self.q, self.k, self.v, self.o = nn.Linear(width, width, False), nn.Linear(width, kv_heads * (width // heads), False), nn.Linear(width, kv_heads * (width // heads), False), nn.Linear(width, width, False)

    def forward(self, x, valid, positions):
        b, t, width = x.shape
        q = self.q(x).view(b, t, self.heads, self.dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.kv_heads, self.dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv_heads, self.dim).transpose(1, 2)
        q, k = apply_rotary(q, positions, self.theta), apply_rotary(k, positions, self.theta)
        if self.kv_heads != self.heads:
            k, v = k.repeat_interleave(self.heads // self.kv_heads, 1), v.repeat_interleave(self.heads // self.kv_heads, 1)
        order = torch.arange(t, device=x.device)
        allowed = (order[None, :] <= order[:, None])[None, None] & valid[:, None, None, :]
        allowed = allowed | (~valid[:, None, :, None] & torch.eye(t, device=x.device, dtype=torch.bool)[None, None])
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.dim)
        score = score.masked_fill(~allowed, torch.finfo(score.dtype).min)
        out = (score.float().softmax(-1).to(score.dtype) @ v).transpose(1, 2).reshape(b, t, width)
        return self.o(out) * valid.unsqueeze(-1)


class Block(nn.Module):
    def __init__(self, width, heads, kv_heads, hidden, experts, eps, theta):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(width, eps), Attention(width, heads, kv_heads, theta)
        self.ffn_norm, self.moe = RMSNorm(width, eps), SparseMoE(width, hidden, experts)

    def forward(self, x, valid, positions):
        x = (x + self.attn(self.attn_norm(x), valid, positions)) * valid.unsqueeze(-1)
        y, aux, logits = self.moe(self.ffn_norm(x), valid)
        return (x + y) * valid.unsqueeze(-1), aux, logits


class Mixtral(nn.Module):
    architecture_id = "mixtral"
    PRESETS = {"tiny": {"preset": "tiny"}, "small": {"preset": "small"}, "base": {"preset": "base"}, "mixtral-8x7b": {"size": "mixtral-8x7b"}}
    BUDGET_KNOBS = {"d_model": range(16, 257, 8)}

    def __init__(self, vocab_size=32000, context_length=32768, d_model=256, num_layers=4, num_heads=4, num_kv_heads=2, intermediate_size=704, num_experts=8, top_k=2, rope_theta=1_000_000.0, rms_norm_eps=1e-5, dropout=0.0, pad_token_id=None, preset=None, size=None):
        super().__init__()
        if preset is not None:
            if preset not in {"tiny", "small", "base"}: raise ValueError("preset must be tiny, small, or base")
            d_model, num_layers, num_heads, num_kv_heads, intermediate_size, num_experts = {"tiny": (32, 2, 4, 2, 88, 4), "small": (128, 4, 4, 2, 352, 8), "base": (256, 6, 8, 4, 704, 8)}[preset]
        if size is not None:
            if size != "mixtral-8x7b": raise ValueError("size must be mixtral-8x7b")
            vocab_size, context_length, d_model, num_layers = 32000, 32768, 4096, 32
            num_heads, num_kv_heads, intermediate_size, num_experts = 32, 8, 14336, 8
            rope_theta, rms_norm_eps = 1_000_000.0, 1e-5
        vals = (vocab_size, context_length, d_model, num_layers, num_heads, num_kv_heads, intermediate_size, num_experts)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in vals): raise ValueError("model dimensions must be positive integers")
        if top_k != 2 or num_experts < 2 or d_model % num_heads or num_heads % num_kv_heads or (d_model // num_heads) % 2: raise ValueError("require top_k=2, valid expert count and divisible GQA/even head geometry")
        if not 0 <= dropout < 1 or rope_theta <= 0 or rms_norm_eps <= 0: raise ValueError("invalid dropout, RoPE theta or RMSNorm epsilon")
        if pad_token_id is not None and not 0 <= pad_token_id < vocab_size: raise ValueError("pad_token_id outside vocabulary")
        self.vocab_size, self.context_length, self.d_model, self.pad_token_id = vocab_size, context_length, d_model, pad_token_id
        self.num_experts, self.top_k, self.num_heads, self.num_kv_heads = num_experts, top_k, num_heads, num_kv_heads
        self.token_embedding, self.dropout = nn.Embedding(vocab_size, d_model), nn.Dropout(dropout)
        self.layers = nn.ModuleList([Block(d_model, num_heads, num_kv_heads, intermediate_size, num_experts, rms_norm_eps, rope_theta) for _ in range(num_layers)])
        self.norm, self.lm_head = RMSNorm(d_model, rms_norm_eps), nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, mean=0., std=.02)

    def forward_features(self, tokens, mask=None, position_ids=None):
        if tokens.ndim != 2 or tokens.dtype != torch.long: raise ValueError("tokens must be LongTensor [batch,time]")
        b, t = tokens.shape
        if not b or not 0 < t <= self.context_length: raise ValueError("tokens exceed context or empty")
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.vocab_size): raise ValueError("token outside vocabulary")
        valid = torch.ones_like(tokens, dtype=torch.bool) if mask is None else mask.to(torch.bool)
        if valid.shape != tokens.shape or valid.device != tokens.device: raise ValueError("mask must match tokens")
        if position_ids is None: position_ids = (valid.long().cumsum(-1) - 1).clamp_min(0)
        if position_ids.shape != tokens.shape or position_ids.dtype != torch.long or position_ids.device != tokens.device or (position_ids.numel() and (position_ids.min() < 0 or position_ids.max() >= self.context_length)): raise ValueError("invalid position_ids")
        x = self.dropout(self.token_embedding(tokens)) * valid.unsqueeze(-1)
        router_logits = []
        for layer in self.layers:
            x, _, logits = layer(x, valid, position_ids)
            router_logits.append(logits)
        return self.norm(x) * valid.unsqueeze(-1), router_logits

    def forward(self, tokens, mask=None, position_ids=None):
        features, _ = self.forward_features(tokens, mask, position_ids)
        return self.lm_head(features)

    def forward_with_aux(self, tokens, mask=None, position_ids=None):
        features, router_logits = self.forward_features(tokens, mask, position_ids)
        valid = torch.ones(tokens.shape, dtype=torch.bool, device=tokens.device) if mask is None else mask.bool()
        all_logits = torch.cat([layer_logits[valid] for layer_logits in router_logits], dim=0)
        if all_logits.shape[0]:
            probabilities = all_logits.softmax(-1)
            assignments = all_logits.topk(self.top_k, dim=-1).indices
            frequency = torch.nn.functional.one_hot(assignments, self.num_experts).float().sum((0, 1)) / all_logits.shape[0]
            importance = probabilities.mean(0)
            aux = self.num_experts * (frequency * importance).sum()
        else:
            aux = sum(logits.sum() for logits in router_logits) * 0
        return self.lm_head(features), aux
