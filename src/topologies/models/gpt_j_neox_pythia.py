from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


CONFIGS = {
    "gpt-j-6b": {"family": "gptj", "vocab_size": 50400, "context_length": 2048, "d_model": 4096, "num_layers": 28, "num_heads": 16, "rotary_dim": 64, "intermediate_size": 16384, "parallel": True, "shared_norm": True, "attention_bias": False, "mlp_bias": True, "lm_head_bias": True, "norm_eps": 1e-5, "scale_attention": True, "tie_embeddings": False, "activation": "gelu_new"},
    "gpt-neox-20b": {"family": "neox", "vocab_size": 50432, "context_length": 2048, "d_model": 6144, "num_layers": 44, "num_heads": 64, "rotary_dim": 24, "intermediate_size": 24576, "parallel": True, "shared_norm": False, "attention_bias": True, "qkv_bias": True, "mlp_bias": True, "lm_head_bias": False, "norm_eps": 1e-5, "scale_attention": True, "tie_embeddings": False, "activation": "gelu_fast"},
    "pythia-70m": {"family": "neox", "vocab_size": 50304, "context_length": 2048, "d_model": 512, "num_layers": 6, "num_heads": 8, "rotary_dim": 16, "intermediate_size": 2048, "parallel": True, "shared_norm": False, "attention_bias": True, "qkv_bias": True, "mlp_bias": True, "lm_head_bias": False, "norm_eps": 1e-5, "scale_attention": True, "tie_embeddings": False, "activation": "gelu"},
    "pythia-160m": {"family": "neox", "vocab_size": 50304, "context_length": 2048, "d_model": 768, "num_layers": 12, "num_heads": 12, "rotary_dim": 16, "intermediate_size": 3072, "parallel": True, "shared_norm": False, "attention_bias": True, "qkv_bias": True, "mlp_bias": True, "lm_head_bias": False, "norm_eps": 1e-5, "scale_attention": True, "tie_embeddings": False, "activation": "gelu"},
}


def _rotate_half(x):
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def _rotary(x, positions, rotary_dim, layout, theta):
    if rotary_dim == 0:
        return x
    inv = theta ** (-torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim)
    phase = positions.float()[..., None] * inv
    phase = torch.cat((phase, phase), -1) if layout == "half" else phase.repeat_interleave(2, -1)
    cos, sin = phase.cos().to(x.dtype)[:, None], phase.sin().to(x.dtype)[:, None]
    part, tail = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated = part * cos + (_rotate_half(part) if layout == "half" else torch.stack((-part[..., 1::2], part[..., 0::2]), -1).flatten(-2)) * sin
    return torch.cat((rotated, tail), -1)


class GPTJNeoXPythia(nn.Module):
    architecture_id = "gpt-j-neox-pythia"
    PRESETS = {"tiny": {"preset": "tiny"}, "small": {"preset": "small"}, "base": {"preset": "base"}, **{k: {"native_config": k} for k in CONFIGS}}
    BUDGET_KNOBS = {"d_model": range(32, 257, 8)}

    def __init__(self, vocab_size=128, context_length=128, d_model=64, num_layers=2, num_heads=4, dropout=0.0, family="gptj", preset=None, native_config=None, rotary_dim=None, rotary_theta=10000.0, attention_bias=None, mlp_bias=True, lm_head_bias=None, scale_attention=True, tie_embeddings=False, parallel=True, shared_norm=None, activation="gelu", intermediate_size=None, norm_eps=1e-5, pad_token_id=None):
        super().__init__()
        if preset is not None:
            if preset not in {"tiny", "small", "base"}:
                raise ValueError("preset must be tiny, small, or base")
            d_model, num_layers, num_heads = {"tiny": (32, 2, 4), "small": (128, 4, 4), "base": (256, 6, 8)}[preset]
        if native_config is not None:
            if native_config not in CONFIGS:
                raise ValueError("unknown native_config")
            cfg = CONFIGS[native_config]
            family, vocab_size, context_length = cfg["family"], cfg["vocab_size"], cfg["context_length"]
            d_model, num_layers, num_heads = cfg["d_model"], cfg["num_layers"], cfg["num_heads"]
            rotary_dim, parallel, shared_norm = cfg["rotary_dim"], cfg["parallel"], cfg["shared_norm"]
            attention_bias, mlp_bias, lm_head_bias = cfg["attention_bias"], cfg["mlp_bias"], cfg["lm_head_bias"]
            intermediate_size, norm_eps = cfg["intermediate_size"], cfg["norm_eps"]
            scale_attention, tie_embeddings, activation = cfg["scale_attention"], cfg["tie_embeddings"], cfg["activation"]
        if family not in {"gptj", "neox"}:
            raise ValueError("family must be gptj or neox")
        if shared_norm is None:
            shared_norm = family == "gptj"
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (vocab_size, context_length, d_model, num_layers, num_heads)) or d_model % num_heads:
            raise ValueError("dimensions must be positive and width divisible by heads")
        head_dim = d_model // num_heads
        rotary_dim = head_dim if rotary_dim is None else rotary_dim
        if rotary_dim < 0 or rotary_dim > head_dim or rotary_dim % 2 or rotary_theta <= 0 or not 0 <= dropout < 1:
            raise ValueError("invalid rotary dimensions, theta, or dropout")
        self.family, self.vocab_size, self.context_length, self.d_model, self.pad_token_id = family, vocab_size, context_length, d_model, pad_token_id
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.native_config = native_config
        attention_bias = (family == "neox") if attention_bias is None else attention_bias
        lm_head_bias = (family == "gptj") if lm_head_bias is None else lm_head_bias
        intermediate_size = 4 * d_model if intermediate_size is None else intermediate_size
        self.layers = nn.ModuleList([GPTJNeoXBlock(d_model, num_heads, rotary_dim, rotary_theta, family, attention_bias, mlp_bias, scale_attention, parallel, shared_norm, activation, intermediate_size, norm_eps) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=lm_head_bias)
        if tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, tokens, mask=None, position_ids=None):
        if tokens.ndim != 2 or tokens.dtype != torch.long or not 0 < tokens.shape[1] <= self.context_length:
            raise ValueError("tokens must be LongTensor [batch,time] within context")
        valid = torch.ones_like(tokens, dtype=torch.bool) if mask is None else mask.bool()
        if valid.shape != tokens.shape or valid.device != tokens.device:
            raise ValueError("mask must match tokens")
        if position_ids is None:
            position_ids = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        if position_ids.shape != tokens.shape or position_ids.dtype != torch.long or position_ids.device != tokens.device:
            raise ValueError("position_ids must match tokens")
        x = self.dropout(self.embedding(tokens)) * valid.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, valid, position_ids)
        return self.lm_head(self.final_norm(x) * valid.unsqueeze(-1))


class GPTJNeoXBlock(nn.Module):
    def __init__(self, width, heads, rotary_dim, theta, family, attention_bias, mlp_bias, scale, parallel, shared, activation, intermediate_size=None, norm_eps=1e-5):
        super().__init__()
        self.heads, self.head_dim, self.rotary_dim, self.theta, self.family = heads, width // heads, rotary_dim, theta, family
        self.qkv = nn.Linear(width, 3 * width, bias=attention_bias) if family == "neox" else None
        self.q_proj = nn.Linear(width, width, bias=False) if family == "gptj" else None
        self.k_proj = nn.Linear(width, width, bias=False) if family == "gptj" else None
        self.v_proj = nn.Linear(width, width, bias=False) if family == "gptj" else None
        self.out = nn.Linear(width, width, bias=attention_bias)
        self.norm = nn.LayerNorm(width, eps=norm_eps)
        self.attn_norm = self.norm
        self.mlp_norm = self.norm if shared else nn.LayerNorm(width, eps=norm_eps)
        intermediate_size = 4 * width if intermediate_size is None else intermediate_size
        self.fc_in, self.fc_out = nn.Linear(width, intermediate_size, bias=mlp_bias), nn.Linear(intermediate_size, width, bias=mlp_bias)
        self.activation, self.scale, self.parallel = (nn.GELU(approximate="tanh") if activation in {"gelu_new", "gelu_fast"} else nn.GELU()), scale, parallel

    def forward(self, x, valid, positions):
        source = self.attn_norm(x)
        b, t, width = x.shape
        if self.family == "gptj":
            q, k, v = (proj(source).view(b, t, self.heads, self.head_dim).transpose(1, 2) for proj in (self.q_proj, self.k_proj, self.v_proj))
        else:
            q, k, v = self.qkv(source).view(b, t, self.heads, 3, self.head_dim).permute(3, 0, 2, 1, 4)
        layout = "interleaved" if self.family == "gptj" else "half"
        q, k = _rotary(q, positions, self.rotary_dim, layout, self.theta), _rotary(k, positions, self.rotary_dim, layout, self.theta)
        scores = q @ k.transpose(-2, -1)
        if self.scale:
            scores = scores / math.sqrt(self.head_dim)
        causal = torch.ones((t, t), device=x.device, dtype=torch.bool).tril()
        allowed = causal[None, None] & valid[:, None, None, :]
        allowed = allowed | (~allowed.any(-1, keepdim=True) & causal[None, None])
        probs = F.softmax(scores.masked_fill(~allowed, torch.finfo(scores.dtype).min).float(), -1).to(scores.dtype)
        a = self.out((probs @ v).transpose(1, 2).reshape(b, t, width)) * valid.unsqueeze(-1)
        if self.parallel:
            m = self.fc_out(self.activation(self.fc_in(self.mlp_norm(x))))
            x = x + a + m
        else:
            x = x + a
            m = self.fc_out(self.activation(self.fc_in(self.mlp_norm(x))))
            x = x + m
        return x * valid.unsqueeze(-1)
