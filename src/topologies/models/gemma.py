from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GemmaRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * (1 + self.weight)


def _rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    half = x.shape[-1] // 2
    inv_freq = theta ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    phase = positions.float()[..., None] * inv_freq
    cos, sin = phase.cos().to(x.dtype)[:, None], phase.sin().to(x.dtype)[:, None]
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class GemmaAttention(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int, head_dim: int, query_scalar: float,
                 theta: float, window: int | None, attn_cap: float | None, eps: float,
                 qk_norm: bool) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.query_scalar, self.theta, self.window, self.attn_cap = query_scalar, theta, window, attn_cap
        self.q_proj = nn.Linear(width, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(width, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(width, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, width, bias=False)
        self.q_norm = GemmaRMSNorm(head_dim, eps) if qk_norm else nn.Identity()
        self.k_norm = GemmaRMSNorm(head_dim, eps) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, valid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        q = self.q_proj(x).view(batch, time, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q = _rope(q, positions, self.theta)
        k = _rope(k, positions, self.theta)
        if self.heads != self.kv_heads:
            ratio = self.heads // self.kv_heads
            k, v = k.repeat_interleave(ratio, 1), v.repeat_interleave(ratio, 1)
        scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.query_scalar)
        qpos = torch.arange(time, device=x.device)
        kpos = torch.arange(time, device=x.device)
        allowed = kpos[None, :] <= qpos[:, None]
        if self.window is not None:
            allowed &= kpos[None, :] > qpos[:, None] - self.window
        allowed = allowed[None, None] & valid[:, None, None, :]
        allowed |= ~allowed.any(-1, keepdim=True) & (kpos[None, None, None, :] == qpos[None, None, :, None])
        if self.attn_cap is not None:
            scores = self.attn_cap * torch.tanh(scores / self.attn_cap)
        probs = F.softmax(scores.masked_fill(~allowed, torch.finfo(scores.dtype).min), dim=-1).to(v.dtype)
        result = (probs @ v).transpose(1, 2).reshape(batch, time, self.heads * self.head_dim)
        return self.o_proj(result) * valid.unsqueeze(-1)


class GemmaBlock(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int, head_dim: int, hidden: int,
                 query_scalar: float, theta: float, window: int | None, attn_cap: float | None,
                 eps: float, qk_norm: bool) -> None:
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(width, eps)
        self.post_attention_layernorm = GemmaRMSNorm(width, eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(width, eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(width, eps)
        self.attention = GemmaAttention(width, heads, kv_heads, head_dim, query_scalar,
                                        theta, window, attn_cap, eps, qk_norm)
        self.gate_proj = nn.Linear(width, hidden, bias=False)
        self.up_proj = nn.Linear(width, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        attn = self.attention(self.input_layernorm(x), valid, positions)
        x = (x + self.post_attention_layernorm(attn)) * valid.unsqueeze(-1)
        hidden = F.gelu(self.gate_proj(self.pre_feedforward_layernorm(x)), approximate="tanh")
        hidden = hidden * self.up_proj(self.pre_feedforward_layernorm(x))
        return (x + self.post_feedforward_layernorm(self.down_proj(hidden))) * valid.unsqueeze(-1)


class Gemma(nn.Module):
    architecture_id = "gemma"
    PRESETS = {"tiny": {"preset": "tiny"}, "small": {"preset": "small"}, "base": {"preset": "base"},
               "gemma2-2b": {"size": "gemma2-2b"}, "gemma2-9b": {"size": "gemma2-9b"},
               "gemma3-1b": {"size": "gemma3-1b"}, "gemma3-4b": {"size": "gemma3-4b"}}
    BUDGET_KNOBS = {"hidden_size": range(16, 257, 8)}
    CONFIGS = {
        "gemma2-2b": {"hidden": 2304, "layers": 26, "heads": 8, "kv_heads": 4, "head_dim": 256, "intermediate": 9216, "context": 8192, "window": 4096, "schedule": "alternate_local_global", "rope_global": 10000.0, "rope_local": 10000.0, "qk_norm": False, "attn_cap": 50.0, "final_cap": 30.0, "query_scalar": 256.0, "vocab": 256000},
        "gemma2-9b": {"hidden": 3584, "layers": 42, "heads": 16, "kv_heads": 8, "head_dim": 256, "intermediate": 14336, "context": 8192, "window": 4096, "schedule": "alternate_local_global", "rope_global": 10000.0, "rope_local": 10000.0, "qk_norm": False, "attn_cap": 50.0, "final_cap": 30.0, "query_scalar": 256.0, "vocab": 256000},
        "gemma3-1b": {"hidden": 1152, "layers": 26, "heads": 4, "kv_heads": 1, "head_dim": 256, "intermediate": 4304, "context": 32768, "window": 1024, "schedule": "five_local_one_global", "rope_global": 1000000.0, "rope_local": 10000.0, "qk_norm": True, "attn_cap": None, "final_cap": None, "query_scalar": 256.0, "vocab": 262208},
        "gemma3-4b": {"hidden": 2560, "layers": 34, "heads": 8, "kv_heads": 4, "head_dim": 256, "intermediate": 10240, "context": 131072, "window": 1024, "schedule": "five_local_one_global", "rope_global": 1000000.0, "rope_local": 10000.0, "qk_norm": True, "attn_cap": None, "final_cap": None, "query_scalar": 256.0, "vocab": 262208},
    }

    def __init__(self, vocab_size: int = 256, context_length: int = 128, hidden_size: int = 64,
                 num_layers: int = 4, num_heads: int = 4, num_kv_heads: int = 2,
                 head_dim: int = 16, intermediate_size: int = 128, window_size: int = 8,
                 layer_types: tuple[str, ...] | None = None, rope_global_theta: float = 10000.0,
                 rope_local_theta: float = 10000.0, query_pre_attn_scalar: float | None = None,
                 attn_logit_softcapping: float | None = None, final_logit_softcapping: float | None = None,
                 qk_norm: bool = False, embedding_scale: bool = True, rms_norm_eps: float = 1e-6,
                 dropout: float = 0.0, pad_token_id: int | None = 0,
                 preset: str | None = None, size: str | None = None) -> None:
        super().__init__()
        if preset is not None:
            if preset not in {"tiny", "small", "base"}:
                raise ValueError("preset must be tiny, small, or base")
            hidden_size, num_layers, num_heads, num_kv_heads, head_dim, intermediate_size, window_size = {
                "tiny": (32, 2, 4, 2, 8, 64, 4), "small": (128, 4, 4, 2, 32, 256, 16),
                "base": (256, 6, 8, 4, 32, 512, 32)}[preset]
        config = None
        if size is not None:
            if size not in self.CONFIGS:
                raise ValueError(f"unsupported native size {size!r}")
            config = self.CONFIGS[size]
            vocab_size, context_length = config["vocab"], config["context"]
            hidden_size, num_layers, num_heads, num_kv_heads = config["hidden"], config["layers"], config["heads"], config["kv_heads"]
            head_dim, intermediate_size, window_size = config["head_dim"], config["intermediate"], config["window"]
            rope_global_theta, rope_local_theta = config["rope_global"], config["rope_local"]
            query_pre_attn_scalar, attn_logit_softcapping = config["query_scalar"], config["attn_cap"]
            final_logit_softcapping, qk_norm = config["final_cap"], config["qk_norm"]
            layer_types = (tuple("local" if i % 2 == 0 else "global" for i in range(num_layers))
                           if config["schedule"] == "alternate_local_global" else
                            tuple("global" if i % 6 == 5 else "local" for i in range(num_layers)))
        dims = (vocab_size, context_length, hidden_size, num_layers, num_heads, num_kv_heads,
                head_dim, intermediate_size, window_size)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in dims):
            raise ValueError("model dimensions must be positive integers")
        if num_heads % num_kv_heads or head_dim % 2:
            raise ValueError("heads must divide into KV heads and head_dim must be even")
        if layer_types is None:
            layer_types = tuple("local" if index % 2 == 0 else "global" for index in range(num_layers))
        if len(layer_types) != num_layers or any(kind not in {"local", "global"} for kind in layer_types):
            raise ValueError("layer_types must contain local/global for every layer")
        if not 0 <= dropout < 1 or rms_norm_eps <= 0 or rope_global_theta <= 0 or rope_local_theta <= 0:
            raise ValueError("dropout must be in [0,1), norm epsilon and RoPE bases positive")
        if pad_token_id is not None and not 0 <= pad_token_id < vocab_size:
            raise ValueError("pad_token_id must be in vocabulary")
        self.vocab_size, self.context_length, self.hidden_size = vocab_size, context_length, hidden_size
        self.layer_types, self.pad_token_id = tuple(layer_types), pad_token_id
        self.embedding_scale = math.sqrt(hidden_size) if embedding_scale else 1.0
        self.final_logit_softcapping = final_logit_softcapping
        scalar = query_pre_attn_scalar or head_dim
        caps = (attn_logit_softcapping, final_logit_softcapping)
        if scalar <= 0 or any(cap is not None and cap <= 0 for cap in caps):
            raise ValueError("query scalar and enabled softcap values must be positive")
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([GemmaBlock(hidden_size, num_heads, num_kv_heads, head_dim,
            intermediate_size, scalar, rope_local_theta if kind == "local" else rope_global_theta,
            window_size if kind == "local" else None, attn_logit_softcapping, rms_norm_eps, qk_norm)
            for kind in self.layer_types])
        self.norm = GemmaRMSNorm(hidden_size, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward_features(self, tokens: torch.Tensor, mask: torch.Tensor | None = None,
                         position_ids: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.dtype != torch.long or not 0 < tokens.shape[1] <= self.context_length:
            raise ValueError("tokens must be LongTensor [batch,time] within context_length")
        valid = torch.ones_like(tokens, dtype=torch.bool) if mask is None else mask.bool()
        if valid.shape != tokens.shape or valid.device != tokens.device:
            raise ValueError("mask must match token shape and device")
        if position_ids is None:
            position_ids = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        if position_ids.shape != tokens.shape or position_ids.dtype != torch.long or position_ids.device != tokens.device:
            raise ValueError("position_ids must match tokens")
        x = self.dropout(self.embed_tokens(tokens) * self.embedding_scale) * valid.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, valid, position_ids)
        return self.norm(x) * valid.unsqueeze(-1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.lm_head(self.forward_features(tokens, mask, position_ids))
        if self.final_logit_softcapping is not None:
            logits = self.final_logit_softcapping * torch.tanh(logits / self.final_logit_softcapping)
        return logits
