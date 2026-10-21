from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class PhiRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def _rotary(x: torch.Tensor, positions: torch.Tensor, theta: float,
            original_context: int, short_factor: tuple[float, ...] | None,
            long_factor: tuple[float, ...] | None, attention_factor: float) -> torch.Tensor:
    half = x.shape[-1] // 2
    if short_factor is None:
        factors = torch.ones(half, device=x.device, dtype=torch.float32)
    else:
        selected = long_factor if positions.max().item() + 1 > original_context else short_factor
        factors = torch.tensor(selected, device=x.device, dtype=torch.float32)
    inv_freq = theta ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half) / factors
    phase = positions.float()[..., None] * inv_freq
    cos = phase.cos().to(x.dtype)[:, None] * attention_factor
    sin = phase.sin().to(x.dtype)[:, None] * attention_factor
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class PhiAttention(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int, head_dim: int, bias: bool,
                 theta: float, original_context: int, short_factor: tuple[float, ...] | None,
                 long_factor: tuple[float, ...] | None, attention_factor: float,
                 sliding_window: int | None) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.theta, self.original_context = theta, original_context
        self.short_factor, self.long_factor = short_factor, long_factor
        self.attention_factor, self.sliding_window = attention_factor, sliding_window
        self.qkv_proj = nn.Linear(width, (heads + 2 * kv_heads) * head_dim, bias=bias)
        self.o_proj = nn.Linear(heads * head_dim, width, bias=False)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        packed = self.qkv_proj(x)
        q_width = self.heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        q, k, v = packed.split((q_width, kv_width, kv_width), dim=-1)
        q = q.view(batch, time, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
        q = _rotary(q, positions, self.theta, self.original_context, self.short_factor,
                    self.long_factor, self.attention_factor)
        k = _rotary(k, positions, self.theta, self.original_context, self.short_factor,
                    self.long_factor, self.attention_factor)
        if self.heads != self.kv_heads:
            repeats = self.heads // self.kv_heads
            k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        qpos, kpos = torch.arange(time, device=x.device), torch.arange(time, device=x.device)
        allowed = kpos[None, :] <= qpos[:, None]
        if self.sliding_window is not None:
            allowed &= kpos[None, :] > qpos[:, None] - self.sliding_window
        allowed = allowed[None, None] & valid[:, None, None, :]
        allowed |= ~allowed.any(-1, keepdim=True) & (kpos[None, None, None, :] == qpos[None, None, :, None])
        probs = F.softmax(scores.masked_fill(~allowed, torch.finfo(scores.dtype).min).float(), -1).to(v.dtype)
        output = (probs @ v).transpose(1, 2).reshape(batch, time, q_width)
        return self.o_proj(output) * valid.unsqueeze(-1)


class PhiBlock(nn.Module):
    def __init__(self, width: int, heads: int, kv_heads: int, head_dim: int, intermediate: int,
                 eps: float, bias: bool, theta: float, original_context: int,
                 short_factor: tuple[float, ...] | None, long_factor: tuple[float, ...] | None,
                 attention_factor: float, sliding_window: int | None) -> None:
        super().__init__()
        self.input_layernorm = PhiRMSNorm(width, eps)
        self.attention = PhiAttention(width, heads, kv_heads, head_dim, bias, theta,
            original_context, short_factor, long_factor, attention_factor, sliding_window)
        self.post_attention_layernorm = PhiRMSNorm(width, eps)
        self.gate_up_proj = nn.Linear(width, 2 * intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = (x + self.attention(self.input_layernorm(x), valid, positions)) * valid.unsqueeze(-1)
        gate, up = self.gate_up_proj(self.post_attention_layernorm(x)).chunk(2, dim=-1)
        return (x + self.down_proj(F.silu(gate) * up)) * valid.unsqueeze(-1)


class Phi(nn.Module):
    architecture_id = "phi"
    PRESETS = {"tiny": {"preset": "tiny"}, "small": {"preset": "small"}, "base": {"preset": "base"},
               "phi3-mini-4k": {"size": "phi3-mini-4k"}, "phi3-mini-128k": {"size": "phi3-mini-128k"},
               "phi4-14b": {"size": "phi4-14b"}}
    BUDGET_KNOBS = {"hidden_size": range(16, 257, 8)}
    CONFIGS = {
        "phi3-mini-4k": {"hidden": 3072, "intermediate": 8192, "layers": 32, "heads": 32, "kv_heads": 32, "head_dim": 96, "context": 4096, "original_context": 4096, "theta": 10000.0, "bias": False, "window": 2047, "rope": None, "vocab": 32064},
        "phi3-mini-128k": {"hidden": 3072, "intermediate": 8192, "layers": 32, "heads": 32, "kv_heads": 32, "head_dim": 96, "context": 131072, "original_context": 4096, "theta": 10000.0, "bias": False, "window": 262144, "rope": "longrope", "vocab": 32064},
        "phi4-14b": {"hidden": 5120, "intermediate": 17920, "layers": 40, "heads": 40, "kv_heads": 10, "head_dim": 128, "context": 16384, "original_context": 16384, "theta": 250000.0, "bias": False, "window": None, "rope": None, "vocab": 100352},
    }
    PHI3_MINI_128K_SHORT = (1.1, 1.1, 1.1, 1.3000000000000003, 1.3500000000000003, 1.3500000000000003, 1.4000000000000004, 1.5500000000000005, *([2.000000000000001] * 17), *([2.0500000000000007] * 6), 2.1000000000000005, 2.1000000000000005, 2.1500000000000004, *([2.25] * 5), 2.3999999999999995, 2.4499999999999993, 2.499999999999999, 2.6999999999999984, 2.6999999999999984, 2.7499999999999982, 2.799999999999998, 2.8999999999999977, 3.0499999999999977)
    PHI3_MINI_128K_LONG = (1.0700000524520874, 1.1200000047683716, 1.149999976158142, 1.4199999570846558, 1.5699999332427979, 1.7999999523162842, 2.129999876022339, 2.129999876022339, 3.0099999904632568, 5.910000324249268, 6.950000286102295, 9.070000648498535, 9.930000305175781, 10.710000038146973, 11.130000114440918, 14.609999656677246, 15.409998893737793, 19.809999465942383, 37.279998779296875, 38.279998779296875, 38.599998474121094, 40.12000274658203, 46.20000457763672, 50.940006256103516, 53.66000747680664, 54.9373893737793, 56.89738845825195, 57.28738784790039, 59.98738479614258, 60.86738586425781, 60.887386322021484, 61.71739196777344, 62.91739273071289, 62.957393646240234, 63.41739273071289, 63.8173942565918, 63.83739471435547, 63.897396087646484, 63.93739700317383, 64.06739807128906, 64.11434936523438, 64.12435150146484, 64.15435028076172, 64.19435119628906, 64.24435424804688, 64.57435607910156, 64.69000244140625, 64.76000213623047)

    def __init__(self, vocab_size: int = 256, context_length: int = 128, hidden_size: int = 64,
                 intermediate_size: int = 128, num_layers: int = 2, num_heads: int = 4,
                 num_kv_heads: int = 2, head_dim: int = 16, attention_bias: bool = False,
                 rope_theta: float = 10000.0, original_context: int | None = None,
                 rope_short_factor: tuple[float, ...] | None = None,
                 rope_long_factor: tuple[float, ...] | None = None,
                 rope_attention_factor: float | None = None, sliding_window: int | None = None,
                 rms_norm_eps: float = 1e-5, dropout: float = 0.0,
                 tie_word_embeddings: bool = False, pad_token_id: int | None = None,
                 preset: str | None = None, size: str | None = None) -> None:
        super().__init__()
        if preset is not None:
            if preset not in {"tiny", "small", "base"}:
                raise ValueError("preset must be tiny, small, or base")
            hidden_size, intermediate_size, num_layers, num_heads, num_kv_heads, head_dim = {
                "tiny": (32, 64, 2, 4, 2, 8), "small": (128, 256, 4, 4, 2, 32),
                "base": (256, 512, 6, 8, 4, 32)}[preset]
        if size is not None:
            if size not in self.CONFIGS:
                raise ValueError(f"unsupported native size {size!r}")
            cfg = self.CONFIGS[size]
            vocab_size, context_length = cfg["vocab"], cfg["context"]
            hidden_size, intermediate_size, num_layers = cfg["hidden"], cfg["intermediate"], cfg["layers"]
            num_heads, num_kv_heads, head_dim = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
            attention_bias, rope_theta, original_context = cfg["bias"], cfg["theta"], cfg["original_context"]
            sliding_window = cfg["window"]
            if cfg["rope"] == "longrope":
                rope_short_factor, rope_long_factor = self.PHI3_MINI_128K_SHORT, self.PHI3_MINI_128K_LONG
                rope_attention_factor = math.sqrt(1 + math.log(context_length / original_context) / math.log(original_context))
        values = (vocab_size, context_length, hidden_size, intermediate_size, num_layers,
                  num_heads, num_kv_heads, head_dim)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
            raise ValueError("model dimensions must be positive integers")
        if num_heads % num_kv_heads or head_dim % 2:
            raise ValueError("num_heads must divide by num_kv_heads and head_dim must be even")
        if not 0 <= dropout < 1 or rms_norm_eps <= 0 or rope_theta <= 0:
            raise ValueError("dropout must be in [0,1), and norm epsilon/RoPE base positive")
        half = head_dim // 2
        if (rope_short_factor is None) != (rope_long_factor is None):
            raise ValueError("LongRoPE requires both short_factor and long_factor")
        if rope_short_factor is not None:
            if len(rope_short_factor) != half or len(rope_long_factor) != half or any(v <= 0 for v in (*rope_short_factor, *rope_long_factor)):
                raise ValueError("LongRoPE factors must be positive and match half the attention head dimension")
        if pad_token_id is not None and not 0 <= pad_token_id < vocab_size:
            raise ValueError("pad_token_id must be in vocabulary")
        self.vocab_size, self.context_length, self.hidden_size, self.pad_token_id = vocab_size, context_length, hidden_size, pad_token_id
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([PhiBlock(hidden_size, num_heads, num_kv_heads, head_dim,
            intermediate_size, rms_norm_eps, attention_bias, rope_theta, original_context or context_length,
            rope_short_factor, rope_long_factor, rope_attention_factor or 1.0, sliding_window)
            for _ in range(num_layers)])
        self.norm = PhiRMSNorm(hidden_size, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward_features(self, tokens: torch.Tensor, mask: torch.Tensor | None = None,
                         position_ids: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.dtype != torch.long or not 0 < tokens.shape[1] <= self.context_length:
            raise ValueError("tokens must be LongTensor [batch,time] within context_length")
        valid = torch.ones_like(tokens, dtype=torch.bool) if mask is None else mask.bool()
        if valid.shape != tokens.shape or valid.device != tokens.device:
            raise ValueError("mask must match tokens shape and device")
        if position_ids is None:
            position_ids = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        if position_ids.shape != tokens.shape or position_ids.dtype != torch.long or position_ids.device != tokens.device:
            raise ValueError("position_ids must match tokens")
        x = self.embed_dropout(self.embed_tokens(tokens)) * valid.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, valid, position_ids)
        return self.norm(x) * valid.unsqueeze(-1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.lm_head(self.forward_features(tokens, mask, position_ids))
