from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

_PRESETS = {
    "gpt3": {"tiny": (64, 2, 2), "small": (256, 4, 4), "base": (768, 12, 12)},
    "opt": {"tiny": (64, 2, 2), "small": (256, 4, 4), "base": (768, 12, 12)},
    "bloom": {"tiny": (64, 2, 2), "small": (256, 4, 4), "base": (768, 12, 12)},
    "falcon": {"tiny": (64, 2, 2), "small": (256, 4, 4), "base": (768, 12, 12)},
}

REFERENCE_CONFIGS = {
    "opt-350m": {"family": "opt", "d_model": 1024, "num_layers": 24, "num_heads": 16,
                 "opt_embedding_dim": 512, "opt_post_layer_norm": True},
    "bloom-560m": {"family": "bloom", "d_model": 1024, "num_layers": 24, "num_heads": 16,
                   "pad_token_id": 3, "bos_token_id": 1, "eos_token_id": 2},
    "falcon-7b": {"family": "falcon", "d_model": 4544, "num_layers": 32, "num_heads": 71,
                  "falcon_kv_heads": 1, "falcon_shared_input_norm": True,
                  "falcon_parallel": True, "falcon_rope": True},
    "falcon-40b": {"family": "falcon", "d_model": 8192, "num_layers": 60, "num_heads": 128,
                   "falcon_kv_heads": 8, "falcon_shared_input_norm": False,
                   "falcon_parallel": True, "falcon_rope": True},
}


def alibi_slopes(heads: int, *, device=None, dtype=None) -> torch.Tensor:
    if heads < 1:
        raise ValueError("heads must be positive")

    def powers_of_two(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start**i for i in range(n)]

    p = 2 ** math.floor(math.log2(heads))
    values = powers_of_two(p)
    if p != heads:
        values += powers_of_two(2 * p)[0::2][: heads - p]
    return torch.tensor(values, device=device, dtype=dtype or torch.float32)


class DecoderConfig:
    _NAMES = ("family", "vocab_size", "context_length", "d_model", "num_layers", "num_heads",
              "dropout", "pad_token_id", "bos_token_id", "eos_token_id", "attention", "window_size",
              "falcon_kv_heads", "falcon_parallel", "falcon_shared_input_norm", "opt_embedding_dim",
              "opt_post_layer_norm", "falcon_rope")

    def __init__(self, *values):
        if len(values) != len(self._NAMES):
            raise ValueError("configuration field count mismatch")
        for name, value in zip(self._NAMES, values):
            setattr(self, name, value)


class SelfAttention(nn.Module):
    def __init__(self, c: DecoderConfig):
        super().__init__()
        self.c = c
        self.head_dim = c.d_model // c.num_heads
        self.kv_heads = (c.falcon_kv_heads or c.num_heads) if c.family == "falcon" else c.num_heads
        bias = c.family != "falcon"
        if c.family == "bloom":
            self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=True)
            self.q = self.k = self.v = None
        else:
            self.qkv = None
            self.q = nn.Linear(c.d_model, c.num_heads * self.head_dim, bias=bias)
            self.k = nn.Linear(c.d_model, self.kv_heads * self.head_dim, bias=bias)
            self.v = nn.Linear(c.d_model, self.kv_heads * self.head_dim, bias=bias)
        self.out = nn.Linear(c.d_model, c.d_model, bias=c.family != "falcon")

    def forward(self, x, valid, position_ids=None):
        b, t, d = x.shape
        if self.c.family == "bloom":
            q, k, v = self.qkv(x).view(b, t, 3, self.c.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            q = self.q(x).view(b, t, self.c.num_heads, self.head_dim).transpose(1, 2)
            k = self.k(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
            v = self.v(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.c.family == "falcon" and self.c.falcon_rope:
            q, k = _apply_rope(q, k, position_ids)
        if self.kv_heads != self.c.num_heads:
            repeat = self.c.num_heads // self.kv_heads
            k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        causal = torch.ones((t, t), device=x.device, dtype=torch.bool).tril()
        allowed = causal[None, None] & valid[:, None, None, :]
        eye = torch.eye(t, device=x.device, dtype=torch.bool)[None, None]
        allowed |= ~valid[:, None, :, None] & eye
        pattern = self.c.attention
        if pattern == "alternating":
            pattern = "local" if self.attention_layer_index % 2 else "dense"
        if self.c.family == "gpt3" and pattern == "local":
            qpos = torch.arange(t, device=x.device)[:, None]
            kpos = torch.arange(t, device=x.device)[None, :]
            allowed &= (qpos - kpos < self.c.window_size)[None, None]
        bias = None
        if self.c.family == "bloom":
            delta = torch.arange(t, device=x.device)[:, None] - torch.arange(t, device=x.device)[None, :]
            bias = -alibi_slopes(self.c.num_heads, device=x.device, dtype=q.dtype)[None, :, None, None] * delta.clamp_min(0)[None, None]
            bias = bias.masked_fill(~allowed, torch.finfo(q.dtype).min)
        if self.c.family == "falcon" and bool(valid.all()) and self.c.attention == "dense":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=(bias if bias is not None else allowed), is_causal=False)
        return self.out(y.transpose(1, 2).reshape(b, t, d)) * valid.unsqueeze(-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, theta: float = 10000.0):
    dim = q.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim))
    angles = position_ids.to(torch.float32)[..., None] * inv_freq
    cos, sin = angles.cos().to(q.dtype)[:, None], angles.sin().to(q.dtype)[:, None]
    def rotate(x):
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)
    return rotate(q), rotate(k)


class DecoderBlock(nn.Module):
    def __init__(self, c, layer_index=0):
        super().__init__()
        self.c = c
        self.attn = SelfAttention(c)
        self.attn.attention_layer_index = layer_index
        eps = 1e-5
        self.norm1 = nn.LayerNorm(c.d_model, eps=eps) if c.family != "falcon" else None
        self.norm2 = nn.LayerNorm(c.d_model, eps=eps) if c.family != "falcon" else None
        hidden = 4 * c.d_model
        activation = nn.ReLU() if c.family == "opt" else nn.GELU()
        self.mlp = nn.Sequential(nn.Linear(c.d_model, hidden, bias=c.family != "falcon"), activation,
                                 nn.Linear(hidden, c.d_model, bias=c.family != "falcon"))
        self.input_norm = nn.LayerNorm(c.d_model, eps=eps) if c.family == "falcon" else None
        self.mlp_norm = nn.LayerNorm(c.d_model, eps=eps) if c.family == "falcon" and not c.falcon_shared_input_norm else None

    def forward(self, x, valid, position_ids):
        if self.c.family == "falcon":
            a = self.attn(self.input_norm(x), valid, position_ids)
            if self.c.falcon_parallel:
                m = self.mlp((self.mlp_norm or self.input_norm)(x))
                x = x + a + m
            else:
                x = x + a
                x = x + self.mlp(self.input_norm(x))
        elif self.c.family == "opt" and self.c.opt_post_layer_norm:
            x = self.norm1(x + self.attn(x, valid))
            x = self.norm2(x + self.mlp(x))
        else:
            x = x + self.attn(self.norm1(x), valid, position_ids)
            x = x + self.mlp(self.norm2(x))
        return x * valid.unsqueeze(-1)


class GPTDecoderFamily(nn.Module):
    architecture_id = "gpt-decoder-family"
    PRESETS = {name: {"family": "gpt3", "preset": name} for name in _PRESETS["gpt3"]}
    PRESETS.update({f"{family}-{name}": {"family": family, "preset": name}
                    for family, presets in _PRESETS.items() for name in presets})

    def __init__(self, vocab_size: int, context_length: int, family: str = "gpt3", *, preset: str | None = None,
                 d_model: int = 768, num_layers: int = 12, num_heads: int = 12, dropout: float = 0.0,
                 pad_token_id: int | None = None, bos_token_id: int | None = None, eos_token_id: int | None = None,
                 attention: str = "dense", window_size: int = 256,
                 falcon_kv_heads: int | None = None, falcon_parallel: bool = True,
                 falcon_shared_input_norm: bool = True, opt_embedding_dim: int | None = None,
                 opt_post_layer_norm: bool | None = None, falcon_rope: bool = True):
        super().__init__()
        if not isinstance(family, str):
            raise TypeError("family must be a string")
        family = family.lower()
        if family not in _PRESETS:
            raise ValueError("family must be gpt3, opt, bloom, or falcon")
        if preset is not None:
            if preset not in _PRESETS[family]:
                raise ValueError(f"unsupported {family} preset: {preset}")
            d_model, num_layers, num_heads = _PRESETS[family][preset]
        if family == "falcon" and falcon_kv_heads is None:
            falcon_kv_heads = 1
        if pad_token_id is None:
            pad_token_id = {"opt": 1, "bloom": 3}.get(family)
        native_control = {"opt": (2, 2), "bloom": (1, 2), "falcon": (11, 11)}.get(family)
        if native_control is not None:
            bos_token_id = native_control[0] if bos_token_id is None and vocab_size > native_control[0] else bos_token_id
            eos_token_id = native_control[1] if eos_token_id is None and vocab_size > native_control[1] else eos_token_id
        if pad_token_id is not None and not 0 <= pad_token_id < vocab_size:
            raise ValueError("pad_token_id must be in vocab range")
        if any(token is not None and not 0 <= token < vocab_size for token in (bos_token_id, eos_token_id)):
            raise ValueError("control token ids must be in vocab range")
        if family == "opt" and opt_post_layer_norm is None:
            opt_post_layer_norm = False
        c = DecoderConfig(family, vocab_size, context_length, d_model, num_layers, num_heads, dropout, pad_token_id,
                          bos_token_id, eos_token_id,
                          attention, window_size, falcon_kv_heads, falcon_parallel, falcon_shared_input_norm,
                          opt_embedding_dim, opt_post_layer_norm, falcon_rope)
        for key in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads"):
            value = getattr(c, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if family == "gpt3" and attention not in {"dense", "local", "alternating"}:
            raise ValueError("GPT-3 attention must be dense, local, or alternating")
        if (isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0
                or family == "falcon" and (isinstance(falcon_kv_heads, bool)
                or not isinstance(falcon_kv_heads, int) or falcon_kv_heads > num_heads
                or num_heads % falcon_kv_heads)):
            raise ValueError("invalid window or Falcon KV head count")
        if family != "gpt3" and attention != "dense":
            raise ValueError("sparse attention variants are supported only for GPT-3 pedagogical configurations")
        if family == "falcon" and not falcon_rope:
            raise ValueError("the supported Falcon variants use RoPE; no unverified ALiBi variant is exposed")
        if family != "opt" and (opt_embedding_dim is not None or opt_post_layer_norm is not None):
            raise ValueError("OPT embedding and normalization options apply only to OPT")
        if family != "falcon" and (falcon_kv_heads is not None or not falcon_parallel
                                    or not falcon_shared_input_norm or not falcon_rope):
            raise ValueError("Falcon attention options apply only to Falcon")
        if family == "falcon" and not falcon_parallel and not falcon_shared_input_norm:
            raise ValueError("sequential Falcon variant supports the shared input norm only")
        self.config = c
        self.family, self.vocab_size, self.context_length, self.d_model, self.pad_token_id = family, vocab_size, context_length, d_model, pad_token_id
        emb = opt_embedding_dim or d_model
        if emb <= 0:
            raise ValueError("opt_embedding_dim must be positive")
        self.token_embedding = nn.Embedding(vocab_size, emb)
        pos_width = d_model if family == "opt" else emb
        self.position_embedding = nn.Embedding(context_length + (2 if family == "opt" else 0), pos_width) if family in {"gpt3", "opt"} else None
        self.project_in = nn.Linear(emb, d_model, bias=False) if emb != d_model else nn.Identity()
        self.embedding_norm = nn.LayerNorm(d_model) if family == "bloom" else None
        self.blocks = nn.ModuleList([DecoderBlock(c, i) for i in range(num_layers)])
        self.final_norm = nn.Identity() if family == "opt" and c.opt_post_layer_norm else nn.LayerNorm(d_model)
        self.project_out = nn.Linear(d_model, emb, bias=False) if emb != d_model else nn.Identity()
        self.lm_head = nn.Linear(emb, vocab_size, bias=False)
        if family in {"gpt3", "opt", "bloom", "falcon"}:
            self.lm_head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init)
        if family == "gpt3":
            residual_std = 0.02 / math.sqrt(2 * num_layers)
            for block in self.blocks:
                nn.init.normal_(block.attn.out.weight, mean=0.0, std=residual_std)
                nn.init.normal_(block.mlp[-1].weight, mean=0.0, std=residual_std)
        if family in {"gpt3", "opt", "bloom", "falcon"}:
            self.lm_head.weight = self.token_embedding.weight

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            if self.family == "opt":
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            else:
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward_features(self, tokens, mask=None, position_ids=None):
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be LongTensor [batch,time]")
        b, t = tokens.shape
        if b <= 0 or not 0 < t <= self.context_length or (tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.vocab_size)):
            raise ValueError("token or sequence range invalid")
        valid = torch.ones_like(tokens, dtype=torch.bool) if mask is None else mask.bool()
        if valid.shape != tokens.shape or valid.device != tokens.device:
            raise ValueError("mask must match tokens")
        if position_ids is None:
            position_ids = valid.long().cumsum(-1) - 1
            if self.family == "opt":
                position_ids = position_ids.masked_fill(~valid, -1)
            else:
                position_ids = position_ids.clamp_min(0)
        if position_ids.shape != tokens.shape or position_ids.dtype != torch.long or position_ids.device != tokens.device:
            raise ValueError("position_ids must match tokens as LongTensor")
        if self.family == "opt":
            position_ids = position_ids + 2
        if position_ids.min() < 0 or position_ids.max() >= (self.position_embedding.num_embeddings if self.position_embedding else self.context_length):
            raise ValueError("position id out of range")
        x = self.project_in(self.token_embedding(tokens))
        if self.position_embedding is not None:
            x = x + self.position_embedding(position_ids)
        if self.embedding_norm is not None: x = self.embedding_norm(x)
        x = self.dropout(x) * valid.unsqueeze(-1)
        for block in self.blocks: x = block(x, valid, position_ids)
        if not (self.family == "opt" and self.config.opt_post_layer_norm): x = self.final_norm(x)
        return self.project_out(x) * valid.unsqueeze(-1)

    def forward(self, tokens, mask=None, position_ids=None):
        return self.lm_head(self.forward_features(tokens, mask, position_ids))

    def get_config(self):
        return dict(vars(self.config))


GPTDecoderLM = GPTDecoderFamily
