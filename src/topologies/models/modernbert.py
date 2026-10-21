import math

import torch
from torch import nn
from torch.nn import functional as F


class ModernBERT(nn.Module):
    architecture_id = "modernbert"
    PRESETS = {
        "tiny": {"d_model": 128, "num_layers": 4, "num_heads": 4},
        "small": {"d_model": 384, "num_layers": 6, "num_heads": 6},
        "base": {"d_model": 768, "num_layers": 22, "num_heads": 12},
    }
    BUDGET_KNOBS = {"d_model": range(32, 385, 32)}

    def __init__(self, vocab_size=50368, context_length=8192, d_model=128,
                 num_layers=4, num_heads=4, intermediate_size=None,
                 local_window=128, global_attn_every_n_layers=3,
                 rope_theta=10000.0, global_rope_theta=160000.0,
                 dropout=0.0, pad_token_id=50283, mask_token_id=50284,
                 preset=None):
        super().__init__()
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError("preset must be tiny, small, or base")
            cfg = self.PRESETS[preset]
            d_model, num_layers, num_heads = cfg["d_model"], cfg["num_layers"], cfg["num_heads"]
        intermediate_size = 3 * d_model // 2 if intermediate_size is None else intermediate_size
        values = (("vocab_size", vocab_size), ("context_length", context_length),
                  ("d_model", d_model), ("num_layers", num_layers),
                  ("num_heads", num_heads), ("intermediate_size", intermediate_size),
                  ("local_window", local_window),
                  ("global_attn_every_n_layers", global_attn_every_n_layers))
        for name, value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % num_heads or (d_model // num_heads) % 2:
            raise ValueError("d_model must divide evenly into an even head dimension")
        if local_window % 2:
            raise ValueError("local_window must be even")
        for name, value in (("rope_theta", rope_theta), ("global_rope_theta", global_rope_theta)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (isinstance(pad_token_id, bool) or isinstance(mask_token_id, bool)
                or not isinstance(pad_token_id, int) or not isinstance(mask_token_id, int)
                or not 0 <= pad_token_id < vocab_size or not 0 <= mask_token_id < vocab_size):
            raise ValueError("special token IDs must be inside vocabulary")
        if pad_token_id == mask_token_id:
            raise ValueError("padding and mask IDs must differ")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        self.vocab_size, self.context_length, self.d_model = vocab_size, context_length, d_model
        self.num_layers, self.num_heads = num_layers, num_heads
        self.local_window, self.global_attn_every_n_layers = local_window, global_attn_every_n_layers
        self.rope_theta, self.global_rope_theta = float(rope_theta), float(global_rope_theta)
        self.pad_token_id, self.mask_token_id = pad_token_id, mask_token_id
        self.token_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.embedding_norm = nn.LayerNorm(d_model, eps=1e-5, bias=False)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([ModernBERTLayer(
            d_model, num_heads, intermediate_size, local_window,
            i % global_attn_every_n_layers == 0,
            global_rope_theta if i % global_attn_every_n_layers == 0 else rope_theta,
            dropout, i == 0) for i in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model, eps=1e-5, bias=False)
        self.mlm_transform = nn.Linear(d_model, d_model, bias=False)
        self.mlm_norm = nn.LayerNorm(d_model, eps=1e-5, bias=False)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))
        self.apply(self._initialize)
        residual_std = 0.02 / math.sqrt(2 * num_layers)
        for layer in self.layers:
            nn.init.trunc_normal_(layer.attn_output.weight, std=residual_std, a=-2 * residual_std, b=2 * residual_std)
            nn.init.trunc_normal_(layer.ffn.output.weight, std=residual_std, a=-2 * residual_std, b=2 * residual_std)
        with torch.no_grad():
            self.token_embeddings.weight[pad_token_id].zero_()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, tokens, mask=None):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be LongTensor [batch, time]")
        batch, length = tokens.shape
        if not batch or not length or length > self.context_length:
            raise ValueError("non-empty tokens must fit context_length")
        if bool((tokens < 0).any()) or bool((tokens >= self.vocab_size).any()):
            raise ValueError("token IDs outside vocabulary")
        if mask is None:
            valid = tokens.ne(self.pad_token_id)
        else:
            if not isinstance(mask, torch.Tensor) or mask.shape != tokens.shape or mask.device != tokens.device:
                raise ValueError("mask must match token shape and device")
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("mask must contain only 0/1; 1 denotes a valid token")
            valid = mask.bool()
        if bool((~valid.any(dim=1)).any()):
            raise ValueError("each row must contain a valid token")
        x = self.embedding_dropout(self.embedding_norm(self.token_embeddings(tokens)))
        x = x * valid.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, valid)
        return self.final_norm(x) * valid.unsqueeze(-1)

    def forward(self, tokens, mask=None):
        hidden = self.encode(tokens, mask)
        valid = tokens.ne(self.pad_token_id) if mask is None else mask.bool()
        logits = F.linear(self.mlm_norm(F.gelu(self.mlm_transform(hidden))),
                          self.token_embeddings.weight, self.mlm_bias)
        return logits * valid.unsqueeze(-1)

    def forward_features(self, tokens, mask=None):
        return self.encode(tokens, mask)


class GeGLU(nn.Module):
    def __init__(self, d_model, intermediate_size):
        super().__init__()
        self.gate_value = nn.Linear(d_model, 2 * intermediate_size, bias=False)
        self.output = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        value, gate = self.gate_value(x).chunk(2, dim=-1)
        return self.output(F.gelu(value) * gate)


def apply_rope(x, theta):
    length, dim = x.shape[-2], x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    angles = torch.arange(length, device=x.device, dtype=torch.float32)[:, None] * inv[None, :]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    cos = torch.cat((cos, cos), dim=-1)[None, None]
    sin = torch.cat((sin, sin), dim=-1)[None, None]
    half = dim // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


class ModernBERTLayer(nn.Module):
    def __init__(self, d_model, num_heads, intermediate_size, local_window,
                 global_attention, rope_theta, dropout, first_layer=False):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, d_model // num_heads
        self.local_window, self.global_attention, self.rope_theta = local_window, global_attention, rope_theta
        self.attn_norm = nn.Identity() if first_layer else nn.LayerNorm(d_model, eps=1e-5, bias=False)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_output = nn.Linear(d_model, d_model, bias=False)
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-5, bias=False)
        self.ffn = GeGLU(d_model, intermediate_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(batch, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_rope(q, self.rope_theta), apply_rope(k, self.rope_theta)
        if self.global_attention:
            allowed = valid[:, None, None, :]
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        else:
            radius = self.local_window // 2
            padded_k = F.pad(k, (0, 0, radius, radius))
            padded_v = F.pad(v, (0, 0, radius, radius))
            key_windows = padded_k.unfold(2, self.local_window + 1, 1).permute(0, 1, 2, 4, 3)
            value_windows = padded_v.unfold(2, self.local_window + 1, 1).permute(0, 1, 2, 4, 3)
            padded_valid = F.pad(valid, (radius, radius), value=False)
            valid_windows = padded_valid.unfold(1, self.local_window + 1, 1)[:, None, :, :]
            scores = (q.unsqueeze(-2) * key_windows).sum(-1) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~valid_windows, torch.finfo(scores.dtype).min)
            weights = scores.softmax(dim=-1)
            attended = (weights.unsqueeze(-1) * value_windows).sum(-2)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + self.dropout(self.attn_output(attended))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x * valid.unsqueeze(-1)
