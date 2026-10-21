from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerXL(nn.Module):
    architecture_id = "transformer-xl"
    PRESETS = {"tiny": {"d_model": 32, "num_layers": 2, "num_heads": 4, "d_inner": 64},
               "small": {"d_model": 128, "num_layers": 4, "num_heads": 4, "d_inner": 256},
               "base": {"d_model": 256, "num_layers": 6, "num_heads": 8, "d_inner": 512}}
    BUDGET_KNOBS = {"d_model": range(16, 513, 8)}

    def __init__(self, vocab_size: int, context_length: int = 128, d_model: int = 64,
                 num_layers: int = 2, num_heads: int = 4, d_inner: int | None = None,
                 mem_len: int = 128, dropout: float = 0.0, ext_len: int = 0,
                 pad_token_id: int | None = None):
        super().__init__()
        if vocab_size < 2 or context_length < 1 or d_model < 1 or num_layers < 1 or num_heads < 1:
            raise ValueError("vocab_size, context_length, d_model, num_layers and num_heads must be positive (vocab_size >= 2)")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if d_model % 2:
            raise ValueError("d_model must be even for sinusoidal positional embeddings")
        if mem_len < 0 or ext_len < 0 or not 0 <= dropout < 1:
            raise ValueError("mem_len/ext_len must be nonnegative and dropout in [0, 1)")
        self.vocab_size, self.context_length, self.d_model = vocab_size, context_length, d_model
        self.num_layers, self.num_heads, self.mem_len, self.ext_len = num_layers, num_heads, mem_len, ext_len
        self.pad_token_id = pad_token_id
        self.head_dim = d_model // num_heads
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.layers = nn.ModuleList([_Layer(d_model, num_heads, d_inner or 4 * d_model, dropout) for _ in range(num_layers)])
        self.r_w_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.r_r_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.drop = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits, _, _ = self.forward_with_mems(tokens, None, mask=mask)
        return logits

    def forward_with_mems(self, tokens: torch.Tensor, mems: tuple[torch.Tensor, ...] | None = None,
                          mask: torch.Tensor | None = None, memory_mask: torch.Tensor | None = None):
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be LongTensor[B, T]")
        batch, length = tokens.shape
        if length > self.context_length:
            raise ValueError("segment length exceeds context_length")
        if mems is not None:
            if not isinstance(mems, (tuple, list)) or len(mems) != self.num_layers:
                raise ValueError("mems must contain one [M, B, d_model] tensor per layer")
            lengths = set()
            for memory in mems:
                if not isinstance(memory, torch.Tensor) or memory.ndim != 3 or memory.shape[1:] != (batch, self.d_model):
                    raise ValueError("each memory must have shape [M, B, d_model]")
                if memory.device != tokens.device or memory.dtype != self.word_emb.weight.dtype:
                    raise ValueError("memories must match the token device and model dtype")
                lengths.add(memory.shape[0])
            if len(lengths) != 1:
                raise ValueError("all layer memories must have the same length")
        valid = torch.ones_like(tokens, dtype=torch.bool) if mask is None else mask.to(device=tokens.device, dtype=torch.bool)
        if valid.shape != tokens.shape:
            raise ValueError("mask must have shape [B, T]")
        if self.pad_token_id is not None:
            valid = valid & tokens.ne(self.pad_token_id)
        if memory_mask is None:
            if mems is not None and mems[0].shape[0]:
                memory_mask = torch.ones(batch, mems[0].shape[0], dtype=torch.bool, device=tokens.device)
            else:
                memory_mask = torch.zeros(batch, 0, dtype=torch.bool, device=tokens.device)
        else:
            memory_mask = memory_mask.to(device=tokens.device, dtype=torch.bool)
            if memory_mask.ndim != 2 or memory_mask.shape[0] != batch or memory_mask.shape[1] != (mems[0].shape[0] if mems is not None else 0):
                raise ValueError("memory_mask must match the memory [B, M] axes")
        x = self.drop(self.word_emb(tokens)).masked_fill(~valid.unsqueeze(-1), 0)
        new_mems = []
        next_mask = torch.cat((memory_mask, valid), dim=1)[:, -self.mem_len:] if self.mem_len else valid[:, :0]
        for index, layer in enumerate(self.layers):
            old = None if mems is None else mems[index].detach()
            x, cached = layer(x, old, valid, memory_mask, self.r_w_bias, self.r_r_bias)
            new_mems.append(cached[-self.mem_len:].detach() if self.mem_len else cached[:0].detach())
        hidden = self.final_norm(x)
        logits = F.linear(hidden, self.word_emb.weight, self.out_bias)
        logits = logits.masked_fill(~valid.unsqueeze(-1), 0)
        return logits, tuple(new_mems), next_mask.detach()


class _Layer(nn.Module):
    def __init__(self, width, heads, inner, dropout):
        super().__init__()
        self.width, self.heads, self.head_dim = width, heads, width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.r = nn.Linear(width, width, bias=False)
        self.o = nn.Linear(width, width)
        self.ff1, self.ff2 = nn.Linear(width, inner), nn.Linear(inner, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, valid, memory_mask, r_w_bias, r_r_bias):
        b, t, d = x.shape
        prior = 0 if mem is None else mem.shape[0]
        cat = x.transpose(0, 1) if mem is None else torch.cat((mem, x.transpose(0, 1)), dim=0)
        z = self.norm1(x)
        q = self.qkv(z).view(b, t, 3, self.heads, self.head_dim)[:, :, 0].permute(0, 2, 1, 3)
        kv = self.qkv(self.norm1(cat.transpose(0, 1))).view(b, prior + t, 3, self.heads, self.head_dim)
        k, v = kv[:, :, 1].permute(0, 2, 1, 3), kv[:, :, 2].permute(0, 2, 1, 3)
        distances = (torch.arange(t, device=x.device)[:, None] + prior - torch.arange(prior + t, device=x.device)[None, :]).transpose(0, 1)
        inv = torch.exp(torch.arange(0, self.width, 2, device=x.device, dtype=torch.float32) * (-math.log(10000.0) / self.width))
        ang = distances.float().unsqueeze(-1) * inv
        r = torch.cat((ang.sin(), ang.cos()), dim=-1)
        r = self.r(r.to(x.dtype)).view(prior + t, t, self.heads, self.head_dim).permute(2, 1, 0, 3)
        ac = torch.einsum("bhtd,bhkd->bhtk", q + r_w_bias[None, :, None], k)
        bd = torch.einsum("bhtd,htkd->bhtk", q + r_r_bias[None, :, None], r)
        score = (ac + bd) / math.sqrt(self.head_dim)
        query_index = torch.arange(t, device=x.device)[:, None] + prior
        key_index = torch.arange(prior + t, device=x.device)[None, :]
        blocked = key_index > query_index
        key_valid = torch.cat((memory_mask, valid), 1)
        blocked = blocked[None, None] | ~key_valid[:, None, None, :]
        weights = torch.softmax(score.masked_fill(blocked, torch.finfo(score.dtype).min), -1)
        attended = torch.einsum("bhtk,bhkd->bhtd", weights, v).transpose(1, 2).reshape(b, t, d)
        cached = cat
        x = x + self.drop(self.o(attended))
        x = x + self.drop(self.ff2(F.gelu(self.ff1(self.norm2(x)))))
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        return x, cached
