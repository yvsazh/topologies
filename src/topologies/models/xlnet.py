from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _positive(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _binary(name, value, shape, device):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if value.device != device or not ((value == 0) | (value == 1)).all():
        raise ValueError(f"{name} must contain only 0/1 on the tokens device")
    return value.bool()


class RelativeAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, clamp_len=0):
        super().__init__()
        self.d_model, self.num_heads = d_model, num_heads
        self.head_dim = d_model // num_heads
        self.clamp_len = clamp_len
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.position = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.content_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.position_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.segment_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.segment_embeddings = nn.Parameter(torch.empty(2, num_heads, self.head_dim))
        nn.init.normal_(self.segment_embeddings, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def split(self, x):
        return x.reshape(*x.shape[:-1], self.num_heads, self.head_dim)

    def scores(self, query, context, positions, query_segments, key_segments):
        q, k = self.split(self.query(query)), self.split(self.key(context))
        content = torch.einsum("bihd,bjhd->bhij", q + self.content_bias, k)
        key_positions = torch.arange(context.shape[1], device=query.device)
        offsets = positions[:, :, None] - key_positions[None, None, :]
        low, high = int(offsets.min()), int(offsets.max())
        distances = torch.arange(low, high + 1, device=query.device, dtype=torch.float32)
        if self.clamp_len:
            distances = distances.clamp(-self.clamp_len, self.clamp_len)
        frequencies = torch.exp(-math.log(10000.0) *
            torch.arange(0, self.d_model, 2, device=query.device, dtype=torch.float32) / self.d_model)
        angles = distances[:, None] * frequencies[None, :]
        relative = self.split(self.position(torch.cat((angles.sin(), angles.cos()), -1).to(query.dtype)))
        positional = torch.einsum("bihd,rhd->bhir", q + self.position_bias, relative)
        positional = positional.gather(-1, (offsets - low)[:, None].expand(-1, self.num_heads, -1, -1))
        choices = torch.einsum("bihd,shd->bhis", q + self.segment_bias, self.segment_embeddings)
        different = (query_segments[:, :, None] != key_segments[:, None, :]).long()
        segment = choices.gather(-1, different[:, None].expand(-1, self.num_heads, -1, -1))
        return (content + positional + segment) / math.sqrt(self.head_dim)

    def forward(self, query, context, positions, query_segments, key_segments, allowed):
        scores = self.scores(query, context, positions, query_segments, key_segments)
        allowed = allowed[:, None]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        normalized = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
        weights = torch.softmax(normalized, -1).to(scores.dtype) * allowed
        mixed = torch.einsum("bhij,bjhd->bihd", self.dropout(weights), self.split(self.value(context)))
        return self.output(mixed.reshape(*query.shape)), weights


class XLNetBlock(nn.Module):
    def __init__(self, d_model, num_heads, intermediate_size, dropout, layer_norm_eps, clamp_len):
        super().__init__()
        self.attention = RelativeAttention(d_model, num_heads, dropout, clamp_len)
        self.attention_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, intermediate_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(intermediate_size, d_model))
        self.output_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stream, context, positions, segments, key_segments, allowed, valid):
        update, weights = self.attention(stream, context, positions, segments, key_segments, allowed)
        hidden = self.attention_norm(stream + self.dropout(update))
        hidden = self.output_norm(hidden + self.dropout(self.feed_forward(hidden)))
        return hidden * valid.unsqueeze(-1), weights


class XLNet(nn.Module):
    architecture_id = "xlnet"
    PRESETS = {
        "tiny": {"d_model": 64, "num_layers": 2, "num_heads": 4},
        "small": {"d_model": 256, "num_layers": 6, "num_heads": 8},
        "base": {"d_model": 768, "num_layers": 12, "num_heads": 12},
    }
    BUDGET_KNOBS = {"d_model": range(16, 513, 8)}

    def __init__(self, vocab_size=32000, context_length=512, d_model=64,
                 num_layers=2, num_heads=4, intermediate_size=None, dropout=0.0,
                 mem_len=128, attention_type="bi", pad_token_id=5,
                 clamp_len=0, initializer_range=0.02, layer_norm_eps=1e-12):
        super().__init__()
        for name, value in (("vocab_size", vocab_size), ("context_length", context_length),
                            ("d_model", d_model), ("num_layers", num_layers), ("num_heads", num_heads)):
            _positive(name, value)
        if d_model % 2 or d_model % num_heads:
            raise ValueError("d_model must be even and divisible by num_heads")
        intermediate_size = 4 * d_model if intermediate_size is None else intermediate_size
        _positive("intermediate_size", intermediate_size)
        for name, value in (("mem_len", mem_len), ("clamp_len", clamp_len)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if attention_type not in ("bi", "uni"):
            raise ValueError("attention_type must be 'bi' (bidirectional) or 'uni' (causal)")
        if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for name, value in (("initializer_range", initializer_range), ("layer_norm_eps", layer_norm_eps)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if pad_token_id is not None and (not isinstance(pad_token_id, int) or
                isinstance(pad_token_id, bool) or not 0 <= pad_token_id < vocab_size):
            raise ValueError("pad_token_id must be None or an integer inside vocab_size")
        self.vocab_size, self.context_length = vocab_size, context_length
        self.d_model, self.num_layers, self.num_heads = d_model, num_layers, num_heads
        self.mem_len, self.attention_type, self.pad_token_id = mem_len, attention_type, pad_token_id
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        self.query_embedding = nn.Parameter(torch.empty(1, 1, d_model))
        self.layers = nn.ModuleList([XLNetBlock(d_model, num_heads, intermediate_size,
            dropout, layer_norm_eps, clamp_len) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.lm_bias = nn.Parameter(torch.zeros(vocab_size))
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=initializer_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, RelativeAttention):
                for parameter in (module.content_bias, module.position_bias, module.segment_bias,
                                  module.segment_embeddings):
                    nn.init.normal_(parameter, std=initializer_range)
        nn.init.normal_(self.query_embedding, std=initializer_range)

    def _segments(self, value, shape, device, name):
        if value is None:
            return torch.zeros(shape, device=device, dtype=torch.long)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {tuple(shape)}")
        if value.dtype != torch.long or value.device != device or (value < 0).any():
            raise ValueError(f"{name} must be nonnegative LongTensor on the tokens device")
        return value

    def _memory(self, memory, batch, device, dtype):
        if memory is None:
            return (tuple(torch.empty(batch, 0, self.d_model, device=device, dtype=dtype)
                          for _ in self.layers), torch.empty(batch, 0, device=device, dtype=torch.bool),
                    torch.empty(batch, 0, device=device, dtype=torch.long))
        if not isinstance(memory, dict) or set(memory) != {"hidden_states", "mask", "token_type_ids"}:
            raise ValueError("memory requires hidden_states, mask and token_type_ids; reset at document boundaries")
        states = memory["hidden_states"]
        if not isinstance(states, (tuple, list)) or len(states) != self.num_layers:
            raise ValueError("memory hidden_states must contain one pre-layer state per layer")
        length = states[0].shape[1] if isinstance(states[0], torch.Tensor) and states[0].ndim == 3 else -1
        if not 0 <= length <= self.mem_len:
            raise ValueError("memory length must be between 0 and mem_len")
        for state in states:
            if (not isinstance(state, torch.Tensor) or state.shape != (batch, length, self.d_model)
                    or state.device != device or state.dtype != dtype or not torch.isfinite(state).all()):
                raise ValueError("memory states must be finite [B, M, d_model] with matching device/dtype")
        valid = _binary("memory mask", memory["mask"], (batch, length), device)
        segments = self._segments(memory["token_type_ids"], (batch, length), device, "memory token_type_ids")
        return tuple(state.detach() for state in states), valid, segments

    def forward(self, tokens, mask=None, *, perm_mask=None, target_mapping=None,
                token_type_ids=None, memory=None, query_stream=False, return_dict=False,
                output_attentions=False):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens must be a LongTensor[B, T]")
        batch, length = tokens.shape
        if batch == 0 or not 0 < length <= self.context_length:
            raise ValueError("tokens require B > 0 and 0 < T <= context_length")
        if (tokens < 0).any() or (tokens >= self.vocab_size).any():
            raise ValueError("tokens must contain IDs inside vocab_size")
        valid = (tokens != self.pad_token_id if self.pad_token_id is not None else torch.ones_like(tokens, dtype=torch.bool))
        if mask is not None:
            valid = valid & _binary("mask", mask, tokens.shape, tokens.device)
        segments = self._segments(token_type_ids, tokens.shape, tokens.device, "token_type_ids")
        use_query = query_stream or target_mapping is not None or perm_mask is not None
        indices = torch.arange(length, device=tokens.device).expand(batch, -1)
        if target_mapping is not None:
            if not isinstance(target_mapping, torch.Tensor) or target_mapping.ndim != 3:
                raise ValueError("target_mapping must be one-hot [B, N, T]")
            if target_mapping.shape[0] != batch or target_mapping.shape[2] != length or target_mapping.shape[1] == 0:
                raise ValueError("target_mapping must be nonempty [B, N, T]")
            mapping = _binary("target_mapping", target_mapping, target_mapping.shape, tokens.device)
            if not (mapping.sum(-1) == 1).all():
                raise ValueError("target_mapping rows must each select exactly one position")
            indices = mapping.long().argmax(-1)
        diagonal = torch.eye(length, device=tokens.device, dtype=torch.bool)[None]
        natural = torch.arange(length, device=tokens.device)
        causal = natural[None, :] <= natural[:, None]
        if perm_mask is not None:
            blocked = _binary("perm_mask", perm_mask, (batch, length, length), tokens.device)
            preceding = ~blocked & ~diagonal
            if self.attention_type == "uni":
                preceding = preceding & causal
            paths = torch.bmm(preceding.float(), preceding.float()) > 0
            if (paths & ~preceding).any():
                raise ValueError("perm_mask must define a transitive acyclic factorization order; otherwise queries leak target content")
        elif use_query:
            preceding = (causal & ~diagonal).expand(batch, -1, -1)
        else:
            preceding = (causal if self.attention_type == "uni" else torch.ones_like(causal)).expand(batch, -1, -1)
        content_allowed = (preceding | diagonal) & valid[:, None, :]
        query_allowed = preceding.gather(1, indices[:, :, None].expand(-1, -1, length)) & valid[:, None, :]
        content = self.dropout(self.word_embeddings(tokens)) * valid.unsqueeze(-1)
        old, memory_valid, memory_segments = self._memory(memory, batch, tokens.device, content.dtype)
        mem_length = memory_valid.shape[1]
        key_segments = torch.cat((memory_segments, segments), 1)
        content_allowed = torch.cat((memory_valid[:, None].expand(-1, length, -1), content_allowed), -1)
        query_allowed = torch.cat((memory_valid[:, None].expand(-1, indices.shape[1], -1), query_allowed), -1)
        query_valid = valid.gather(1, indices)
        query_segments = segments.gather(1, indices)
        query = self.dropout(self.query_embedding.expand(batch, indices.shape[1], -1)) * query_valid.unsqueeze(-1) if use_query else None
        cached, attentions = [], []
        positions = natural.expand(batch, -1) + mem_length
        for layer, previous in zip(self.layers, old):
            cached.append(torch.cat((previous, content), 1)[:, -self.mem_len:].detach().clone()
                          if self.mem_len else content[:, :0].detach().clone())
            context = torch.cat((previous, content), 1)
            next_content, h_weights = layer(content, context, positions, segments,
                key_segments, content_allowed, valid)
            g_weights = None
            if query is not None:
                query, g_weights = layer(query, context, indices + mem_length, query_segments,
                    key_segments, query_allowed, query_valid)
            content = next_content
            if output_attentions:
                attentions.append({"content": h_weights, "query": g_weights})
        hidden = self.dropout(query if use_query else content)
        logits = F.linear(hidden, self.word_embeddings.weight, self.lm_bias)
        logits = logits * (query_valid if use_query else valid).unsqueeze(-1)
        if return_dict:
            combined_mask = torch.cat((memory_valid, valid), 1)
            combined_segments = key_segments
            return {"logits": logits, "hidden_states": hidden, "content_states": content,
                    "memory": {"hidden_states": tuple(cached),
                        "mask": combined_mask[:, -self.mem_len:].detach().clone() if self.mem_len else combined_mask[:, :0],
                        "token_type_ids": combined_segments[:, -self.mem_len:].detach().clone() if self.mem_len else combined_segments[:, :0]},
                    "attentions": tuple(attentions)}
        return logits

    def forward_pretraining(self, tokens, mask=None, **kwargs):
        return self(tokens, mask, query_stream=True, return_dict=True, **kwargs)

    def encode(self, tokens, mask=None, **kwargs):
        return self(tokens, mask, return_dict=True, **kwargs)["hidden_states"]
