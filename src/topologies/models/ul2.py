from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def relative_position_bucket(relative_position, num_buckets=32, max_distance=128, bidirectional=True):
    buckets = 0
    if bidirectional:
        num_buckets //= 2
        buckets = (relative_position > 0).long() * num_buckets
        distance = relative_position.abs()
    else:
        distance = (-relative_position).clamp_min(0)
    exact = num_buckets // 2
    large = exact + (torch.log(distance.float().clamp_min(exact) / exact) /
                     math.log(max_distance / exact) * (num_buckets - exact)).long()
    return buckets + torch.where(distance < exact, distance, large.clamp_max(num_buckets - 1))


class _Norm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class _Attention(nn.Module):
    def __init__(self, dim, heads, d_kv, dropout=0.0):
        super().__init__()
        self.heads, self.d_kv, self.dropout = heads, d_kv, dropout
        self.q = nn.Linear(dim, heads * d_kv, bias=False)
        self.k = nn.Linear(dim, heads * d_kv, bias=False)
        self.v = nn.Linear(dim, heads * d_kv, bias=False)
        self.o = nn.Linear(heads * d_kv, dim, bias=False)

    def forward(self, qx, kx, key_mask, causal=False, bias=None, bidirectional=True, visibility=None):
        b, qlen, _ = qx.shape
        klen = kx.shape[1]
        split = lambda x: x.view(b, -1, self.heads, self.d_kv).transpose(1, 2)
        q, k, v = split(self.q(qx)), split(self.k(kx)), split(self.v(kx))
        scores = q @ k.transpose(-2, -1)
        qi = torch.arange(qlen, device=qx.device)[:, None]
        ki = torch.arange(klen, device=qx.device)[None, :]
        if bias is not None:
            buckets = relative_position_bucket(ki - qi, bias.num_embeddings, bidirectional=bidirectional)
            scores = scores + bias(buckets).permute(2, 0, 1).unsqueeze(0)
        valid = key_mask[:, None, None, :]
        if visibility is not None: valid = valid & visibility[None, None]
        if causal: valid = valid & (ki <= qi)[None, None]
        probs = torch.softmax(scores.masked_fill(~valid, torch.finfo(scores.dtype).min), -1)
        probs = F.dropout(probs, self.dropout, self.training)
        return self.o((probs @ v).transpose(1, 2).contiguous().view(b, qlen, -1))


class _FF(nn.Module):
    def __init__(self, dim, hidden, gated=False, dropout=0.0):
        super().__init__()
        self.gated, self.dropout = gated, dropout
        self.wi = nn.Linear(dim, hidden, bias=False)
        self.wi_gate = nn.Linear(dim, hidden, bias=False) if gated else None
        self.wo = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        y = self.wi(x)
        y = F.gelu(self.wi_gate(x), approximate="tanh") * y if self.gated else F.relu(y)
        return self.wo(F.dropout(y, self.dropout, self.training))


class _MoE(nn.Module):
    def __init__(self, dim, hidden, experts, route, capacity_factor=2.0, dropout=0.0):
        super().__init__()
        self.experts = nn.ModuleList([_FF(dim, hidden, False, dropout) for _ in range(experts)])
        self.router = nn.Linear(dim, experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.expert_count, self.route, self.capacity_factor = experts, route, capacity_factor
        self.aux_loss = torch.tensor(0.0)
        self.router_probs = None
        self.token_fraction = None
        self.capacity = None

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        probabilities = self.router(flat).softmax(-1)
        self.router_probs = probabilities
        out = torch.zeros_like(flat)
        if self.route == "expert-choice":
            self.capacity = max(1, int(self.capacity_factor * flat.shape[0] / self.expert_count))
            assignments = torch.zeros_like(probabilities)
            for index in range(self.expert_count):
                weights, tokens = probabilities[:, index].topk(min(self.capacity, flat.shape[0]))
                assignments[tokens, index] = 1
                out.index_add_(0, tokens, self.experts[index](flat[tokens]) * weights[:, None])
            self.token_fraction = assignments.mean(0)
            self.aux_loss = self.expert_count ** 2 * (self.token_fraction * probabilities.mean(0)).sum()
        else:
            weights, indices = probabilities.topk(2, dim=-1)
            weights = weights / weights.sum(-1, keepdim=True)
            assignments = F.one_hot(indices, self.expert_count).float().sum(1)
            for index in range(self.expert_count):
                rows, slots = torch.where(indices == index)
                if rows.numel(): out.index_add_(0, rows, self.experts[index](flat[rows]) * weights[rows, slots, None])
            self.token_fraction = assignments.mean(0)
            self.aux_loss = self.expert_count ** 2 * (self.token_fraction * probabilities.mean(0)).sum()
        return out.reshape(shape)


class UL2(nn.Module):
    architecture_id = "ul2"
    MODE_TOKENS = {"R": "<extra_id_0>", "S": "<extra_id_1>", "X": "<extra_id_2>"}
    PRESETS = {
        "tiny": dict(vocab_size=128, d_model=32, d_ff=64, num_heads=4, d_kv=8, num_layers=2, num_experts=4),
        "xl-moe": dict(vocab_size=32128, context_length=512, d_model=2048, d_ff=5120, num_heads=32, d_kv=64, num_layers=24, num_experts=32, dropout=0.0),
        "longt5-base": dict(vocab_size=32128, context_length=4096, d_model=768, d_ff=2048, num_heads=12, d_kv=64, num_layers=12, num_experts=4, local_radius=127, global_block_size=16),
    }

    def __init__(self, vocab_size=128, context_length=128, d_model=32, d_ff=64, num_heads=4, d_kv=None,
                 num_layers=2, num_experts=4, dropout=0.0, local_radius=None, global_block_size=16,
                 relative_attention_num_buckets=32, relative_attention_max_distance=128,
                 layer_norm_epsilon=1e-6, gated_ff=False, pad_token_id=0, tie_word_embeddings=True, **kwargs):
        super().__init__()
        d_kv = d_model // num_heads if d_kv is None else d_kv
        if min(vocab_size, context_length, d_model, d_ff, num_heads, d_kv, num_layers, num_experts) <= 0 or not 0 <= dropout < 1:
            raise ValueError("invalid model geometry or dropout")
        if d_model % num_heads or relative_attention_num_buckets < 4 or relative_attention_num_buckets % 2:
            raise ValueError("invalid attention geometry")
        self.vocab_size, self.context_length, self.d_model, self.pad_token_id = vocab_size, context_length, d_model, pad_token_id
        self.local_radius, self.global_block_size = local_radius, global_block_size
        self.num_experts, self.dropout_rate, self.tie_word_embeddings = num_experts, dropout, tie_word_embeddings
        self.shared = nn.Embedding(vocab_size, d_model)
        self.encoder_bias = nn.Embedding(relative_attention_num_buckets, num_heads)
        self.decoder_bias = nn.Embedding(relative_attention_num_buckets, num_heads)
        self.encoder_attn = nn.ModuleList([_Attention(d_model, num_heads, d_kv, dropout) for _ in range(num_layers)])
        self.decoder_attn = nn.ModuleList([_Attention(d_model, num_heads, d_kv, dropout) for _ in range(num_layers)])
        self.cross_attn = nn.ModuleList([_Attention(d_model, num_heads, d_kv, dropout) for _ in range(num_layers)])
        self.encoder_norms = nn.ModuleList([_Norm(d_model, layer_norm_epsilon) for _ in range(2 * num_layers)])
        self.decoder_norms = nn.ModuleList([_Norm(d_model, layer_norm_epsilon) for _ in range(3 * num_layers)])
        self.encoder_ff = nn.ModuleList([_MoE(d_model, d_ff, num_experts, "expert-choice", 2.0, dropout) if i % 2 == 1 else _FF(d_model, d_ff, gated_ff, dropout) for i in range(num_layers)])
        self.decoder_ff = nn.ModuleList([_MoE(d_model, d_ff, num_experts, "top2", 2.0, dropout) if i % 2 == 1 else _FF(d_model, d_ff, gated_ff, dropout) for i in range(num_layers)])
        self.encoder_final_norm, self.decoder_final_norm = _Norm(d_model, layer_norm_epsilon), _Norm(d_model, layer_norm_epsilon)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.shared.weight, std=1.0)
        for emb in (self.encoder_bias, self.decoder_bias): nn.init.normal_(emb.weight, std=1.0)
        for stack in (self.encoder_attn, self.decoder_attn, self.cross_attn):
            for attn in stack:
                for name, layer in (("q", attn.q), ("k", attn.k), ("v", attn.v), ("o", attn.o)):
                    denom = (self.d_model * attn.d_kv) ** .5 if name == "q" else self.d_model ** .5 if name in ("k", "v") else (attn.heads * attn.d_kv) ** .5
                    nn.init.normal_(layer.weight, std=1 / denom)
        if self.tie_word_embeddings: self.lm_head.weight = self.shared.weight

    def attention_visibility(self, length, device=None, global_tokens=True):
        pos = torch.arange(length, device=device)
        if self.local_radius is None: return torch.ones(length, length, dtype=torch.bool, device=device)
        local = (pos[:, None] - pos[None, :]).abs() <= self.local_radius
        if not global_tokens: return local
        globals_ = torch.arange(0, length, self.global_block_size, device=device)
        visibility = local.clone()
        for start in globals_:
            end = min(int(start) + self.global_block_size, length)
            visibility[start, start:end] = True
            visibility[:, start] = True
        return visibility

    def encode(self, src, mode="R", src_mask=None, global_tokens=True):
        if mode not in self.MODE_TOKENS: raise ValueError("mode must be R, S, or X")
        mask = src.ne(self.pad_token_id) if src_mask is None else src_mask.bool()
        x = F.dropout(self.shared(src), self.dropout_rate, self.training) * mask.unsqueeze(-1)
        visibility = self.attention_visibility(src.shape[1], src.device, global_tokens)
        for i, (attn, ff) in enumerate(zip(self.encoder_attn, self.encoder_ff)):
            normalized = self.encoder_norms[2*i](x)
            x = x + F.dropout(attn(normalized, normalized, mask, bias=self.encoder_bias, visibility=visibility), self.dropout_rate, self.training)
            x = x + F.dropout(ff(self.encoder_norms[2*i+1](x)), self.dropout_rate, self.training)
            x = x * mask.unsqueeze(-1)
        return self.encoder_final_norm(x) * mask.unsqueeze(-1), mask

    def forward(self, src, tgt, mode="R", src_mask=None, tgt_mask=None, global_tokens=True):
        memory, memory_mask = self.encode(src, mode, src_mask, global_tokens)
        x = F.dropout(self.shared(tgt), self.dropout_rate, self.training)
        mask = tgt.ne(self.pad_token_id) if tgt_mask is None else tgt_mask.bool()
        mask = mask.clone()
        mask[:, 0] = True
        for i, (self_attn, cross_attn, ff) in enumerate(zip(self.decoder_attn, self.cross_attn, self.decoder_ff)):
            normalized = self.decoder_norms[3*i](x)
            x = x + F.dropout(self_attn(normalized, normalized, mask, causal=True, bias=self.decoder_bias, bidirectional=False), self.dropout_rate, self.training)
            normalized = self.decoder_norms[3*i+1](x)
            x = x + F.dropout(cross_attn(normalized, memory, memory_mask), self.dropout_rate, self.training)
            x = x + F.dropout(ff(self.decoder_norms[3*i+2](x)), self.dropout_rate, self.training)
        x = self.decoder_final_norm(x)
        if self.tie_word_embeddings: x = x * self.d_model ** -.5
        return F.linear(x, self.lm_head.weight)

    def auxiliary_loss(self):
        losses = [layer.aux_loss for layer in self.decoder_ff if isinstance(layer, _MoE)]
        return 0.01 * sum(losses, torch.tensor(0.0, device=self.shared.weight.device))
