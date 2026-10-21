from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, query, key, key_mask, causal=False):
        b, qlen, dim = query.shape
        klen, d = key.shape[1], dim // self.heads
        split = lambda x: x.view(b, -1, self.heads, d).transpose(1, 2)
        q, k, v = split(self.q_proj(query)), split(self.k_proj(key)), split(self.v_proj(key))
        scores = q @ k.transpose(-2, -1) / math.sqrt(d)
        allowed = key_mask[:, None, None, :]
        if causal:
            qi = torch.arange(qlen, device=query.device)[:, None]
            ki = torch.arange(klen, device=query.device)[None, :]
            allowed = allowed & (ki <= qi)[None, None]
        weights = torch.softmax(scores.masked_fill(~allowed, torch.finfo(scores.dtype).min), -1)
        result = F.dropout(weights, self.dropout, self.training) @ v
        return self.out_proj(result.transpose(1, 2).contiguous().view(b, qlen, dim))


class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = Attention(dim, heads, dropout)
        self.fc1, self.fc2 = nn.Linear(dim, ff_dim), nn.Linear(ff_dim, dim)
        self.self_attn_layer_norm, self.final_layer_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, x, mask):
        x = self.self_attn_layer_norm(x + F.dropout(self.self_attn(x, x, mask), self.dropout, self.training))
        y = self.fc2(F.gelu(self.fc1(x)))
        return self.final_layer_norm(x + F.dropout(y, self.dropout, self.training)) * mask.unsqueeze(-1)


class DecoderLayer(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout):
        super().__init__()
        self.self_attn, self.encoder_attn = Attention(dim, heads, dropout), Attention(dim, heads, dropout)
        self.fc1, self.fc2 = nn.Linear(dim, ff_dim), nn.Linear(ff_dim, dim)
        self.self_attn_layer_norm, self.encoder_attn_layer_norm, self.final_layer_norm = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, x, memory, mask, memory_mask):
        x = self.self_attn_layer_norm(x + F.dropout(self.self_attn(x, x, mask, True), self.dropout, self.training))
        x = self.encoder_attn_layer_norm(x + F.dropout(self.encoder_attn(x, memory, memory_mask), self.dropout, self.training))
        y = self.fc2(F.gelu(self.fc1(x)))
        return self.final_layer_norm(x + F.dropout(y, self.dropout, self.training)) * mask.unsqueeze(-1)


class BART(nn.Module):
    architecture_id = "bart"
    PRESETS = {
        "tiny": dict(vocab_size=128, d_model=32, num_heads=4, d_ff=64, num_layers=1, num_decoder_layers=1, context_length=64),
        "base": dict(context_length=1024, d_model=768, num_heads=12, d_ff=3072, num_layers=6, num_decoder_layers=6),
        "large": dict(context_length=1024, d_model=1024, num_heads=16, d_ff=4096, num_layers=12, num_decoder_layers=12),
        "mbart-base": dict(context_length=1024, d_model=768, num_heads=12, d_ff=3072, num_layers=6, num_decoder_layers=6, scale_embedding=True),
        "mbart-large": dict(context_length=1024, d_model=1024, num_heads=16, d_ff=4096, num_layers=12, num_decoder_layers=12, scale_embedding=True),
    }

    def __init__(self, vocab_size=128, context_length=128, d_model=32, num_heads=4, d_ff=64, num_layers=1,
                 num_decoder_layers=1, dropout=0.0, pad_token_id=1, eos_token_id=2, bos_token_id=0,
                 decoder_start_token_id=None, tie_word_embeddings=True, scale_embedding=False):
        super().__init__()
        if min(vocab_size, context_length, d_model, num_heads, d_ff, num_layers, num_decoder_layers) <= 0 or d_model % num_heads:
            raise ValueError("positive dimensions and divisible d_model/num_heads required")
        if not 0 <= dropout < 1 or len({pad_token_id, eos_token_id, bos_token_id}) != 3:
            raise ValueError("invalid dropout or special token IDs")
        self.vocab_size, self.context_length, self.d_model = vocab_size, context_length, d_model
        self.pad_token_id, self.eos_token_id, self.bos_token_id = pad_token_id, eos_token_id, bos_token_id
        self.decoder_start_token_id = eos_token_id if decoder_start_token_id is None else decoder_start_token_id
        self.dropout, self.scale_embedding, self.tie_word_embeddings = dropout, scale_embedding, tie_word_embeddings
        self.embed_tokens = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.encoder_positions, self.decoder_positions = nn.Embedding(context_length + 2, d_model, padding_idx=pad_token_id), nn.Embedding(context_length + 2, d_model, padding_idx=pad_token_id)
        self.encoder_embedding_norm, self.decoder_embedding_norm = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.encoder = nn.ModuleList(EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers))
        self.decoder = nn.ModuleList(DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None: nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.embed_tokens.weight[self.pad_token_id].zero_()
            self.encoder_positions.weight[self.pad_token_id].zero_()
            self.decoder_positions.weight[self.pad_token_id].zero_()
        if self.tie_word_embeddings: self.lm_head.weight = self.embed_tokens.weight

    def encode(self, src, src_mask=None):
        if src.ndim != 2 or src.size(1) > self.context_length: raise ValueError("source exceeds context_length")
        mask = src.ne(self.pad_token_id) if src_mask is None else src_mask.bool()
        if mask.shape != src.shape: raise ValueError("src_mask must match source")
        positions = (torch.cumsum(mask.int(), dim=1) * mask.int()).long() + self.pad_token_id
        scale = math.sqrt(self.d_model) if self.scale_embedding else 1.0
        x = self.encoder_embedding_norm(self.embed_tokens(src) * scale + self.encoder_positions(positions))
        x = F.dropout(x, self.dropout, self.training) * mask.unsqueeze(-1)
        for layer in self.encoder: x = layer(x, mask)
        return x, mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory, memory_mask = self.encode(src, src_mask)
        mask = tgt.ne(self.pad_token_id) if tgt_mask is None else tgt_mask.bool()
        positions = (torch.cumsum(mask.int(), dim=1) * mask.int()).long() + self.pad_token_id
        scale = math.sqrt(self.d_model) if self.scale_embedding else 1.0
        x = self.embed_tokens(tgt) * scale
        x = self.decoder_embedding_norm(x + self.decoder_positions(positions))
        x = F.dropout(x, self.dropout, self.training) * mask.unsqueeze(-1)
        for layer in self.decoder: x = layer(x, memory, mask, memory_mask)
        return F.linear(x, self.lm_head.weight)
