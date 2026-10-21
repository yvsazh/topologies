from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LongformerAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, attention_window: int):
        super().__init__()
        if hidden_size % num_heads or attention_window < 2:
            raise ValueError("hidden_size must divide num_heads and window must be >= 2")
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.head_dim, self.attention_window = hidden_size // num_heads, attention_window
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.global_query = nn.Linear(hidden_size, hidden_size)
        self.global_key = nn.Linear(hidden_size, hidden_size)
        self.global_value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

    def _heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, attention_mask=None, global_attention_mask=None):
        batch, length, _ = x.shape
        device = x.device
        valid = torch.ones((batch, length), dtype=torch.bool, device=device) if attention_mask is None else attention_mask.bool()
        glob = torch.zeros_like(valid) if global_attention_mask is None else global_attention_mask.bool() & valid
        q, k, v = (self._heads(layer(x)) for layer in (self.query, self.key, self.value))
        gq, gk, gv = (self._heads(layer(x)) for layer in (self.global_query, self.global_key, self.global_value))
        positions = torch.arange(length, device=device)
        local = (positions[:, None] - positions[None, :]).abs() <= self.attention_window // 2
        allowed = local[None, None] & valid[:, None, None, :] & ~glob[:, None, None, :]
        allowed = allowed.expand(batch, 1, length, length).clone()
        allowed |= glob[:, None, :, None] & valid[:, None, None, :]
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.head_dim**0.5
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        if glob.any():
            gscores = torch.matmul(q, gk.transpose(-1, -2)) / self.head_dim**0.5
            gallowed = glob[:, None, None, :] & valid[:, None, None, :]
            gscores = gscores.masked_fill(~gallowed, torch.finfo(gscores.dtype).min)
            gw = torch.softmax(gscores, dim=-1)
            gcontext = torch.matmul(gw, gv)
            context = context + gcontext
            global_context = torch.matmul(torch.softmax(
                (gq @ gk.transpose(-1, -2) / self.head_dim**0.5).masked_fill(
                    ~(valid[:, None, None, :] & glob[:, None, None, :]), torch.finfo(x.dtype).min), -1), gv)
            context = torch.where(glob[:, None, :, None], global_context, context)
        context = context.transpose(1, 2).contiguous().view(batch, length, self.hidden_size)
        return self.output(context) * valid.unsqueeze(-1)


class LongformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, attention_window):
        super().__init__()
        self.attention = LongformerAttention(hidden_size, num_heads, attention_window)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, attention_mask=None, global_attention_mask=None):
        x = self.attention_norm(x + self.attention(x, attention_mask, global_attention_mask))
        return self.output_norm(x + self.output(F.gelu(self.intermediate(x))))


class Longformer(nn.Module):
    architecture_id = "longformer"
    PRESETS = {
        "tiny": {"vocab_size": 32, "context_length": 16, "hidden_size": 16, "num_layers": 2, "num_heads": 2, "intermediate_size": 24, "attention_window": [8, 8], "pad_token_id": 0, "mask_token_id": 4},
        "base": {"vocab_size": 50265, "context_length": 4098, "hidden_size": 768, "num_layers": 12, "num_heads": 12, "intermediate_size": 3072, "attention_window": [512] * 12},
    }

    def __init__(self, vocab_size=50265, context_length=4098, hidden_size=768,
                  num_layers=12, num_heads=12, intermediate_size=3072,
                  attention_window=512, pad_token_id=1, mask_token_id=50264, dropout=0.0):
        super().__init__()
        windows = [attention_window] * num_layers if isinstance(attention_window, int) else list(attention_window)
        if len(windows) != num_layers:
            raise ValueError("attention_window must have one value per layer")
        self.vocab_size, self.context_length = vocab_size, context_length
        self.hidden_size, self.num_layers, self.num_heads = hidden_size, num_layers, num_heads
        self.attention_window, self.pad_token_id, self.mask_token_id = windows, pad_token_id, mask_token_id
        self.embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(context_length, hidden_size)
        self.embedding_norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList(LongformerLayer(hidden_size, num_heads, intermediate_size, w) for w in windows)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.weight
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, attention_mask=None, global_attention_mask=None):
        if tokens.shape[1] > self.context_length:
            raise ValueError("input exceeds configured context_length")
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        valid = tokens.ne(self.pad_token_id) if attention_mask is None else attention_mask.bool()
        x = self.dropout(self.embedding_norm(self.embeddings(tokens) + self.position_embeddings(positions)))
        for layer in self.layers:
            x = layer(x, valid, global_attention_mask)
        return self.lm_head(self.final_norm(x))
