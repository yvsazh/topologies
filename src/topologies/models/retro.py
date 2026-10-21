import math

import torch
from torch import nn


class Retro(nn.Module):
    architecture_id = "retro"
    PRESETS = {
        "tiny": {"vocab_size": 64, "context_length": 8, "d_model": 16, "num_heads": 2, "chunk_size": 2},
        "base": {"vocab_size": 256, "context_length": 16, "d_model": 64, "num_heads": 4, "chunk_size": 4},
    }

    def __init__(self, vocab_size=128, context_length=16, d_model=32, num_heads=4, chunk_size=4, dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if min(vocab_size, context_length, d_model, num_heads, chunk_size) <= 0 or vocab_size <= 1:
            raise ValueError("invalid model dimensions")
        if d_model % num_heads or context_length < chunk_size:
            raise ValueError("d_model must be divisible by num_heads and context must hold one chunk")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(context_length, d_model)
        self.frozen = nn.Linear(d_model, d_model, bias=False)
        self.frozen.weight.requires_grad_(False)
        self.self_q = nn.Linear(d_model, d_model, bias=False)
        self.self_k = nn.Linear(d_model, d_model, bias=False)
        self.self_v = nn.Linear(d_model, d_model, bias=False)
        self.self_o = nn.Linear(d_model, d_model, bias=False)
        self.cross_q = nn.Linear(d_model, d_model, bias=False)
        self.cross_k = nn.Linear(d_model, d_model, bias=False)
        self.cross_v = nn.Linear(d_model, d_model, bias=False)
        self.cross_o = nn.Linear(d_model, d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _attend(self, q_proj, k_proj, v_proj, o_proj, query, key, causal):
        batch, steps, width = query.shape
        heads = self.num_heads
        head_dim = width // heads
        q = q_proj(query).view(batch, steps, heads, head_dim).transpose(1, 2)
        k = k_proj(key).view(batch, key.shape[1], heads, head_dim).transpose(1, 2)
        v = v_proj(key).view(batch, key.shape[1], heads, head_dim).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
        if causal:
            mask = torch.ones(steps, key.shape[1], device=query.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        mixed = torch.softmax(scores, dim=-1) @ v
        return o_proj(mixed.transpose(1, 2).contiguous().view(batch, steps, width))

    def _database(self, chunks, database):
        if database is None:
            neighbor = chunks.detach()
            continuation = torch.zeros_like(neighbor)
            continuation[:, :-1] = neighbor[:, 1:]
            return torch.cat((neighbor, continuation), dim=2)
        database = torch.as_tensor(database)
        if database.ndim != 3 or database.shape[-1] != self.chunk_size or database.shape[1] < 1:
            raise ValueError("database must have shape (batch, items, chunk_size)")
        with torch.no_grad():
            encoded = self.frozen(self.embed(database))
        scores = torch.einsum("bcd,bid->bci", chunks.mean(dim=2), encoded.mean(dim=2))
        choice = scores.argmax(dim=-1)
        following = (choice + 1).clamp(max=encoded.shape[1] - 1)
        batch = torch.arange(encoded.shape[0], device=encoded.device)[:, None]
        return torch.cat((encoded[batch, choice], encoded[batch, following]), dim=2)

    def forward(self, tokens, database=None):
        tokens = torch.as_tensor(tokens)
        if tokens.ndim != 2 or tokens.shape[1] < 1 or tokens.shape[1] > self.context_length:
            raise ValueError("tokens must have shape (batch, length<=context_length)")
        original = tokens.shape[1]
        remainder = original % self.chunk_size
        if remainder:
            tokens = torch.nn.functional.pad(tokens, (0, self.chunk_size - remainder))
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.embed(tokens) + self.position(positions)
        hidden = hidden + self._attend(self.self_q, self.self_k, self.self_v, self.self_o, hidden, hidden, True)
        chunks = hidden.view(tokens.shape[0], tokens.shape[1] // self.chunk_size, self.chunk_size, self.d_model)
        retrieved = self._database(chunks, database)
        attended = chunks.clone()
        for index in range(1, chunks.shape[1]):
            attended[:, index] = chunks[:, index] + self._attend(
                self.cross_q, self.cross_k, self.cross_v, self.cross_o, chunks[:, index], retrieved[:, index - 1], False)
        flat = attended.reshape(tokens.shape[0], tokens.shape[1], self.d_model)
        return self.lm_head(flat)[:, :original]
