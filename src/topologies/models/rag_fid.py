import math

import torch
from torch import nn


class RagFid(nn.Module):
    architecture_id = "rag-fid"
    PRESETS = {
        "tiny": {"vocab_size": 64, "context_length": 16, "d_model": 16, "num_heads": 2, "num_passages": 2, "mode": "rag"},
        "base": {"vocab_size": 256, "context_length": 32, "d_model": 64, "num_heads": 4, "num_passages": 4, "mode": "fid"},
    }

    def __init__(self, vocab_size=128, context_length=16, d_model=32, num_heads=4, num_passages=2, mode="rag", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"rag", "fid"}:
            raise ValueError("mode must be rag or fid")
        if min(vocab_size, context_length, d_model, num_heads, num_passages) <= 0 or vocab_size <= 1:
            raise ValueError("invalid model dimensions")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_passages = num_passages
        self.mode = mode
        self.embed = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(context_length, d_model)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.passage = nn.Linear(d_model, d_model, bias=False)
        self.encoder = nn.Linear(d_model, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.cross_q = nn.Linear(d_model, d_model, bias=False)
        self.cross_k = nn.Linear(d_model, d_model, bias=False)
        self.cross_v = nn.Linear(d_model, d_model, bias=False)
        self.cross_o = nn.Linear(d_model, d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def encode_passages(self, passages):
        if passages.ndim != 3 or passages.shape[-1] > self.context_length or passages.shape[1] < 1:
            raise ValueError("passages must have shape (batch, passages, length<=context_length)")
        positions = torch.arange(passages.shape[-1], device=passages.device)
        hidden = self.embed(passages) + self.position(positions)
        return self.encoder(hidden)

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

    def forward(self, src, tgt, passages=None):
        src = torch.as_tensor(src)
        tgt = torch.as_tensor(tgt)
        if src.ndim != 2 or tgt.ndim != 2 or src.shape[1] > self.context_length or tgt.shape[1] > self.context_length:
            raise ValueError("src and tgt must have shape (batch, length<=context_length)")
        if passages is None:
            passages = src.unsqueeze(1)
        else:
            passages = torch.as_tensor(passages)
        encoded = self.encode_passages(passages)
        if encoded.shape[0] != src.shape[0]:
            raise ValueError("passages batch must match src")
        query = self.query(self.embed(src).mean(dim=1))
        passage_vectors = self.passage(encoded.mean(dim=2))
        scores = torch.einsum("bd,bpd->bp", query, passage_vectors)
        if self.mode == "rag":
            choice = scores.argmax(dim=-1)
            memory = encoded[torch.arange(encoded.shape[0], device=encoded.device), choice]
        else:
            memory = encoded.reshape(encoded.shape[0], encoded.shape[1] * encoded.shape[2], encoded.shape[3])
        positions = torch.arange(tgt.shape[1], device=tgt.device)
        hidden = self.embed(tgt) + self.position(positions)
        hidden = hidden + self._attend(self.q_proj, self.k_proj, self.v_proj, self.o_proj, hidden, hidden, True)
        hidden = hidden + self._attend(self.cross_q, self.cross_k, self.cross_v, self.cross_o, hidden, memory, False)
        return self.lm_head(hidden)
