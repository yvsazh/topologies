import math
import torch
from torch import nn


class AttentionNMT(nn.Module):
    architecture_id = "attention-nmt"
    PRESETS = {
        "tiny": {"vocab_size": 64, "context_length": 8, "d_model": 16, "mode": "bahdanau"},
        "base": {"vocab_size": 256, "context_length": 16, "d_model": 32, "mode": "bahdanau"},
    }

    def __init__(self, vocab_size=128, context_length=16, d_model=32, mode="bahdanau", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"bahdanau", "luong", "luong-general", "gnmt", "convs2s", "transformer"}:
            raise ValueError("mode must be bahdanau, luong, luong-general, gnmt, convs2s, or transformer")
        if min(vocab_size, context_length, d_model) <= 0 or vocab_size <= 1 or d_model % 2:
            raise ValueError("invalid translation dimensions")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.mode = mode
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.energy = nn.Parameter(torch.zeros(d_model))
        self.align = nn.Linear(d_model, d_model, bias=False)
        self.gnmt_bi = nn.LSTM(d_model, d_model // 2, batch_first=True, bidirectional=True)
        self.gnmt_res = nn.Linear(d_model, d_model, bias=False)
        self.conv = nn.Conv1d(d_model, d_model * 2, 3, padding=1)
        self.out = nn.Linear(d_model, vocab_size)
        self.energy.data[0] = 1.0

    def _ids(self, tokens, name):
        tokens = torch.as_tensor(tokens)
        if tokens.ndim != 2 or tokens.shape[1] < 1 or tokens.shape[1] > self.context_length:
            raise ValueError(f"{name} must have shape (batch, length<=context_length)")
        return tokens

    def bahdanau(self, query, memory):
        energy = torch.tanh(self.query(query).unsqueeze(1) + self.key(memory))
        return energy @ self.energy

    def luong_dot(self, query, memory):
        return (memory * query.unsqueeze(1)).sum(dim=-1)

    def luong_general(self, query, memory):
        return (self.align(memory) * query.unsqueeze(1)).sum(dim=-1)

    def scaled_dot(self, query, memory):
        return (query.unsqueeze(1) @ memory.transpose(1, 2)).squeeze(1) / math.sqrt(query.shape[-1])

    def glu(self, mixed):
        values, gate = mixed.split(self.d_model, dim=-1)
        return values * torch.sigmoid(gate)

    def gnmt_encode(self, embedded):
        first, _ = self.gnmt_bi(embedded)
        return first + self.gnmt_res(first)

    def encode(self, src):
        embedded = self.src_embed(src)
        if self.mode == "gnmt":
            return self.gnmt_encode(embedded)
        if self.mode == "convs2s":
            return self.glu(self.conv(embedded.transpose(1, 2)).transpose(1, 2))
        return self.encoder(embedded)

    def attend(self, query, memory):
        if self.mode == "luong":
            scores = self.luong_dot(query, memory)
        elif self.mode == "luong-general":
            scores = self.luong_general(query, memory)
        elif self.mode == "transformer":
            scores = self.scaled_dot(query, memory)
        else:
            scores = self.bahdanau(query, memory)
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * memory).sum(dim=1)

    def forward(self, src, tgt):
        src = self._ids(src, "src")
        tgt = self._ids(tgt, "tgt")
        memory = self.encode(src)
        steps = []
        for index in range(tgt.shape[1]):
            query = self.tgt_embed(tgt[:, index])
            steps.append(self.out(self.attend(query, memory) + query))
        return torch.stack(steps, dim=1)
