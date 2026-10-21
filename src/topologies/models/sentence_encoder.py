import torch
from torch import nn


class SentenceEncoder(nn.Module):
    architecture_id = "sentence-encoder"
    PRESETS = {
        "tiny": {"vocab_size": 64, "context_length": 16, "d_model": 16, "pool": "mean"},
        "base": {"vocab_size": 256, "context_length": 32, "d_model": 64, "pool": "mean"},
    }

    def __init__(self, vocab_size=128, context_length=32, d_model=32, pool="mean", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if pool not in {"mean", "cls", "last"}:
            raise ValueError("pool must be mean, cls, or last")
        if min(vocab_size, context_length, d_model) <= 0 or vocab_size <= 1:
            raise ValueError("invalid encoder dimensions")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.pool = pool
        self.embed = nn.Embedding(vocab_size, d_model)
        self.hidden = nn.Linear(d_model, d_model, bias=False)
        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, tokens):
        tokens = torch.as_tensor(tokens)
        if tokens.ndim != 2 or tokens.shape[1] < 1 or tokens.shape[1] > self.context_length:
            raise ValueError("tokens must have shape (batch, length<=context_length)")
        hidden = self.output(self.projection(self.hidden(self.embed(tokens))))
        if self.pool == "cls":
            return hidden[:, 0]
        if self.pool == "last":
            return hidden[:, -1]
        return hidden.mean(dim=1)
