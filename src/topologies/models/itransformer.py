import torch
from torch import nn


class ITransformer(nn.Module):
    architecture_id = "itransformer"
    component_id = "itransformer"

    def __init__(self, input_length, horizon, variates, d_model=16, n_heads=4, d_ff=32, dropout=0.0):
        super().__init__()
        if min(input_length, horizon, variates, d_model, n_heads, d_ff) <= 0:
            raise ValueError("dimensions must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.input_length = input_length
        self.variates = variates
        self.value_embedding = nn.Linear(input_length, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, horizon)

    def forward(self, values):
        if values.ndim != 3 or values.shape[1:] != (self.input_length, self.variates):
            raise ValueError("values must have shape (batch, input_length, variates)")
        tokens = self.value_embedding(values.transpose(1, 2))
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.attention_norm(tokens + attended)
        tokens = self.feed_forward_norm(tokens + self.feed_forward(tokens))
        return self.projection(tokens).transpose(1, 2)
