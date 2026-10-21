import torch
from torch import nn


class ViT22B(nn.Module):
    architecture_id = "vit-22b"
    def __init__(self, channels=8, num_heads=2, mlp_ratio=4, dropout=0.0):
        super().__init__()
        if channels <= 0 or num_heads <= 0 or channels % num_heads:
            raise ValueError("channels must be positive and divisible by num_heads")
        hidden = channels * mlp_ratio
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape (batch, sequence, channels)")
        normalized = self.norm(tokens)
        attention, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        mlp = self.mlp(normalized)
        return tokens + self.dropout(attention) + self.dropout(mlp)
