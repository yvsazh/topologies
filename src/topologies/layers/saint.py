import torch
from torch import nn


class SAINTIntersampleAttention(nn.Module):
    component_id = "saint"

    def __init__(self, n_features=8, width=32, heads=4, dropout=0.0):
        super().__init__()
        if n_features <= 0 or width <= 0 or heads <= 0 or n_features * width % heads:
            raise ValueError("positive feature/width/head dimensions are required and n_features*width must divide by heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.n_features = n_features
        self.width = width
        self.row_width = n_features * width
        self.attention = nn.MultiheadAttention(
            self.row_width, heads, dropout=dropout, batch_first=True
        )

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1:] != (self.n_features, self.width):
            raise ValueError("tokens must have shape (batch, n_features, width)")
        rows = tokens.reshape(tokens.shape[0], self.row_width).unsqueeze(0)
        attended, _ = self.attention(rows, rows, rows, need_weights=False)
        return attended.squeeze(0).reshape_as(tokens)
