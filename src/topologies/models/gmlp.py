import torch
from torch import nn


class SpatialGatingUnit(nn.Module):
    def __init__(self, hidden_dim, num_tokens):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim // 2)
        self.token_projection = nn.Linear(num_tokens, num_tokens)
        nn.init.zeros_(self.token_projection.weight)
        nn.init.ones_(self.token_projection.bias)

    def forward(self, x):
        left, right = x.chunk(2, dim=-1)
        right = self.norm(right).transpose(1, 2)
        right = self.token_projection(right).transpose(1, 2)
        return left * right


class GMLP(nn.Module):
    architecture_id = "gmlp"
    def __init__(self, num_tokens=4, num_classes=2, channels=4, depth=1, expansion_factor=2, dropout=0.0):
        super().__init__()
        if channels % 2:
            raise ValueError("channels must be even")
        if num_tokens < 1 or num_classes < 1 or depth < 1 or expansion_factor < 1:
            raise ValueError("model dimensions and depth must be positive")
        hidden_dim = channels * expansion_factor
        self.num_tokens = num_tokens
        self.embedding = nn.Linear(channels, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(hidden_dim),
                "channel_in": nn.Linear(hidden_dim, hidden_dim * 2),
                "gate": SpatialGatingUnit(hidden_dim * 2, num_tokens),
                "channel_out": nn.Linear(hidden_dim, hidden_dim),
                "dropout": nn.Dropout(dropout),
            })
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("input must have shape (batch, tokens, channels)")
        if x.shape[1] != self.num_tokens:
            raise ValueError(f"expected {self.num_tokens} tokens, got {x.shape[1]}")
        x = self.embedding(x)
        for block in self.blocks:
            residual = x
            y = block["channel_in"](block["norm"](x))
            y = torch.nn.functional.gelu(y)
            y = block["gate"](y)
            y = block["channel_out"](y)
            x = residual + block["dropout"](y)
        return self.classifier(self.norm(x).mean(dim=1))
