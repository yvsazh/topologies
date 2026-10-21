import torch
import torch.nn as nn
import torch.nn.functional as F


class HyenaOperator(nn.Module):
    def __init__(self, d_model, filter_dim=16, dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        self.d_model = d_model
        self.position_filter = nn.Sequential(nn.Linear(1, filter_dim), nn.SiLU(), nn.Linear(filter_dim, d_model))
        self.in_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def filter_from_positions(self, positions):
        return self.position_filter(positions.unsqueeze(-1))

    def _causal_convolution(self, values, kernel):
        length = values.shape[1]
        weights = kernel.transpose(0, 1).flip(-1).unsqueeze(1)
        padded = F.pad(values.transpose(1, 2), (length - 1, 0))
        return F.conv1d(padded, weights, groups=self.d_model).transpose(1, 2)

    def forward(self, x):
        length = x.shape[1]
        positions = torch.arange(length, device=x.device, dtype=x.dtype) / max(length, 1)
        kernel = self.filter_from_positions(positions)
        first, gate, third = self.in_proj(x).chunk(3, dim=-1)
        return self.out_proj(self._causal_convolution(self._causal_convolution(first, kernel) * gate, kernel) * third)


class HyenaBlock(nn.Module):
    def __init__(self, d_model, filter_dim=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.operator = HyenaOperator(d_model, filter_dim)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))

    def forward(self, x):
        x = x + self.operator(self.norm(x))
        return x + self.ff(self.ff_norm(x))


class Hyena(nn.Module):
    architecture_id = "hyena"
    PRESETS = {
        "tiny": dict(vocab_size=32, context_length=32, d_model=16, num_layers=1, filter_dim=16),
        "base": dict(vocab_size=128, context_length=256, d_model=32, num_layers=2, filter_dim=24),
        "hyenadna": dict(vocab_size=5, context_length=1024, d_model=16, num_layers=2, filter_dim=16),
    }

    def __init__(self, vocab_size=32, context_length=32, d_model=16, num_layers=1, filter_dim=16, dropout=0.0, **kwargs):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([HyenaBlock(d_model, filter_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        if tokens.shape[1] > self.context_length:
            raise ValueError("input exceeds context_length")
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))
