import torch
from torch import nn


class MixerBlock(nn.Module):
    component_id = "tsmixer"

    def __init__(self, lookback, features, hidden_dim, dropout):
        super().__init__()
        self.lookback = lookback
        self.features = features
        self.time_norm = nn.LayerNorm((lookback, features))
        self.time_mlp = nn.Sequential(
            nn.Linear(lookback, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, lookback),
            nn.Dropout(dropout),
        )
        self.feature_norm = nn.LayerNorm((lookback, features))
        self.feature_mlp = nn.Sequential(
            nn.Linear(features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, features),
            nn.Dropout(dropout),
        )

    def forward(self, values):
        time_values = self.time_norm(values).transpose(1, 2)
        values = values + self.time_mlp(time_values).transpose(1, 2)
        return values + self.feature_mlp(self.feature_norm(values))


class TSMixer(nn.Module):
    def __init__(self, lookback, features, horizon=None, hidden_dim=None, num_blocks=2, dropout=0.0):
        super().__init__()
        if lookback <= 0 or features <= 0 or num_blocks <= 0:
            raise ValueError("lookback, features, and num_blocks must be positive")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        hidden_dim = hidden_dim or max(lookback, features)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.lookback = lookback
        self.features = features
        self.horizon = lookback if horizon is None else horizon
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        self.blocks = nn.Sequential(
            *(MixerBlock(lookback, features, hidden_dim, dropout) for _ in range(num_blocks))
        )
        self.horizon_projection = nn.Linear(lookback, self.horizon)

    def forward(self, values):
        values = torch.as_tensor(values)
        if values.ndim != 3 or values.shape[1:] != (self.lookback, self.features):
            raise ValueError("values must have shape (batch, lookback, features)")
        mixed = self.blocks(values)
        return self.horizon_projection(mixed.transpose(1, 2)).transpose(1, 2)
