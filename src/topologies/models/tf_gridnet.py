import torch
from torch import nn


class TFGridNet(nn.Module):
    architecture_id = "tf-gridnet"
    def __init__(self, channels=4, num_heads=2):
        super().__init__()
        if channels <= 0 or num_heads <= 0 or channels % num_heads:
            raise ValueError("channels must be positive and divisible by num_heads")
        self.temporal_attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.frequency_attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(channels)
        self.frequency_norm = nn.LayerNorm(channels)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("input must have shape (batch, channels, time, frequency)")
        batch, channels, time, frequency = x.shape
        temporal = x.permute(0, 3, 2, 1).reshape(batch * frequency, time, channels)
        temporal_update, _ = self.temporal_attention(temporal, temporal, temporal, need_weights=False)
        temporal = self.temporal_norm(temporal + temporal_update)
        grid = temporal.reshape(batch, frequency, time, channels).permute(0, 2, 1, 3)
        spectral = grid.reshape(batch * time, frequency, channels)
        spectral_update, _ = self.frequency_attention(spectral, spectral, spectral, need_weights=False)
        spectral = self.frequency_norm(spectral + spectral_update)
        return spectral.reshape(batch, time, frequency, channels).permute(0, 3, 1, 2)
