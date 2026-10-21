import torch
from torch import nn


class ResidualUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.net(x)


class SEANetEncoderStage(nn.Module):
    component_id = "seanet"
    def __init__(self, latent_channels=4, stride=2, hidden_channels=8, in_channels=1):
        super().__init__()
        if latent_channels <= 0 or stride <= 1 or hidden_channels <= 0 or in_channels <= 0:
            raise ValueError("channel counts must be positive and stride must exceed one")
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3),
            ResidualUnit(hidden_channels),
            nn.ELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=2 * stride, stride=stride, padding=stride // 2),
            ResidualUnit(hidden_channels),
            nn.ELU(),
            nn.Conv1d(hidden_channels, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, waveform):
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3:
            raise ValueError("waveform must have shape (batch, time) or (batch, channels, time)")
        if waveform.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} waveform channels, got {waveform.shape[1]}")
        return self.encoder(waveform)
