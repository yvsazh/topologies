import torch
from torch import nn


class WaveGlow(nn.Module):
    component_id = "waveglow"
    def __init__(self, channels=4, hidden_channels=16, mel_channels=80):
        super().__init__()
        if channels < 2 or channels % 2:
            raise ValueError("channels must be an even integer greater than one")
        if hidden_channels <= 0 or mel_channels <= 0:
            raise ValueError("hidden_channels and mel_channels must be positive")
        self.channels = channels
        self.mel_channels = mel_channels
        half = channels // 2
        self.conditioner = nn.Sequential(
            nn.Conv1d(half + mel_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    def forward(self, waveform, mel, inverse=False, return_logdet=False):
        if waveform.ndim != 3 or waveform.shape[1] != self.channels:
            raise ValueError(f"waveform must have shape (batch, {self.channels}, length)")
        if mel.ndim != 3 or mel.shape != (waveform.shape[0], self.mel_channels, waveform.shape[2]):
            raise ValueError(
                f"mel must have shape (batch, {self.mel_channels}, length) aligned with waveform"
            )
        first, second = waveform.chunk(2, dim=1)
        log_scale, shift = self.conditioner(torch.cat((first, mel), dim=1)).chunk(2, dim=1)
        if inverse:
            transformed = (second - shift) * torch.exp(-log_scale)
            logdet = -log_scale.flatten(1).sum(dim=1)
        else:
            transformed = second * torch.exp(log_scale) + shift
            logdet = log_scale.flatten(1).sum(dim=1)
        output = torch.cat((first, transformed), dim=1)
        if return_logdet:
            return output, logdet
        return output

    def inverse(self, latent, mel, return_logdet=False):
        return self.forward(latent, mel, inverse=True, return_logdet=return_logdet)
