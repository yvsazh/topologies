import torch
from torch import nn
from torch.nn import functional as F


class Wav2Vec2(nn.Module):
    architecture_id = "wav2vec2"
    def __init__(self, channels=4, stride=2, kernel_size=4):
        super().__init__()
        if channels <= 0 or stride <= 0 or kernel_size <= 0:
            raise ValueError("channels, stride, and kernel_size must be positive")
        self.feature_encoder = nn.Conv1d(1, channels, kernel_size=kernel_size, stride=stride)
        self.feature_norm = nn.LayerNorm(channels)
        self.mask_token = nn.Parameter(torch.zeros(channels))

    def forward(self, waveform, mask=None):
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must have shape (batch, time) or (batch, 1, time)")
        waveform = F.layer_norm(waveform, (waveform.shape[-1],))
        latent = self.feature_encoder(waveform).transpose(1, 2)
        latent = F.gelu(self.feature_norm(latent))
        if mask is not None:
            if mask.shape != latent.shape[:2]:
                raise ValueError("mask must have shape (batch, latent_time)")
            mask = mask.to(device=latent.device, dtype=torch.bool)
            latent = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), latent)
        return latent
