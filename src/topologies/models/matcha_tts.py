import torch
from torch import nn


class MatchaTTS(nn.Module):
    architecture_id = "matcha-tts"
    PRESETS = {"tiny": {"channels": 4, "conditioner_channels": 4, "hidden_channels": 16}}

    def __init__(self, channels=4, conditioner_channels=4, hidden_channels=16, sigma_min=1e-4):
        super().__init__()
        if channels <= 0 or conditioner_channels <= 0 or hidden_channels <= 0:
            raise ValueError("channel dimensions must be positive")
        if not 0.0 <= sigma_min < 1.0:
            raise ValueError("sigma_min must be in [0, 1)")
        self.channels = channels
        self.conditioner_channels = conditioner_channels
        self.sigma_min = float(sigma_min)
        self.input_projection = nn.Conv1d(channels + conditioner_channels + 1, hidden_channels, 3, padding=1)
        self.time_projection = nn.Sequential(nn.Linear(1, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, hidden_channels))
        self.network = nn.Sequential(nn.SiLU(), nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1), nn.SiLU(), nn.Conv1d(hidden_channels, channels, 3, padding=1))

    @staticmethod
    def training_target(clean, noise, sigma_min=1e-4):
        if clean.shape != noise.shape:
            raise ValueError("clean and noise must have the same shape")
        if not 0.0 <= sigma_min < 1.0:
            raise ValueError("sigma_min must be in [0, 1)")
        return clean - (1.0 - sigma_min) * noise

    @staticmethod
    def interpolate(clean, noise, time, sigma_min=1e-4):
        if clean.shape != noise.shape:
            raise ValueError("clean and noise must have the same shape")
        if not 0.0 <= sigma_min < 1.0:
            raise ValueError("sigma_min must be in [0, 1)")
        time = torch.as_tensor(time, dtype=clean.dtype, device=clean.device)
        if time.ndim == 0:
            time = time.reshape(1, 1, 1)
        elif time.ndim == 1:
            time = time[:, None, None]
        return (1.0 - (1.0 - sigma_min) * time) * noise + time * clean

    def forward(self, noisy_latent, time, conditioner):
        if noisy_latent.ndim != 3 or conditioner.ndim != 3:
            raise ValueError("latent and conditioner must have shape (batch, channels, length)")
        if noisy_latent.shape[0] != conditioner.shape[0] or noisy_latent.shape[2] != conditioner.shape[2]:
            raise ValueError("latent and conditioner batch and sequence dimensions must match")
        time = torch.as_tensor(time, dtype=noisy_latent.dtype, device=noisy_latent.device)
        if time.ndim == 0:
            time = time.expand(noisy_latent.shape[0])
        time = time.reshape(noisy_latent.shape[0], 1)
        time_channel = time.unsqueeze(-1).expand(-1, -1, noisy_latent.shape[-1])
        hidden = self.input_projection(torch.cat((noisy_latent, conditioner, time_channel), dim=1))
        hidden = hidden + self.time_projection(time).unsqueeze(-1)
        return self.network(hidden)
