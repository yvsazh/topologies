import torch
from torch import nn


class GradTTS(nn.Module):
    architecture_id = "grad-tts"

    def __init__(self, channels=4, mel_bins=4, step_size=0.1):
        super().__init__()
        if channels <= 0 or mel_bins <= 0:
            raise ValueError("channels and mel_bins must be positive")
        self.channels = channels
        self.mel_bins = mel_bins
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.step_size = float(step_size)
        self.score = nn.Conv1d(channels * 2 + 1, channels, kernel_size=1)

    def forward(self, latent, mel_conditioner, time=0.5, beta_t=1.0):
        if latent.ndim != 3 or mel_conditioner.ndim != 3:
            raise ValueError("latent and mel_conditioner must have shape (batch, channels, time)")
        if latent.shape[1] != self.channels:
            raise ValueError("latent channel count does not match channels")
        if mel_conditioner.shape[1] != self.mel_bins:
            raise ValueError("mel conditioner channel count does not match mel_bins")
        if latent.shape[0] != mel_conditioner.shape[0] or latent.shape[-1] != mel_conditioner.shape[-1]:
            raise ValueError("latent and mel_conditioner batch and sequence dimensions must match")
        time = torch.as_tensor(time, dtype=latent.dtype, device=latent.device)
        beta_t = torch.as_tensor(beta_t, dtype=latent.dtype, device=latent.device)
        if time.ndim == 0:
            time = time.expand(latent.shape[0])
        if beta_t.ndim == 0:
            beta_t = beta_t.expand(latent.shape[0])
        if time.shape != (latent.shape[0],) or beta_t.shape != (latent.shape[0],):
            raise ValueError("time and beta_t must be scalars or one value per batch item")
        if torch.any(beta_t < 0):
            raise ValueError("beta_t must be non-negative")
        conditioner = mel_conditioner.mean(dim=1, keepdim=True).expand(-1, self.channels, -1)
        time_channel = time[:, None, None].expand(-1, 1, latent.shape[-1])
        score = self.score(torch.cat((latent, conditioner, time_channel), dim=1))
        drift = 0.5 * (conditioner - latent - score) * beta_t[:, None, None]
        return latent - self.step_size * drift
