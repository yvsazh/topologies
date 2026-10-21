import torch
from torch import nn
import torch.nn.functional as F

class DDSPSynthesizer(nn.Module):
    architecture_id = 'ddsp-synthesizer'

    @classmethod
    def preset(cls, name: str, **geometry):
        harmonics = {'tiny': 8, 'small': 32, 'base': 64}
        if name not in harmonics:
            raise ValueError('preset must be tiny, small, or base')
        return cls(n_harmonics=harmonics[name], **geometry)

    def __init__(self, sample_rate: int=16000, n_harmonics: int=32, control_rate: int=100, fir_size: int=9, noise_bands: int=1):
        super().__init__()
        if min(sample_rate, n_harmonics, control_rate, fir_size, noise_bands) <= 0:
            raise ValueError('dimensions must be positive')
        self.sample_rate, self.n_harmonics, self.control_rate = (sample_rate, n_harmonics, control_rate)
        self.noise_bands = noise_bands
        self.harmonic_amp = nn.Parameter(torch.zeros(n_harmonics))
        taps = torch.zeros(noise_bands, fir_size)
        taps[:, fir_size // 2] = 1.0 / noise_bands
        self.noise_taps = nn.Parameter(taps)

    def forward(self, pitch: torch.Tensor, loudness: torch.Tensor, num_samples: int | None=None, state: torch.Tensor | None=None):
        if pitch.ndim == 1:
            pitch = pitch.unsqueeze(0)
        if loudness.ndim == 1:
            loudness = loudness.unsqueeze(0)
        if pitch.ndim != 2 or loudness.ndim != 2 or pitch.shape[0] != loudness.shape[0]:
            raise ValueError('controls must be [B,frames]')
        n = num_samples if num_samples is not None else round(pitch.shape[1] * self.sample_rate / self.control_rate)
        if n <= 0:
            raise ValueError('num_samples must be positive')
        p = F.interpolate(pitch.unsqueeze(1), size=n, mode='linear', align_corners=True).squeeze(1)
        a = F.interpolate(loudness.unsqueeze(1), size=n, mode='linear', align_corners=True).squeeze(1).clamp_min(0)
        phase0 = torch.zeros(p.shape[0], device=p.device, dtype=p.dtype) if state is None else state
        increments = 2 * torch.pi * p / self.sample_rate
        phase = torch.cumsum(increments, dim=1) + phase0[:, None]
        harmonics = torch.arange(1, self.n_harmonics + 1, device=p.device, dtype=p.dtype)
        freq = p.unsqueeze(-1) * harmonics
        valid = (freq < self.sample_rate / 2).to(p.dtype)
        wave = (torch.sin(phase.unsqueeze(-1) * harmonics) * valid * self.harmonic_amp.to(p).exp()).sum(-1)
        wave = wave * a / valid.sum(-1).clamp_min(1)
        noise = torch.randn_like(wave) * a
        taps = self.noise_taps.to(p)
        filtered = sum((F.conv1d(F.pad(noise.unsqueeze(1), (taps.shape[-1] - 1, 0)), band.flip(0).view(1, 1, -1)).squeeze(1) for band in taps))
        sample = wave + filtered
        next_phase = phase0 + increments.sum(-1)
        return (sample, next_phase)
