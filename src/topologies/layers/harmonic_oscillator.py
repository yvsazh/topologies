import torch
from torch import nn

class HarmonicOscillator(nn.Module):
    component_id = 'harmonic-oscillator'

    def __init__(self, sample_rate, harmonics=8):
        super().__init__()
        if min(sample_rate, harmonics) <= 0:
            raise ValueError('sample rate/harmonics must be positive')
        self.sample_rate = sample_rate
        self.harmonics = harmonics

    def forward(self, pitch, amplitude, num_samples, phase=0.0):
        if num_samples < 0:
            raise ValueError('num_samples must be nonnegative')
        pitch = torch.as_tensor(pitch)
        if pitch.ndim != 2 or not pitch.is_floating_point():
            raise ValueError('pitch controls must be floating [B,T]')
        device = pitch.device
        amplitude = torch.as_tensor(amplitude, device=device, dtype=pitch.dtype)
        if amplitude.ndim == 0:
            amplitude = amplitude.expand(pitch.shape[0], pitch.shape[-1])
        elif amplitude.ndim == 1:
            if amplitude.numel() != pitch.shape[0]:
                raise ValueError('one-dimensional amplitude must contain one value per batch')
            amplitude = amplitude[:, None]
        if amplitude.ndim != 2 or amplitude.shape[0] != pitch.shape[0]:
            raise ValueError('amplitude controls must be scalar or [B,T]')
        if pitch.shape[-1] != num_samples:
            pitch = torch.nn.functional.interpolate(pitch.unsqueeze(1), size=num_samples, mode='linear', align_corners=False).squeeze(1)
        if amplitude.shape[-1] != num_samples:
            amplitude = torch.nn.functional.interpolate(amplitude.unsqueeze(1), size=num_samples, mode='linear', align_corners=False).squeeze(1)
        freq = pitch.clamp_min(0)
        increments = torch.cat((freq.new_zeros((*freq.shape[:-1], 1)), freq[..., :-1]), dim=-1)
        ph = phase + 2 * torch.pi * torch.cumsum(increments / self.sample_rate, dim=-1)
        hs = torch.arange(1, self.harmonics + 1, device=device, dtype=pitch.dtype)
        wave = torch.sin(ph.unsqueeze(-1) * hs) * (freq.unsqueeze(-1) * hs < self.sample_rate / 2)
        return (wave / hs).sum(-1) * amplitude
