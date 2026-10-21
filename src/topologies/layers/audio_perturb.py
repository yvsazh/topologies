import torch
from torch import nn


class AudioPerturb(nn.Module):
    component_id = "audio-perturb"
    def __init__(self):
        super().__init__()

    def forward(self, wave, rate=1.0, sample_rate=16000, n_semitones=0.0, mode="speed"):
        if mode == "speed":
            return self.speed(wave, rate)
        if mode == "pitch":
            return self.pitch(wave, sample_rate, n_semitones)
        raise ValueError("mode must be 'speed' or 'pitch'")

    def speed(self, wave, rate):
        rate = float(rate)
        if rate <= 0:
            raise ValueError("rate must be greater than zero")
        if rate == 1:
            return wave
        if wave.ndim not in (1, 2):
            raise ValueError("wave must have shape (time,) or (batch, time)")
        if not wave.is_floating_point():
            raise TypeError("wave must be a floating-point tensor")
        old_length = wave.shape[-1]
        length = int(old_length / rate)
        if length < 1 and old_length:
            length = 1
        if length == old_length:
            return wave
        if old_length == 0:
            return wave.new_empty((*wave.shape[:-1], length))
        if length == 0:
            return wave[..., :0]
        positions = torch.arange(length, device=wave.device, dtype=wave.dtype)
        positions = (positions * rate).clamp(max=old_length - 1)
        left = positions.floor().long()
        right = (left + 1).clamp(max=old_length - 1)
        fraction = positions - left.to(wave.dtype)
        return wave[..., left] + (wave[..., right] - wave[..., left]) * fraction

    def pitch(self, wave, sample_rate, n_semitones):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if n_semitones == 0:
            return wave
        if wave.ndim not in (1, 2):
            raise ValueError("wave must have shape (time,) or (batch, time)")
        if not wave.is_floating_point():
            raise TypeError("wave must be a floating-point tensor")
        factor = 2.0 ** (float(n_semitones) / 12.0)
        length = wave.shape[-1]
        if length == 0:
            return wave
        positions = (torch.arange(length, device=wave.device, dtype=wave.dtype) * factor) % length
        left = positions.floor().long()
        right = (left + 1) % length
        fraction = positions - left.to(wave.dtype)
        return wave[..., left] + (wave[..., right] - wave[..., left]) * fraction
