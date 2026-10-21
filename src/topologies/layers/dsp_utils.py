import math

import torch
from torch import nn
import torch.nn.functional as F


def bandlimited_saw(phase, harmonics):
    orders = torch.arange(1, harmonics + 1, dtype=phase.dtype, device=phase.device)
    series = torch.sin(phase.unsqueeze(-1) * orders) / orders
    return (2 / math.pi) * series.sum(-1)


def bandlimited_square(phase, harmonics):
    orders = torch.arange(1, harmonics + 1, 2, dtype=phase.dtype, device=phase.device)
    series = torch.sin(phase.unsqueeze(-1) * orders) / orders
    return (4 / math.pi) * series.sum(-1)


def noise_exciter(noise, magnitudes):
    spectrum = torch.fft.rfft(noise)
    if spectrum.shape[-1] != magnitudes.shape[-1]:
        raise ValueError("magnitude bins must match the real FFT")
    return torch.fft.irfft(spectrum * magnitudes, n=noise.shape[-1])


def exponential_reverb(dry, decay):
    steps = torch.arange(dry.shape[-1], dtype=dry.dtype, device=dry.device)
    impulse = torch.exp(-steps / decay)
    kernel = impulse.flip(0).view(1, 1, -1)
    wet = F.conv1d(dry.view(1, 1, -1), kernel, padding=dry.shape[-1] - 1)
    return wet.view(-1)[: dry.shape[-1]]


def adsr(length, attack, decay, sustain, release):
    if min(attack, release) <= 0 or decay < 0 or attack + decay + release > length:
        raise ValueError("invalid envelope")
    envelope = torch.empty(length)
    for index in range(attack):
        envelope[index] = (index + 1) / attack
    for index in range(decay):
        envelope[attack + index] = 1 + (sustain - 1) * (index + 1) / decay
    hold = length - release
    envelope[attack + decay:hold] = sustain
    for index in range(release):
        envelope[hold + index] = sustain * (1 - (index + 1) / release)
    return envelope


def log_magnitude_loss(predicted, target, floor=1e-5):
    return (torch.log(predicted.abs() + floor) - torch.log(target.abs() + floor)).abs().mean()


class DSPUtils(nn.Module):
    component_id = "dsp-utils"
    PRESETS = {
        "tiny": {"harmonics": 4, "mode": "saw"},
        "base": {"harmonics": 8, "mode": "square"},
    }

    def __init__(self, harmonics=4, mode="saw", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"saw", "square", "reverb"}:
            raise ValueError("unknown dsp mode")
        if harmonics <= 0:
            raise ValueError("invalid harmonic count")
        self.harmonics = harmonics
        self.mode = mode

    def forward(self, signal):
        signal = torch.as_tensor(signal, dtype=torch.float32)
        if signal.ndim != 2:
            raise ValueError("signal must have shape (batch, time)")
        if self.mode == "reverb":
            return torch.stack([exponential_reverb(row, torch.tensor(4.0)) for row in signal])
        oscillator = bandlimited_square if self.mode == "square" else bandlimited_saw
        return oscillator(signal, self.harmonics)
