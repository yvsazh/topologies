import math

import torch
from torch import nn


def hz_to_mel(hz):
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def triangle_weights(left, center, right, n_freqs):
    weights = torch.zeros(n_freqs)
    for index in range(left, center):
        if 0 <= index < n_freqs and center > left:
            weights[index] = (index - left) / (center - left)
    for index in range(center, right):
        if 0 <= index < n_freqs and right > center:
            weights[index] = (right - index) / (right - center)
    return weights


def mel_filterbank(n_freqs, n_mels, sample_rate, n_fft):
    with torch.device("cpu"):
        start = hz_to_mel(torch.tensor(0.0))
        end = hz_to_mel(torch.tensor(sample_rate / 2))
        points = torch.linspace(start, end, n_mels + 2)
        hz = 700.0 * (torch.pow(10.0, points / 2595.0) - 1.0)
        bins = torch.floor((n_fft + 1) * hz / sample_rate).to(torch.long).tolist()
    bank = []
    for index in range(n_mels):
        bank.append(triangle_weights(int(bins[index]), int(bins[index + 1]), int(bins[index + 2]), n_freqs))
    return torch.stack(bank)


def log_mel(power, filterbank, floor=1e-10):
    return torch.log(torch.clamp(power @ filterbank.transpose(0, 1), min=floor))


def mfcc_from_logmel(values, n_coeffs):
    bins = values.shape[-1]
    time = torch.arange(bins, dtype=values.dtype, device=values.device)
    order = torch.arange(n_coeffs, dtype=values.dtype, device=values.device)
    basis = torch.cos(math.pi / bins * (time + 0.5) * order.unsqueeze(1))
    return values @ basis.transpose(0, 1)


def delta_features(features, width=2):
    if features.ndim == 1:
        padded = nn.functional.pad(features, (width, width))
        total = torch.zeros_like(features)
        for step in range(1, width + 1):
            total = total + step * (
                padded[width + step:width + step + features.shape[0]]
                - padded[width - step:width - step + features.shape[0]]
            )
    else:
        padded = nn.functional.pad(features, (0, 0, width, width))
        total = torch.zeros_like(features)
        for step in range(1, width + 1):
            total = total + step * (
                padded[..., width + step:width + step + features.shape[-2], :]
                - padded[..., width - step:width - step + features.shape[-2], :]
            )
    denom = 2.0 * sum(step * step for step in range(1, width + 1))
    return total / denom


def pcen(energy, smoother, alpha, offset, root, eps=1e-6):
    compressed = energy / torch.pow(eps + smoother, alpha) + offset
    return torch.pow(compressed, root) - offset ** root


def sinc_lowpass(cutoff, kernel_size):
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel size must be a positive odd integer")
    half = kernel_size // 2
    taps = torch.arange(-half, half + 1, dtype=torch.float32)
    window_index = torch.arange(kernel_size, dtype=torch.float32)
    window = 0.5 - 0.5 * torch.cos(2 * math.pi * window_index / (kernel_size - 1))
    center = torch.full_like(taps, 2.0 * cutoff)
    lobe = torch.sin(2 * math.pi * cutoff * taps) / (math.pi * taps)
    impulse = torch.where(taps == 0, center, lobe)
    return impulse * window


def cqt_frequencies(f_min, n_bins, bins_per_octave):
    return f_min * torch.pow(2.0, torch.arange(n_bins, dtype=torch.float32) / bins_per_octave)


def gabor_center(sigma):
    return torch.exp(torch.tensor(0.0) / (2 * sigma * sigma))


class AudioFrontend(nn.Module):
    component_id = "audio-frontend"
    PRESETS = {
        "tiny": {"n_fft": 16, "hop": 8, "n_mels": 4, "sample_rate": 16000, "mode": "logmel"},
        "base": {"n_fft": 16, "hop": 8, "n_mels": 8, "sample_rate": 16000, "mode": "mfcc"},
    }

    def __init__(self, n_fft=16, hop=8, n_mels=4, sample_rate=16000, mode="logmel", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"logmel", "mfcc", "pcen", "sinc"}:
            raise ValueError("unknown frontend mode")
        if min(n_fft, hop, n_mels, sample_rate) <= 0 or n_fft % 2:
            raise ValueError("invalid frontend dimensions")
        self.n_fft = n_fft
        self.hop = hop
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mode = mode
        n_freqs = n_fft // 2 + 1
        bank = mel_filterbank(n_freqs, n_mels, sample_rate, n_fft)
        self.register_buffer("filterbank", bank)
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

    def forward(self, waveform):
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape (batch, time)")
        if self.mode == "sinc":
            kernel = sinc_lowpass(0.25, 5).to(waveform.device)
            return nn.functional.conv1d(waveform.unsqueeze(1), kernel.view(1, 1, -1), padding=2)
        spectrum = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop, win_length=self.n_fft,
            window=self.window, center=False, return_complex=True,
        )
        power = spectrum.abs().square().transpose(1, 2)
        if self.mode == "pcen":
            mel = torch.clamp(power @ self.filterbank.transpose(0, 1), min=1e-10)
            return pcen(mel, mel, 0.98, 2.0, 0.5)
        values = log_mel(power, self.filterbank)
        if self.mode == "mfcc":
            return mfcc_from_logmel(values, self.n_mels)
        return values
