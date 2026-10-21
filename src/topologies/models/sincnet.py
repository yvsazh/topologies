import torch
from torch import nn
from torch.nn import functional as F


class SincNet(nn.Module):
    architecture_id = "sincnet"
    def __init__(self, out_channels=4, kernel_size=8, sample_rate=16000):
        super().__init__()
        if out_channels < 1:
            raise ValueError("out_channels must be positive")
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_band = min(sample_rate / 4, 50.0)
        self.min_low = 1.0
        max_high = sample_rate / 2 - self.min_band
        low = torch.linspace(80.0, max(self.min_low + 1, max_high * 0.45), out_channels).clamp(max=max_high - self.min_band)
        band = torch.full((out_channels,), min(max_high / (out_channels + 1), max(100.0, self.min_band + 1)))
        band = torch.minimum(band, max_high - low).clamp_min(1.0)
        self.low_hz_ = nn.Parameter(low)
        self.band_hz_ = nn.Parameter(band)

    def cutoff_frequencies(self):
        low = self.min_low + F.softplus(self.low_hz_ - self.min_low)
        high = low + self.min_band + F.softplus(self.band_hz_ - self.min_band)
        nyquist = self.sample_rate / 2
        high = high.clamp(max=nyquist)
        low = torch.minimum(low, high - self.min_band)
        return low, high

    def filters(self):
        low, high = self.cutoff_frequencies()
        half = self.kernel_size // 2
        offsets = torch.arange(-half, self.kernel_size - half, device=low.device, dtype=low.dtype)
        window = torch.hamming_window(self.kernel_size, periodic=False, device=low.device, dtype=low.dtype)
        t = offsets.unsqueeze(0)
        low_norm = (low / self.sample_rate).unsqueeze(1)
        high_norm = (high / self.sample_rate).unsqueeze(1)
        high_kernel = 2 * high_norm * torch.sinc(2 * high_norm * t)
        low_kernel = 2 * low_norm * torch.sinc(2 * low_norm * t)
        bandpass = (high_kernel - low_kernel) * window
        return bandpass.unsqueeze(1)

    def forward(self, waveform):
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must have shape (batch, length) or (batch, 1, length)")
        return F.conv1d(waveform, self.filters(), padding=self.kernel_size // 2)
