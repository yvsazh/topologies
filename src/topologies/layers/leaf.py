import math

import torch
from torch import nn
from torch.nn import functional as F


class LEAF(nn.Module):
    component_id = "leaf"

    def __init__(self, num_filters=4, kernel_size=8, stride=4):
        super().__init__()
        if num_filters < 1 or kernel_size < 2 or stride < 1:
            raise ValueError("configuration values must be positive and kernel_size at least 2")
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.center_frequencies = nn.Parameter(torch.linspace(0.08, 0.42, num_filters))
        initial_sigma = max(4 * math.sqrt(2 * math.log(2)), kernel_size / 3)
        self.log_sigmas = nn.Parameter(torch.full((num_filters,), math.log(initial_sigma)))
        self.log_pooling_width = nn.Parameter(torch.tensor(math.log(max(1.0, kernel_size / 3))))

    def _filters(self, x):
        device, dtype = x.device, x.dtype
        time = torch.arange(self.kernel_size, device=device, dtype=dtype) - (self.kernel_size - 1) / 2
        centers = self.center_frequencies.to(dtype=dtype).clamp(0.0, 0.5)
        min_sigma = 4 * math.sqrt(2 * math.log(2))
        max_sigma = 2 * self.kernel_size * math.sqrt(2 * math.log(2))
        sigmas = self.log_sigmas.to(dtype=dtype).clamp(
            min=math.log(min_sigma), max=math.log(max_sigma)
        ).exp()
        envelopes = torch.exp(-0.5 * (time[None, :] / sigmas[:, None]) ** 2)
        envelopes = envelopes / (math.sqrt(2 * math.pi) * sigmas[:, None])
        phase = 2 * math.pi * centers[:, None] * time[None, :]
        real = envelopes * torch.cos(phase)
        imag = envelopes * torch.sin(phase)
        return real.unsqueeze(1), imag.unsqueeze(1)

    def forward(self, waveform):
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("waveform must have shape (batch, time) or (batch, 1, time)")
        real_filter, imag_filter = self._filters(waveform)
        left = self.kernel_size // 2
        right = self.kernel_size - 1 - left
        padded = F.pad(waveform, (left, right))
        real = F.conv1d(padded, real_filter)
        imag = F.conv1d(padded, imag_filter)
        energy = real.square() + imag.square()
        pool_width = self.log_pooling_width.to(dtype=waveform.dtype).exp().clamp_min(0.1)
        positions = torch.arange(self.kernel_size, device=waveform.device, dtype=waveform.dtype) - (self.kernel_size - 1) / 2
        pooling = torch.exp(-0.5 * (positions / pool_width) ** 2)
        pooling = pooling / (math.sqrt(2 * math.pi) * pool_width)
        energy = F.conv1d(
            F.pad(energy, (left, right)),
            pooling.view(1, 1, -1).expand(self.num_filters, 1, -1),
            groups=self.num_filters,
        )
        return energy[:, :, ::self.stride]
