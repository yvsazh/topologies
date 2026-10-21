import math

import torch
from torch import nn


class FITS(nn.Module):
    architecture_id = "fits"
    def __init__(self, forecast_length, low_frequency_bins=3):
        super().__init__()
        if forecast_length <= 0 or low_frequency_bins <= 0:
            raise ValueError("forecast_length and low_frequency_bins must be positive")
        self.forecast_length = forecast_length
        self.low_frequency_bins = low_frequency_bins
        output_bins = forecast_length // 2 + 1
        initial_weight = torch.zeros(output_bins, low_frequency_bins, dtype=torch.cfloat)
        diagonal = min(output_bins, low_frequency_bins)
        initial_weight[torch.arange(diagonal), torch.arange(diagonal)] = 1 / math.sqrt(low_frequency_bins)
        self.frequency_weight = nn.Parameter(initial_weight)
        self.frequency_bias = nn.Parameter(torch.zeros(output_bins, dtype=torch.cfloat))

    def forward(self, values):
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values, dtype=torch.float32)
        if values.ndim < 1 or values.shape[-1] == 0:
            raise ValueError("values must have a non-empty time dimension")
        spectrum = torch.fft.rfft(values, dim=-1)
        keep = min(self.low_frequency_bins, spectrum.shape[-1])
        spectrum = spectrum[..., :keep]
        target_bins = self.forecast_length // 2 + 1
        resized = torch.einsum(
            "...i,oi->...o", spectrum, self.frequency_weight[:target_bins, :keep]
        ) + self.frequency_bias[:target_bins]
        return torch.fft.irfft(resized, n=self.forecast_length, dim=-1)
