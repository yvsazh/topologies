import math

import torch
from torch import nn


class CREPE(nn.Module):
    component_id = "crepe"
    cents_per_bin = 20.0
    reference_frequency = 10.0
    lowest_frequency = 31.7

    def __init__(self, num_bins=360):
        super().__init__()
        if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 1:
            raise ValueError("num_bins must be a positive integer")
        self.num_bins = num_bins
        lowest_cents = 1200.0 * math.log2(self.lowest_frequency / self.reference_frequency)
        self.register_buffer(
            "bin_cents",
            lowest_cents + self.cents_per_bin * torch.arange(num_bins, dtype=torch.float64),
        )

    def decode_cents(self, activations):
        if not torch.is_tensor(activations):
            raise TypeError("activations must be a tensor")
        if not activations.is_floating_point():
            raise TypeError("activations must have a floating-point dtype")
        if activations.ndim == 0 or activations.shape[-1] != self.num_bins:
            raise ValueError(f"expected activations with a final dimension of {self.num_bins}")
        if not bool(torch.isfinite(activations).all()):
            raise ValueError("activations must be finite")
        if bool((activations < 0).any()):
            raise ValueError("activations must be nonnegative")
        weight_sum = activations.sum(dim=-1)
        if bool((weight_sum <= 0).any()):
            raise ValueError("each activation vector must have positive total weight")
        centers = self.bin_cents.to(device=activations.device, dtype=activations.dtype)
        return (activations * centers).sum(dim=-1) / weight_sum

    def forward(self, activations):
        cents = self.decode_cents(activations)
        return self.reference_frequency * torch.pow(2.0, cents / 1200.0)
