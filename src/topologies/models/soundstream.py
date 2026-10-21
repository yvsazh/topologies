import torch
from torch import nn


class SoundStream(nn.Module):
    architecture_id = "soundstream"
    def __init__(self, channels=8, num_codebooks=2, codebook_size=4):
        super().__init__()
        if channels < 1 or num_codebooks < 1 or codebook_size < 1:
            raise ValueError("channels, num_codebooks, and codebook_size must be positive")
        self.channels = channels
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebooks = nn.Parameter(torch.randn(num_codebooks, codebook_size, channels) / channels**0.5)

    def quantize(self, encoded):
        if encoded.ndim < 2 or encoded.shape[-1] != self.channels:
            raise ValueError(f"encoded frames must have final dimension {self.channels}")
        residual = encoded
        total = torch.zeros_like(encoded)
        residuals = []
        for index in range(self.num_codebooks):
            residuals.append(residual)
            codebook = self.codebooks[index]
            distances = residual.square().sum(dim=-1, keepdim=True) - 2 * residual @ codebook.t() + codebook.square().sum(dim=-1)
            codes = codebook[distances.argmin(dim=-1)]
            total = total + codes
            residual = residual - codes
        quantized = total + (encoded - encoded.detach())
        return quantized, residuals

    def forward(self, encoded):
        return self.quantize(encoded)[0]
