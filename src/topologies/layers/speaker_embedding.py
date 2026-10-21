import torch
from torch import nn


class XVectorStatsPool(nn.Module):
    component_id = "speaker-embedding"
    """Concatenate temporal mean and standard deviation for x-vector frames."""

    def __init__(self, input_dim):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = input_dim

    def forward(self, frames):
        if frames.ndim != 3 or frames.shape[-1] != self.input_dim:
            raise ValueError("frames must have shape (batch, time, input_dim)")
        if frames.shape[1] == 0:
            raise ValueError("time dimension must be non-empty")
        mean = frames.mean(dim=1)
        std = frames.var(dim=1, unbiased=False).clamp_min(0).sqrt()
        return torch.cat((mean, std), dim=-1)
