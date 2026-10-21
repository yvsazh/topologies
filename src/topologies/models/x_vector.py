import torch
from torch import nn


class XVector(nn.Module):
    architecture_id = "x-vector"

    def __init__(self, input_channels=8, **kwargs):
        super().__init__()
        if "in_channels" in kwargs:
            input_channels = kwargs.pop("in_channels")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs)}")
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        self.input_channels = input_channels

    def forward(self, frames):
        if frames.ndim != 3:
            raise ValueError("frames must have shape (batch, time, channels)")
        if frames.shape[1] == 0:
            raise ValueError("frames must contain at least one time step")
        if frames.shape[2] != self.input_channels:
            raise ValueError(f"expected {self.input_channels} input channels, got {frames.shape[2]}")
        mean = frames.mean(dim=1)
        standard_deviation = frames.var(dim=1, unbiased=False).clamp_min(0).sqrt()
        return torch.cat((mean, standard_deviation), dim=-1)
