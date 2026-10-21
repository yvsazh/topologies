import torch
from torch import nn


class InstanceNormBlock(nn.Module):
    component_id = "instancenorm-block"

    def __init__(self, channels=1, eps=1e-5):
        super().__init__()
        if channels <= 0 or eps <= 0:
            raise ValueError("channels and eps must be positive")
        self.channels = channels
        self.eps = eps

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels or x.shape[-2] * x.shape[-1] == 0:
            raise ValueError("x must have shape (batch, channels, height, width) with nonempty spatial axes")
        mean = x.mean(dim=(-2, -1), keepdim=True)
        variance = (x - mean).square().mean(dim=(-2, -1), keepdim=True)
        return (x - mean) / torch.sqrt(variance + self.eps)
