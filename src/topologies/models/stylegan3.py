import torch
from torch import nn
from torch.nn import functional as F


class StyleGAN3(nn.Module):
    architecture_id = "stylegan-3"
    def __init__(self, channels=4, kernel_size=3):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.channels = channels
        self.projection = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        base = torch.tensor([1.0, 2.0, 1.0])
        filt = torch.outer(base, base)
        filt = filt / filt.sum()
        self.register_buffer("low_pass", filt.expand(channels, 1, 3, 3).contiguous())

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("input must have shape (batch, channels, height, width)")
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.conv2d(x, self.low_pass, padding=1, groups=self.channels)
        return self.projection(x)
