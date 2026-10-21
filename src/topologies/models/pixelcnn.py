import torch
from torch import nn
from torch.nn import functional as F


class PixelCNN(nn.Module):
    architecture_id = "pixelcnn"
    def __init__(self, channels=4, kernel_size=3):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        mask = torch.ones_like(self.conv.weight)
        center = kernel_size // 2
        mask[:, :, center, center:] = 0
        mask[:, :, center + 1:, :] = 0
        self.register_buffer("mask", mask)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("input must have shape (batch, channels, height, width)")
        return F.conv2d(x, self.conv.weight * self.mask, self.conv.bias, padding=self.kernel_size // 2)
