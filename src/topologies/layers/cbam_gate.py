import torch
from torch import nn


class CBAMGate(nn.Module):
    component_id = "cbam-gate"

    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if isinstance(reduction_ratio, bool) or not isinstance(reduction_ratio, int) or reduction_ratio <= 0:
            raise ValueError("reduction_ratio must be a positive integer")
        hidden_channels = channels // reduction_ratio
        if hidden_channels < 1:
            raise ValueError("channels // reduction_ratio must be at least 1")
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.channel_fc1 = nn.Linear(channels, hidden_channels)
        self.channel_fc2 = nn.Linear(hidden_channels, channels)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.spatial_bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True)

    def _channel_mlp(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.channel_fc2(torch.relu(self.channel_fc1(descriptor)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("input must be a four-dimensional NCHW tensor")
        batch, channels, height, width = x.shape
        if batch <= 0 or height <= 0 or width <= 0:
            raise ValueError("input batch, height, and width dimensions must be positive")
        if channels != self.channels:
            raise ValueError(f"input must have {self.channels} channels")
        average = x.mean(dim=(2, 3))
        maximum = x.amax(dim=(2, 3))
        channel_map = torch.sigmoid(self._channel_mlp(average) + self._channel_mlp(maximum))
        refined = x * channel_map.unsqueeze(-1).unsqueeze(-1)
        spatial_descriptors = torch.cat(
            (refined.mean(dim=1, keepdim=True), refined.amax(dim=1, keepdim=True)), dim=1
        )
        spatial_map = torch.sigmoid(self.spatial_bn(self.spatial_conv(spatial_descriptors)))
        return refined * spatial_map
