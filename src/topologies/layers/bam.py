import torch
from torch import nn


class BAM(nn.Module):
    component_id = "bam"
    PRESETS = {
        "tiny": {"channels": 16, "reduction": 16, "dilation": 4},
        "small": {"channels": 64, "reduction": 16, "dilation": 4},
    }

    def __init__(self, channels, reduction=16, dilation=4):
        super().__init__()
        for name, value in (("channels", channels), ("reduction", reduction), ("dilation", dilation)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if channels % reduction != 0:
            raise ValueError("channels must be divisible by reduction")
        hidden_channels = channels // reduction
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.channel_reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.channel_expand = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.channel_norm = nn.BatchNorm2d(channels)
        self.spatial_reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.spatial_conv1 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.spatial_conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.spatial_expand = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.spatial_norm = nn.BatchNorm2d(1)

    def _normalize(self, layer, x):
        if self.training and x.numel() == x.shape[1]:
            return torch.nn.functional.batch_norm(
                x,
                layer.running_mean,
                layer.running_var,
                layer.weight,
                layer.bias,
                training=False,
                momentum=layer.momentum,
                eps=layer.eps,
            )
        return layer(x)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("x must be a rank-4 NCHW tensor")
        if x.shape[1] != self.channels:
            raise ValueError(f"x must have {self.channels} channels")
        if x.shape[0] <= 0 or x.shape[2] <= 0 or x.shape[3] <= 0:
            raise ValueError("x batch and spatial dimensions must be positive")
        channel_logits = self._normalize(
            self.channel_norm, self.channel_expand(self.channel_reduce(self.pool(x)))
        )
        spatial_logits = self._normalize(
            self.spatial_norm,
            self.spatial_expand(self.spatial_conv2(self.spatial_conv1(self.spatial_reduce(x))))
        )
        attention = torch.sigmoid(channel_logits + spatial_logits)
        return x + x * attention
