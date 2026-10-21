import torch
from torch import nn


class S3DSeparableConv3d(nn.Module):
    component_id = "s3d"

    def __init__(self, in_channels=3, out_channels=16, spatial_kernel=3, temporal_kernel=3):
        super().__init__()
        if min(in_channels, out_channels, spatial_kernel, temporal_kernel) <= 0:
            raise ValueError("channel counts and kernel sizes must be positive")
        if spatial_kernel % 2 == 0 or temporal_kernel % 2 == 0:
            raise ValueError("kernel sizes must be odd to preserve dimensions")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, spatial_kernel, spatial_kernel),
            padding=(0, spatial_kernel // 2, spatial_kernel // 2),
        )
        self.temporal_conv = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(temporal_kernel, 1, 1),
            padding=(temporal_kernel // 2, 0, 0),
        )

    def forward(self, clip):
        clip = torch.as_tensor(clip)
        if clip.ndim != 5 or clip.shape[1] != self.in_channels:
            raise ValueError("clip must have shape (batch, channels, frames, height, width)")
        return self.temporal_conv(self.spatial_conv(clip))
