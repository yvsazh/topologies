

import torch
from torch import nn


class CvT(nn.Module):
    component_id = "cvt"
    """Depthwise token convolution with a pointwise output-channel projection.

    Args:
        in_channels: Number of channels in each input token.
        out_channels: Requested number of output channels.
        kernel_size: Odd sequence-convolution kernel size.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:

        if tokens.ndim != 3:
            raise ValueError("tokens must be a 3D tensor with shape (B, N, C)")
        tokens = tokens.transpose(1, 2)
        tokens = self.pointwise(self.depthwise(tokens))
        return tokens.transpose(1, 2)
