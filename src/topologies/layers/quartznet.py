from __future__ import annotations

import torch
from torch import nn


class TimeChannelSeparableConv1d(nn.Module):
    component_id = "quartznet"

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 8,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, kernel_size, stride, dilation) <= 0:
            raise ValueError("channel counts, kernel size, stride, and dilation must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-length symmetric padding")
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
            groups=in_channels,
        )
        self.depthwise_norm = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pointwise_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != self.depthwise.in_channels:
            raise ValueError("inputs must have shape (batch, in_channels, time)")
        value = self.activation(self.depthwise_norm(self.depthwise(inputs)))
        return self.activation(self.pointwise_norm(self.pointwise(value)))
