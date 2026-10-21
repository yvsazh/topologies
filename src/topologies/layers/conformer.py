import math

import torch
from torch import nn
from torch.nn import functional as F


class ConformerConvModule(nn.Module):
    component_id = "conformer"

    def __init__(self, channels, kernel_size=32, dropout_prob=0.1):
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if not isinstance(kernel_size, int) or isinstance(kernel_size, bool) or kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer")
        if isinstance(dropout_prob, (int, float)) and not isinstance(dropout_prob, bool):
            if not 0.0 <= dropout_prob <= 1.0:
                raise ValueError("dropout_prob must be a finite number in [0, 1]")
            dropout_prob = float(dropout_prob)
        else:
            dropout_prob = math.nan
        if not math.isfinite(dropout_prob) or not 0.0 <= dropout_prob <= 1.0:
            raise ValueError("dropout_prob must be a finite number in [0, 1]")
        self.channels = channels
        self.kernel_size = kernel_size
        self.layer_norm = nn.LayerNorm(channels, eps=1e-6)
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, bias=True)
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(channels, eps=0.001, momentum=0.001)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("input must be a tensor with shape [batch, time, channels]")
        if x.shape[-1] != self.channels:
            raise ValueError(f"input's final dimension must equal channels ({self.channels})")
        if x.shape[0] <= 0 or x.shape[1] <= 0:
            raise ValueError("input batch and time dimensions must be positive")
        normalized = self.layer_norm(x)
        projected = self.pointwise_conv1(normalized.transpose(1, 2)).transpose(1, 2)
        gate, values = projected.chunk(2, dim=-1)
        gated = values * torch.sigmoid(gate)
        temporal = gated.transpose(1, 2)
        padding_left = (self.kernel_size - 1) // 2
        padding_right = self.kernel_size // 2
        temporal = F.pad(temporal, (padding_left, padding_right))
        temporal = self.depthwise_conv(temporal)
        temporal = self.batch_norm(temporal)
        temporal = temporal * torch.sigmoid(temporal)
        projected = self.pointwise_conv2(temporal).transpose(1, 2)
        return x + self.dropout(projected)
