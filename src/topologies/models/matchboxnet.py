import torch
from torch import nn


class MatchboxNet(nn.Module):
    architecture_id = "matchboxnet"
    def __init__(self, channels=8, kernel_size=3, dropout=0.1):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1] != self.depthwise.in_channels:
            raise ValueError("inputs must have shape (batch, channels, time)")
        residual = inputs
        outputs = self.depthwise(inputs)
        outputs = self.pointwise(outputs)
        outputs = self.norm(outputs)
        outputs = self.dropout(outputs)
        return self.activation(outputs + residual)
