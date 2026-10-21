import torch
from torch import nn


class TemporalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.activation(x + residual)


class TCResNet(nn.Module):
    architecture_id = "tc-resnet"
    def __init__(self, input_channels=40, num_classes=12, channels=(16, 24, 32, 48), strides=None, num_blocks=1, dropout=0.0):
        super().__init__()
        channels = tuple(channels)
        if input_channels <= 0 or num_classes <= 0 or not channels or any(c <= 0 for c in channels):
            raise ValueError("channel counts and num_classes must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if strides is None:
            strides = (1,) + (2,) * (len(channels) - 1)
        strides = tuple(strides)
        if len(strides) != len(channels) or any(s not in (1, 2) for s in strides):
            raise ValueError("strides must contain one value of 1 or 2 per channel stage")
        self.input_channels = input_channels
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        blocks = []
        in_channels = channels[0]
        for stage_channels, stage_stride in zip(channels, strides):
            for block_index in range(num_blocks):
                stride = stage_stride if block_index == 0 else 1
                blocks.append(TemporalResidualBlock(in_channels, stage_channels, stride, 9))
                in_channels = stage_channels
        self.blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.input_channels:
            raise ValueError(f"expected input shaped (batch, {self.input_channels}, time)")
        x = self.blocks(self.stem(x))
        x = x.mean(dim=-1)
        return self.classifier(self.dropout(x))
