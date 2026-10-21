import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, mode="v15", widen=1, cardinality=2, scales=2):
        super().__init__()
        if mode not in {"v15", "v1d", "preact", "wide", "resnext", "se", "res2"}:
            raise ValueError("mode must be v15, v1d, preact, wide, resnext, se, or res2")
        if min(in_channels, out_channels, widen, cardinality, scales) <= 0:
            raise ValueError("invalid residual dimensions")
        inner = out_channels * widen
        if mode == "resnext" and inner % cardinality:
            raise ValueError("inner channels must be divisible by cardinality")
        if mode == "res2" and inner % scales:
            raise ValueError("inner channels must be divisible by scales")
        self.mode = mode
        self.stride = stride
        self.scales = scales
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inner, 1, bias=False)
        groups = cardinality if mode == "resnext" else 1
        conv2_in = inner // scales if mode == "res2" else inner
        conv2_groups = groups if mode != "res2" else 1
        self.conv2 = nn.Conv2d(conv2_in, conv2_in, 3, stride=stride, padding=1, groups=conv2_groups, bias=False)
        self.bn2 = nn.BatchNorm2d(inner if mode != "res2" else conv2_in)
        self.conv3 = nn.Conv2d(inner, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        hidden = max(out_channels // 4, 1)
        self.se_fc1 = nn.Linear(out_channels, hidden)
        self.se_fc2 = nn.Linear(hidden, out_channels)
        if mode == "v1d" and stride == 2:
            self.downsample = nn.Sequential(nn.AvgPool2d(2, stride=2), nn.Conv2d(in_channels, out_channels, 1, bias=False))
        elif stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        else:
            self.downsample = None

    def excite(self, values):
        pooled = values.mean(dim=(2, 3))
        gate = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(pooled))))
        return values * gate[:, :, None, None]

    def res2(self, values):
        width = values.shape[1] // self.scales
        parts = list(values.split(width, dim=1))
        if self.stride == 2:
            first = F.avg_pool2d(parts[0], 2, stride=2)
        else:
            first = parts[0]
        return torch.cat((first, self.conv2(parts[1])), dim=1)

    def forward(self, values):
        if self.mode == "preact":
            hidden = self.conv1(F.relu(self.bn1(values)))
            hidden = self.conv2(F.relu(self.bn2(hidden)))
            hidden = self.conv3(F.relu(self.bn3(hidden)))
        else:
            hidden = self.conv1(values)
            if self.mode == "res2":
                hidden = self.res2(hidden)
            else:
                hidden = self.conv2(hidden)
            hidden = self.conv3(hidden)
            if self.mode == "se":
                hidden = self.excite(hidden)
        shortcut = values if self.downsample is None else self.downsample(values)
        return hidden + shortcut


class ResidualVariants(nn.Module):
    architecture_id = "resnet-variants"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 1, "mode": "v15"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 2, "mode": "se"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=1, mode="v15", widen=1, cardinality=2, scales=2, dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid residual network dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        channels = 8 * width
        self.stem = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.block = ResidualBlock(channels, channels, stride=2, mode=mode, widen=widen, cardinality=cardinality, scales=scales)
        self.block2 = ResidualBlock(channels, channels, stride=1, mode=mode, widen=widen, cardinality=cardinality, scales=scales)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        hidden = self.block2(self.block(F.relu(self.stem(pixels))))
        return self.fc(hidden.mean(dim=(2, 3)))
