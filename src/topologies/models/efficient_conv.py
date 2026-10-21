import torch
from torch import nn
import torch.nn.functional as F


def hswish(values):
    return values * F.relu6(values + 3.0) / 6.0


def channel_shuffle(values, groups):
    batch, channels, height, width = values.shape
    values = values.reshape(batch, groups, channels // groups, height, width)
    values = values.transpose(1, 2).contiguous()
    return values.reshape(batch, channels, height, width)


def regnet_width(index, w0, wa, divisor=8):
    raw = w0 + wa * index
    return max(divisor, int(round(raw / divisor)) * divisor)


def standardized_weight(weight):
    mean = weight.mean(dim=(1, 2, 3), keepdim=True)
    variance = weight.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
    return (weight - mean) / torch.sqrt(variance + 1e-5)


class EfficientBlock(nn.Module):
    def __init__(self, channels, mode, expand=2):
        super().__init__()
        if mode not in {"mobilenet-v1", "mobilenet-v3", "shufflenet-v1", "shufflenet-v2", "ghost", "efficientnet", "efficientnet-v2", "regnet", "nfnet"}:
            raise ValueError("unknown efficient convolution mode")
        if channels % 2:
            raise ValueError("channels must be even")
        self.mode = mode
        self.channels = channels
        hidden = channels * expand
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.half_depthwise = nn.Conv2d(channels // 2, channels // 2, 3, padding=1, groups=channels // 2, bias=False)
        self.half_pointwise = nn.Conv2d(channels // 2, channels // 2, 1, bias=False)
        self.expand = nn.Conv2d(channels, hidden, 1, bias=False)
        self.expanded_depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.project = nn.Conv2d(hidden, channels, 1, bias=False)
        self.fused = nn.Conv2d(channels, hidden, 3, padding=1, bias=False)
        self.primary = nn.Conv2d(channels, channels // 2, 1, bias=False)
        self.cheap = nn.Conv2d(channels // 2, channels // 2, 3, padding=1, groups=channels // 2, bias=False)
        self.nf_weight = nn.Parameter(torch.randn(channels, channels, 1, 1))

    def forward(self, values):
        if self.mode == "mobilenet-v1":
            return self.pointwise(self.depthwise(values))
        if self.mode == "mobilenet-v3":
            return hswish(self.pointwise(self.depthwise(values)))
        if self.mode == "shufflenet-v1":
            return channel_shuffle(self.pointwise(self.depthwise(values)), 2)
        if self.mode == "shufflenet-v2":
            first, second = values.chunk(2, dim=1)
            second = self.half_pointwise(self.half_depthwise(second))
            return channel_shuffle(torch.cat((first, second), dim=1), 2)
        if self.mode == "ghost":
            primary = self.primary(values)
            return torch.cat((primary, self.cheap(primary)), dim=1)
        if self.mode == "efficientnet-v2":
            return self.project(self.fused(values))
        if self.mode == "nfnet":
            weight = standardized_weight(self.nf_weight)
            return F.conv2d(values, weight)
        if self.mode == "regnet":
            return self.pointwise(self.depthwise(values))
        return self.project(self.expanded_depthwise(self.expand(values)))


class EfficientConv(nn.Module):
    architecture_id = "efficient-conv"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 4, "mode": "mobilenet-v1"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "efficientnet"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=4, mode="mobilenet-v1", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if min(in_channels, num_classes, width) <= 0 or width % 2:
            raise ValueError("invalid efficient convolution dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        self.stem = nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False)
        self.block = EfficientBlock(width, mode)
        self.fc = nn.Linear(width, num_classes)

    def forward(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        hidden = self.block(F.relu(self.stem(pixels)))
        return self.fc(hidden.mean(dim=(2, 3)))
