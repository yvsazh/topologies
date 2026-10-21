import torch
from torch import nn
import torch.nn.functional as F


def channel_squeeze(values):
    return values.mean(dim=(-2, -1))


def coordinate_pool(values):
    return values.mean(dim=-1), values.mean(dim=-2)


def bilinear_sample(left, right, weight):
    return (1 - weight) * left + weight * right


def axis_pool(values):
    return torch.stack((values.max(), values.mean()))


def cross_covariance(tokens):
    return tokens.transpose(0, 1) @ tokens / tokens.shape[0]


def focal_modulate(query, context):
    return query * context


def bottleneck_gate(values, channel, spatial):
    return values * channel * spatial


class Attn2D(nn.Module):
    architecture_id = "attn2d"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "cbam"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "coord"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="cbam", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"cbam", "bam", "triplet", "coord", "deformable", "xcit", "focal"}:
            raise ValueError("unknown attention mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid attention dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        self.stem = nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.fc = nn.Linear(width, num_classes)

    def forward(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        hidden = F.relu(self.stem(pixels))
        if self.mode == "cbam":
            scale = channel_squeeze(hidden).reshape(hidden.shape[0], hidden.shape[1], 1, 1)
            hidden = hidden * torch.sigmoid(scale)
        pooled = self.proj(hidden.mean(dim=(2, 3)))
        return self.fc(pooled)
