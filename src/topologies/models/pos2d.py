import torch
from torch import nn
import torch.nn.functional as F


def relative_bias(query_index, key_index):
    return key_index - query_index


def rope_pair(x, y, cos, sin):
    return x * cos - y * sin, x * sin + y * cos


def conditional_position(tokens, left_weight):
    padded = F.pad(tokens, (1, 1, 1, 1))
    left = padded[..., 1:-1, :-2]
    return tokens + left_weight * left


class Pos2D(nn.Module):
    architecture_id = "pos2d"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "relative"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "rope"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="relative", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"relative", "rope", "cpvt"}:
            raise ValueError("unknown position mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid position dimensions")
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
        pooled = self.proj(hidden.mean(dim=(2, 3)))
        return self.fc(pooled)
