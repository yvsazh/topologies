import torch
from torch import nn
import torch.nn.functional as F


def bilinear_upsample(values, size):
    return F.interpolate(values, size=size, mode="bilinear", align_corners=False)


def pyramid_pool(values):
    return values.mean(dim=(2, 3), keepdim=True)


def fuse_resolutions(high, low):
    restored = F.interpolate(low, size=high.shape[-2:], mode="nearest")
    return high + restored


def mask_from_query(pixels, query):
    return pixels @ query


def channel_mix(tokens, weight):
    return tokens @ weight


def masked_attention(logits, foreground):
    blocked = logits.masked_fill(foreground == 0, -1e9)
    return torch.softmax(blocked, dim=-1)


class Segmenter(nn.Module):
    architecture_id = "segment"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "fcn"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "deeplab"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="fcn", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"fcn", "deeplab", "psp", "hrnet", "segformer", "setr", "maskformer", "mask2former"}:
            raise ValueError("unknown segmentation mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid segmentation dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        self.stem = nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False)
        self.atrous = nn.Conv2d(width, width, 3, padding=2, dilation=2, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.fc = nn.Linear(width, num_classes)
        self.query = nn.Parameter(torch.zeros(width))

    def forward(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        hidden = F.relu(self.stem(pixels))
        if self.mode == "fcn":
            hidden = bilinear_upsample(hidden, (32, 32))
        elif self.mode == "deeplab":
            hidden = self.atrous(hidden)
        elif self.mode == "psp":
            hidden = hidden + pyramid_pool(hidden)
        elif self.mode == "hrnet":
            low = F.avg_pool2d(hidden, 2)
            hidden = fuse_resolutions(hidden, low)
        elif self.mode == "maskformer":
            flat = hidden.mean(dim=(2, 3))
            _mask = mask_from_query(flat, self.query)
        pooled = self.proj(hidden.mean(dim=(2, 3)))
        return self.fc(pooled)
