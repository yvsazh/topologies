import torch
from torch import nn
import torch.nn.functional as F


def trilinear_upsample(values, size):
    return F.interpolate(values, size=size, mode="trilinear", align_corners=False)


def dice_loss(pred, target, eps=1e-6):
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + eps) / (pred.sum() + target.sum() + eps)


def pool_stages(size, stop=4):
    stages = 0
    remaining = int(size)
    while remaining > int(stop):
        remaining //= 2
        stages += 1
    return stages


class VolumeSeg(nn.Module):
    architecture_id = "volume-seg"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "unet3d"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "vnet"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="unet3d", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"unet3d", "vnet", "nnunet"}:
            raise ValueError("unknown volume mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid volume dimensions")
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
