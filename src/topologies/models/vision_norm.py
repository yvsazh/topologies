import torch
from torch import nn
import torch.nn.functional as F


def group_norm(values, eps=1e-5):
    mean = values.mean()
    var = ((values - mean) ** 2).mean()
    return (values - mean) / torch.sqrt(var + eps)


def evo_norm(values):
    return values * torch.sigmoid(values)


def filter_response_norm(values, eps=1e-6):
    return values / torch.sqrt((values ** 2).mean() + eps)


def drop_block(values, row, col):
    out = values.clone()
    out[int(row), int(col)] = 0
    return out


def mixup(left, right, lam):
    return lam * left + (1 - lam) * right


def cutmix(base, donor, row0, col0, row1, col1):
    out = base.clone()
    out[row0:row1, col0:col1] = donor[row0:row1, col0:col1]
    return out


def label_smooth(index, classes, epsilon):
    smooth = torch.full((classes,), epsilon / classes)
    smooth[int(index)] = smooth[int(index)] + (1 - epsilon)
    return smooth


class VisionNorm(nn.Module):
    architecture_id = "vision-norm"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "group"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "frn"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="group", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"group", "evo", "frn", "dropblock", "mixup", "cutmix", "smooth"}:
            raise ValueError("unknown normalization mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid normalization dimensions")
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
        if self.mode == "frn":
            scale = torch.sqrt((hidden ** 2).mean(dim=(2, 3), keepdim=True) + 1e-6)
            hidden = hidden / scale
        pooled = self.proj(hidden.mean(dim=(2, 3)))
        return self.fc(pooled)
