import torch
from torch import nn
import torch.nn.functional as F


def hungarian_pair(cost):
    direct = cost[0, 0] + cost[1, 1]
    swapped = cost[0, 1] + cost[1, 0]
    if direct <= swapped:
        return direct, (0, 1)
    return swapped, (1, 0)


def deformable_sample(values, reference, offset):
    point = reference + offset
    left = int(torch.floor(point).item())
    right = min(left + 1, values.shape[-1] - 1)
    weight = point - left
    return values[left] * (1 - weight) + values[right] * weight


def denoise_l1(noisy, clean):
    return (noisy - clean).abs().mean()


def hybrid_loss(match_cost, many_cost):
    return match_cost + many_cost


def select_confident(scores, count):
    return torch.topk(scores, count).indices


def open_vocab_scores(image, texts):
    image = F.normalize(image, dim=-1)
    texts = F.normalize(texts, dim=-1)
    return image @ texts.transpose(-1, -2)


class SetDetector(nn.Module):
    architecture_id = "set-detect"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "detr"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "deformable"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="detr", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"detr", "deformable", "dino", "hybrid", "rtdetr"}:
            raise ValueError("unknown set detector mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid set detector dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        self.stem = nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False)
        self.query = nn.Parameter(torch.zeros(1, 2, width))
        self.proj = nn.Linear(width, width, bias=False)
        self.fc = nn.Linear(width, num_classes)

    def forward(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        hidden = F.relu(self.stem(pixels)).mean(dim=(2, 3)).unsqueeze(1)
        tokens = hidden + self.query
        return self.fc(self.proj(tokens.mean(dim=1)))
