import torch
from torch import nn
import torch.nn.functional as F


def same_location(query, key, tokens_per_frame):
    return (int(query) % int(tokens_per_frame)) == (int(key) % int(tokens_per_frame))


def tube_mean(clip):
    return clip.mean(dim=0)


def visible_index(mask):
    return [index for index, value in enumerate(mask.tolist()) if value == 0]


def dual_mask(encoder_drop, decoder_keep):
    return encoder_drop * decoder_keep


def pool_pairs(frames):
    return (frames[::2] + frames[1::2]) / 2


def shifted_window(frames):
    return torch.roll(frames, 1, dims=0)


def local_aggregate(tokens):
    padded = F.pad(tokens, (1, 1))
    return padded[..., :-2] + padded[..., 1:-1] + padded[..., 2:]


class VideoTransformer(nn.Module):
    architecture_id = "video-transformer"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "timesformer"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "videomae"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="timesformer", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"timesformer", "vivit", "videomae", "videomaev2", "mvit", "video-swin", "uniformer"}:
            raise ValueError("unknown video-transformer mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid video-transformer dimensions")
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
