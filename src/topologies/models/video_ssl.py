import torch
from torch import nn
import torch.nn.functional as F


def contrastive(positive, negative):
    scores = torch.stack((positive, negative))
    return -torch.log_softmax(scores, dim=0)[0]


def masked_token_loss(logits, targets, mask):
    loss = F.cross_entropy(logits, targets, reduction="none")
    return (loss * mask).sum() / mask.sum()


def joint_loss(video, image):
    return video + image


class VideoSSL(nn.Module):
    architecture_id = "video-ssl"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "cvrl"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "bevt"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="cvrl", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"cvrl", "vimpac", "bevt"}:
            raise ValueError("unknown video-ssl mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid video-ssl dimensions")
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
