import torch
from torch import nn


def visible_tokens(tokens, mask):
    keep = mask[0] == 0
    return tokens[:, keep]


def masked_mse(prediction, target, mask):
    per_token = (prediction - target).pow(2).mean(dim=-1)
    return (per_token * mask).sum() / mask.sum()


def simmim_input(tokens, mask, mask_token):
    keep = 1.0 - mask.unsqueeze(-1)
    return tokens * keep + mask_token * mask.unsqueeze(-1)


def gradient_histogram(image):
    horizontal = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(dim=(2, 3))
    vertical = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(dim=(2, 3))
    return torch.cat((horizontal, vertical), dim=-1)


class MaskedImage(nn.Module):
    architecture_id = "masked-image"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "mae"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "simmim"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="mae", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"mae", "simmim", "maskfeat"}:
            raise ValueError("unknown masked image mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid masked image dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        self.patch = nn.Conv2d(in_channels, width, 16, stride=16, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.proj = nn.Linear(width, width, bias=False)
        self.fc = nn.Linear(width, num_classes)

    def encode(self, tokens, mask):
        if self.mode == "mae":
            return visible_tokens(tokens, mask)
        return simmim_input(tokens, mask, self.mask_token)

    def forward(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        tokens = self.patch(pixels).flatten(2).transpose(1, 2)
        return self.fc(self.proj(tokens).mean(dim=1))
