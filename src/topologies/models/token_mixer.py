import torch
from torch import nn
import torch.nn.functional as F


def mix_tokens(tokens, token_linear):
    return token_linear(tokens.transpose(-1, -2)).transpose(-1, -2)


def affine(values, gamma, beta):
    return gamma * values + beta


def spatial_gate(values, token_linear):
    first, second = values.chunk(2, dim=-1)
    gated = mix_tokens(second, token_linear)
    return first * gated


def pool_mix(values):
    return F.avg_pool2d(values, 3, stride=1, padding=1, count_include_pad=False) - values


class TokenMixer(nn.Module):
    architecture_id = "token-mixer"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "mixer"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "poolformer"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="mixer", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"mixer", "resmlp", "gmlp", "convmixer", "poolformer"}:
            raise ValueError("unknown token mixer mode")
        if min(in_channels, num_classes, width) <= 0 or width % 2:
            raise ValueError("invalid token mixer dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        self.patch = nn.Conv2d(in_channels, width, 16, stride=16, bias=False)
        self.token = nn.Linear(4, 4, bias=False)
        self.channel = nn.Linear(width, width, bias=False)
        self.gamma = nn.Parameter(torch.ones(1, 1, width))
        self.beta = nn.Parameter(torch.zeros(1, 1, width))
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False)
        self.pointwise = nn.Conv2d(width, width, 1, bias=False)
        self.fc = nn.Linear(width, num_classes)

    def forward(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        features = self.patch(pixels)
        if self.mode == "convmixer":
            hidden = self.pointwise(self.depthwise(features))
        elif self.mode == "poolformer":
            hidden = features + pool_mix(features)
        else:
            tokens = features.flatten(2).transpose(1, 2)
            if self.mode == "resmlp":
                tokens = affine(tokens, self.gamma, self.beta)
            if self.mode == "gmlp":
                gated = spatial_gate(tokens, self.token)
                tokens = torch.cat((gated, torch.zeros_like(gated)), dim=-1)
            else:
                tokens = self.channel(mix_tokens(tokens, self.token))
            hidden = tokens.transpose(1, 2).reshape(features.shape)
        return self.fc(hidden.mean(dim=(2, 3)))
