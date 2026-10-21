import torch
from torch import nn
import torch.nn.functional as F


def clip_loss(image, text, temperature):
    image = F.normalize(image, dim=-1)
    text = F.normalize(text, dim=-1)
    logits = image @ text.transpose(-1, -2) / temperature
    labels = torch.arange(image.shape[0], device=image.device)
    image_loss = F.cross_entropy(logits, labels)
    text_loss = F.cross_entropy(logits.transpose(-1, -2), labels)
    return 0.5 * (image_loss + text_loss)


def siglip_loss(image, text, temperature, bias):
    image = F.normalize(image, dim=-1)
    text = F.normalize(text, dim=-1)
    logits = image @ text.transpose(-1, -2) * temperature.exp() + bias
    labels = torch.eye(image.shape[0], device=image.device) * 2 - 1
    return -F.logsigmoid(labels * logits).mean()


def swiglu(up, gate):
    return up * F.silu(gate)


def parallel_residual(values, attention, mlp):
    return values + attention + mlp


def causal_next_patch(logits, targets):
    length = logits.shape[1]
    mask = torch.tril(torch.ones(length, length, device=logits.device))
    return F.cross_entropy(logits[:, :-1].transpose(1, 2), targets[:, 1:]), mask


class ContrastiveVision(nn.Module):
    architecture_id = "contrastive-vision"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "clip"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "siglip"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="clip", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"clip", "siglip", "eva02", "aim", "vit22b"}:
            raise ValueError("unknown contrastive vision mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid contrastive vision dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        self.patch = nn.Conv2d(in_channels, width, 16, stride=16, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.up = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, width, bias=False)
        self.fc = nn.Linear(width, num_classes)
        self.temperature = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def objective(self, mode, image, text=None, targets=None):
        if mode == "clip":
            return clip_loss(image, text, self.temperature.exp().clamp_min(1e-3))
        if mode == "siglip":
            return siglip_loss(image, text, self.temperature, self.bias)
        if mode == "eva02":
            return swiglu(self.up(image), self.gate(image)).sum()
        if mode == "aim":
            loss, _mask = causal_next_patch(image, targets)
            return loss
        return parallel_residual(image, text, targets).sum()

    def forward(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        tokens = self.patch(pixels).flatten(2).transpose(1, 2)
        if self.mode == "eva02":
            hidden = swiglu(self.up(tokens), self.gate(tokens))
        elif self.mode == "vit22b":
            hidden = parallel_residual(tokens, self.proj(tokens), self.up(tokens))
        else:
            hidden = self.proj(tokens)
        return self.fc(hidden.mean(dim=1))
