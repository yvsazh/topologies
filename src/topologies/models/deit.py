import torch
from torch import nn


class DeiT(nn.Module):
    architecture_id = "deit"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "d_model": 16, "mode": "distilled"},
        "base": {"in_channels": 3, "num_classes": 10, "d_model": 32, "mode": "deit3"},
    }

    def __init__(self, in_channels=3, num_classes=10, d_model=16, mode="distilled", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"deit", "distilled", "deit3"}:
            raise ValueError("unknown DeiT mode")
        if min(in_channels, num_classes, d_model) <= 0:
            raise ValueError("invalid DeiT dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.d_model = d_model
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        self.patch = nn.Conv2d(in_channels, d_model, 16, stride=16, bias=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.distill = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, 6, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.gamma1 = nn.Parameter(torch.ones(d_model))
        self.gamma2 = nn.Parameter(torch.ones(d_model))
        self.head = nn.Linear(d_model, num_classes)
        self.head_dist = nn.Linear(d_model, num_classes)

    def encode(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        patches = self.patch(pixels).flatten(2).transpose(1, 2)
        batch = pixels.shape[0]
        cls = self.cls.expand(batch, -1, -1)
        if self.mode == "distilled":
            sequence = torch.cat((cls, self.distill.expand(batch, -1, -1), patches), dim=1)
        else:
            sequence = torch.cat((cls, patches), dim=1)
        sequence = sequence + self.position[:, :sequence.shape[1]]
        branch = self.attention(self.norm1(sequence))
        sequence = sequence + (self.gamma1 * branch if self.mode == "deit3" else branch)
        branch = self.fc2(torch.relu(self.fc1(self.norm2(sequence))))
        return sequence + (self.gamma2 * branch if self.mode == "deit3" else branch)

    def attention(self, sequence):
        query, key, value = self.qkv(sequence).chunk(3, dim=-1)
        scale = sequence.shape[-1] ** 0.5
        weights = torch.softmax(query @ key.transpose(-1, -2) / scale, dim=-1)
        return self.proj(weights @ value)

    def forward(self, pixels):
        sequence = self.encode(pixels)
        class_logits = self.head(sequence[:, 0])
        if self.mode == "distilled":
            return (class_logits + self.head_dist(sequence[:, 1])) / 2
        return class_logits
