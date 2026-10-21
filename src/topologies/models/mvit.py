import torch
from torch import nn
from torch.nn import functional as F


class MViT(nn.Module):
    architecture_id = "mvit"
    def __init__(self, in_channels=8, num_classes=2, num_heads=2, pooled_tokens=4):
        super().__init__()
        if in_channels % num_heads:
            raise ValueError("in_channels must be divisible by num_heads")
        if pooled_tokens < 1:
            raise ValueError("pooled_tokens must be positive")
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.pooled_tokens = pooled_tokens
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.output = nn.Linear(in_channels, in_channels)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("expected input with shape (batch, tokens, channels)")
        batch, length, channels = x.shape
        attended = self.pooling_attention(x)
        features = x + self.output(attended)
        return self.classifier(features.mean(dim=1))

    def pooling_attention(self, x):
        if x.ndim != 3:
            raise ValueError("expected input with shape (batch, tokens, channels)")
        batch, length, channels = x.shape
        q = self.query(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        pooled_length = min(self.pooled_tokens, length)
        pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), pooled_length).transpose(1, 2)
        k = self.key(pooled).reshape(batch, pooled_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(pooled).reshape(batch, pooled_length, self.num_heads, self.head_dim).transpose(1, 2)
        weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5), dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(batch, length, channels)
        return attended
