import torch
from torch import nn
from torch.nn import functional as F


class EdgeNeXt(nn.Module):
    component_id = "edgenext"
    def __init__(self, channels=32, groups=4, kernel_size=7, expansion=2, layer_scale=1.0):
        super().__init__()
        if channels < 1 or groups < 1 or channels % groups:
            raise ValueError("channels must be positive and divisible by groups")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.channels = channels
        self.groups = groups
        self.depthwise = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.attention_projection = nn.Conv2d(channels, channels, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * expansion, 1),
            nn.GELU(),
            nn.Conv2d(channels * expansion, channels, 1),
        )
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].bias)
        self.layer_scale = nn.Parameter(torch.full((channels,), float(layer_scale)))
        self.last_attention_weights = None

    def attention(self, x):
        batch, channels, height, width = x.shape
        group_channels = channels // self.groups
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = q.reshape(batch, self.groups, group_channels, height * width)
        k = k.reshape(batch, self.groups, group_channels, height * width)
        v = v.reshape(batch, self.groups, group_channels, height * width)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1))
        weights = scores.softmax(dim=-1)
        attended = torch.matmul(weights.transpose(-2, -1), v).reshape(batch, channels, height, width)
        self.last_attention_weights = weights
        return self.attention_projection(attended)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("input must have shape (batch, channels, height, width)")
        residual = x
        y = self.depthwise(x)
        y = y + self.attention(y)
        y = self.norm(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        y = y + self.mlp(y)
        return residual + y * self.layer_scale.view(1, -1, 1, 1)
