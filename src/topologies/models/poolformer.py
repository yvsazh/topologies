import torch
from torch import nn
import torch.nn.functional as F


class PoolFormerBlock(nn.Module):
    def __init__(self, channels, pool_size=3, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.pool_size = pool_size
        self.norm2 = nn.GroupNorm(1, channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),
            nn.GELU(),
            nn.Linear(channels * mlp_ratio, channels),
        )

    def forward(self, x):
        y = self.norm1(x)
        pooled = F.avg_pool2d(
            y, self.pool_size, stride=1, padding=self.pool_size // 2,
            count_include_pad=False,
        )
        x = x + pooled - y
        y = self.norm2(x).permute(0, 2, 3, 1)
        return x + self.mlp(y).permute(0, 3, 1, 2)


class PoolFormer(nn.Module):
    architecture_id = "poolformer"
    def __init__(self, in_channels=3, num_classes=1000, embed_dim=32, depth=2, pool_size=3, channels=None, classes=None):
        super().__init__()
        if channels is not None:
            embed_dim = channels
        if classes is not None:
            num_classes = classes
        if pool_size < 1 or pool_size % 2 == 0:
            raise ValueError("pool_size must be a positive odd integer")
        self.stem = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.blocks = nn.Sequential(*(PoolFormerBlock(embed_dim, pool_size) for _ in range(depth)))
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x):
        x = self.blocks(self.stem(x))
        return self.norm(x.mean(dim=(2, 3)))

    def forward(self, x):
        return self.head(self.forward_features(x))
