import torch
from torch import nn
from torch.nn import functional as F


class SETR(nn.Module):
    architecture_id = "setr"

    def __init__(
        self, in_channels=3, patch_size=16, hidden_dim=768, num_heads=12,
        num_classes=19, base_grid_size=14,
    ):
        super().__init__()
        if min(in_channels, patch_size, hidden_dim, num_heads, num_classes, base_grid_size) <= 0:
            raise ValueError("all channel, patch, embedding, head, class, and grid sizes must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.encoder = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.position_embedding = nn.Parameter(torch.zeros(1, hidden_dim, base_grid_size, base_grid_size))
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != self.patch_embed.in_channels:
            raise ValueError("image must have shape (batch, in_channels, height, width)")
        if image.shape[-2] % self.patch_embed.kernel_size[0] or image.shape[-1] % self.patch_embed.kernel_size[1]:
            raise ValueError("image height and width must be divisible by patch_size")
        features = self.patch_embed(image)
        batch, channels, height, width = features.shape
        positions = F.interpolate(
            self.position_embedding, size=(height, width), mode="bicubic", align_corners=False
        )
        tokens = (features + positions).flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return self.classifier(features)
