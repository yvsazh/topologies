import torch
from torch import nn


class Affine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return x * self.alpha + self.beta


class ResMLPBlock(nn.Module):
    def __init__(self, tokens, channels, expansion, init_scale=1e-5):
        super().__init__()
        self.token_norm = Affine(channels)
        self.token_mixing = nn.Linear(tokens, tokens, bias=False)
        self.channel_norm = Affine(channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, expansion),
            nn.GELU(),
            nn.Linear(expansion, channels),
        )
        self.token_scale = nn.Parameter(torch.full((channels,), init_scale))
        self.channel_scale = nn.Parameter(torch.full((channels,), init_scale))

    def forward(self, x):
        y = self.token_norm(x)
        y = self.token_mixing(y.transpose(1, 2)).transpose(1, 2)
        x = x + y * self.token_scale
        return x + self.channel_mlp(self.channel_norm(x)) * self.channel_scale


class ResMLP(nn.Module):
    architecture_id = "resmlp"
    def __init__(
        self,
        num_classes=1000,
        image_size=224,
        patch_size=16,
        in_channels=3,
        channels=384,
        depth=12,
        expansion=None,
        num_tokens=None,
        init_scale=1e-5,
    ):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        grid = image_size // patch_size
        expected_tokens = grid * grid
        if num_tokens is not None and num_tokens != expected_tokens:
            raise ValueError("num_tokens must match the image and patch dimensions")
        self.patch_size = patch_size
        self.channels = channels
        self.patch_embed = nn.Conv2d(in_channels, channels, patch_size, stride=patch_size)
        hidden = expansion if expansion is not None else channels * 4
        self.blocks = nn.ModuleList(
            ResMLPBlock(expected_tokens, channels, hidden, init_scale=init_scale)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x).mean(dim=1))
