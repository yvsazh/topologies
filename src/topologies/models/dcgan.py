from __future__ import annotations
import math
import torch
from torch import nn

def _check_size(size: int) -> None:
    if size < 4 or size & size - 1:
        raise ValueError('image_size must be a power of two >= 4')

class DCGANGenerator(nn.Module):
    architecture_id = 'dcgan-generator'
    PRESETS = {'tiny': 32, 'small': 64, 'base': 128}

    def __init__(self, latent_dim: int=100, out_channels: int=3, image_size: int=64, features: int=64):
        super().__init__()
        _check_size(image_size)
        if latent_dim < 1 or out_channels < 1 or features < 1:
            raise ValueError('latent_dim, channels, and features must be positive')
        self.latent_dim, self.out_channels, self.image_size = (latent_dim, out_channels, image_size)
        steps = int(math.log2(image_size)) - 2
        widths = [min(features * 2 ** (steps - i - 1), features * 8) for i in range(steps)]
        layers: list[nn.Module] = [nn.ConvTranspose2d(latent_dim, widths[0], 4, 1, 0, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(True)]
        current = widths[0]
        for width in widths[1:]:
            layers.extend((nn.ConvTranspose2d(current, width, 4, 2, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(True)))
            current = width
        layers.extend((nn.ConvTranspose2d(current, out_channels, 4, 2, 1), nn.Tanh()))
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z = z[:, :, None, None]
        if z.ndim != 4 or z.shape[1] != self.latent_dim:
            raise ValueError(f'z must have shape [B,{self.latent_dim}] or [B,{self.latent_dim},1,1]')
        out = self.network(z)
        if out.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError('latent spatial dimensions must be 1x1')
        return out

class DCGANDiscriminator(nn.Module):
    architecture_id = 'dcgan-discriminator'
    PRESETS = {'tiny': 32, 'small': 64, 'base': 128}

    def __init__(self, in_channels: int=3, image_size: int=64, features: int=64):
        super().__init__()
        _check_size(image_size)
        if in_channels < 1 or features < 1:
            raise ValueError('channels and features must be positive')
        self.in_channels, self.image_size = (in_channels, image_size)
        steps = int(math.log2(image_size)) - 2
        widths = [min(features * 2 ** i, features * 8) for i in range(steps)]
        layers: list[nn.Module] = [nn.Conv2d(in_channels, widths[0], 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        current = widths[0]
        for width in widths[1:]:
            layers.extend((nn.Conv2d(current, width, 4, 2, 1, bias=False), nn.BatchNorm2d(width), nn.LeakyReLU(0.2, inplace=True)))
            current = width
        layers.append(nn.Conv2d(current, 1, 4, 1, 0))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (self.in_channels, self.image_size, self.image_size):
            raise ValueError(f'x must have shape [B,{self.in_channels},{self.image_size},{self.image_size}]')
        if x.shape[0] < 1:
            raise ValueError('x batch must be non-empty')
        return self.network(x).reshape(x.shape[0])
