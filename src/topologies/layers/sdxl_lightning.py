from numbers import Real

import torch
from torch import nn


class LightningStep(nn.Module):
    component_id = "sdxl-lightning"

    def __init__(self):
        super().__init__()

    def forward(self, z, noise, sigma):
        if not isinstance(z, torch.Tensor) or not isinstance(noise, torch.Tensor):
            raise TypeError("z and noise must be tensors")
        if not isinstance(sigma, (torch.Tensor, Real)):
            raise TypeError("sigma must be a real scalar or tensor")
        sigma_shape = sigma.shape if isinstance(sigma, torch.Tensor) else ()
        try:
            output_shape = torch.broadcast_shapes(z.shape, noise.shape, sigma_shape)
        except RuntimeError as error:
            raise ValueError("z, noise, and sigma must have broadcast-compatible shapes") from error
        if output_shape != z.shape:
            raise ValueError("noise and sigma must broadcast to z's shape without expanding z")
        noise = torch.where((sigma == 0) & ~torch.isfinite(noise), 0, noise)
        return z - sigma * noise
