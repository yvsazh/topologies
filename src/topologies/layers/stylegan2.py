import torch
from torch import nn
from torch.nn import functional as F


class StyleGAN2ModulatedConv2d(nn.Module):
    component_id = "stylegan-2"

    def __init__(self, in_channels=4, out_channels=None, style_dim=4,
                 kernel_size=3, eps=1e-8):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        if min(in_channels, out_channels, style_dim, kernel_size) <= 0 or eps <= 0:
            raise ValueError("channel counts, style_dim, kernel_size, and eps must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_dim = style_dim
        self.kernel_size = kernel_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.style = nn.Linear(style_dim, in_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def demodulated_weights(self, style):
        style = torch.as_tensor(
            style, dtype=self.weight.dtype, device=self.weight.device
        )
        if style.ndim != 2 or style.shape[-1] != self.style_dim:
            raise ValueError("style must have shape (batch, style_dim)")
        scales = self.style(style) + 1.0
        modulated = self.weight.unsqueeze(0) * scales[:, None, :, None, None]
        norms = torch.rsqrt(
            modulated.square().sum(dim=(2, 3, 4), keepdim=True) + self.eps
        )
        return modulated * norms

    def forward(self, features, style):
        if features.ndim != 4 or style.ndim != 2:
            raise ValueError("features must be NCHW and style must be a batch of vectors")
        batch, channels, height, width = features.shape
        if (channels != self.in_channels or style.shape[0] != batch
                or style.shape[1] != self.style_dim):
            raise ValueError("feature and style batch/channel dimensions do not match")
        weights = self.demodulated_weights(style).reshape(
            batch * self.out_channels, self.in_channels,
            self.kernel_size, self.kernel_size
        )
        grouped_input = features.reshape(1, batch * self.in_channels, height, width)
        result = F.conv2d(grouped_input, weights, padding=self.kernel_size // 2, groups=batch)
        return result.reshape(
            batch, self.out_channels, result.shape[-2], result.shape[-1]
        ) + self.bias.view(1, -1, 1, 1)


StyleGAN2 = StyleGAN2ModulatedConv2d
