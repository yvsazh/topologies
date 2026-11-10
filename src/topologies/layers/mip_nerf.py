import torch
from torch import nn


class IntegratedPositionalEncoding(nn.Module):
    component_id = "mip-nerf"
    PRESETS = {
        "tiny": {"min_deg": 0, "max_deg": 2},
        "small": {"min_deg": 0, "max_deg": 8},
        "base": {"min_deg": 0, "max_deg": 16},
    }

    def __init__(self, min_deg=0, max_deg=16):
        super().__init__()
        if type(min_deg) is not int or min_deg < 0:
            raise ValueError("min_deg must be a non-negative integer")
        if type(max_deg) is not int:
            raise ValueError("max_deg must be an integer")
        if max_deg <= min_deg:
            raise ValueError("max_deg must be greater than min_deg")
        self.min_deg = min_deg
        self.max_deg = max_deg

    def forward(self, mean, variance):
        if not torch.is_tensor(mean) or not torch.is_tensor(variance):
            raise ValueError("mean and variance must be torch tensors")
        if mean.shape != variance.shape:
            raise ValueError("mean and variance must have the same shape")
        if mean.ndim == 0 or mean.shape[-1] != 3:
            raise ValueError("mean and variance must have a last dimension of size 3")
        if not mean.is_floating_point() or not variance.is_floating_point():
            raise ValueError("mean and variance must use floating point dtypes")
        if mean.dtype != variance.dtype or mean.device != variance.device:
            raise ValueError("mean and variance must have the same dtype and device")
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise ValueError("mean and variance must contain only finite values")
        if torch.any(variance < 0):
            raise ValueError("variance values must be non-negative")
        degrees = torch.arange(self.min_deg, self.max_deg, dtype=mean.dtype, device=mean.device)
        scales = torch.exp2(degrees)
        scaled_mean = mean.unsqueeze(-2) * scales.reshape(-1, 1)
        scaled_variance = variance.unsqueeze(-2) * scales.square().reshape(-1, 1)
        attenuation = torch.exp(-0.5 * scaled_variance)
        sine = (attenuation * torch.sin(scaled_mean)).flatten(start_dim=-2)
        cosine = (attenuation * torch.cos(scaled_mean)).flatten(start_dim=-2)
        return torch.cat((sine, cosine), dim=-1)
