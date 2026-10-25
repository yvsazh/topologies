import torch
from torch import nn


class NeuSOpacity(nn.Module):
    component_id = "neus"

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise ValueError("NeuSOpacity does not accept constructor arguments")
        super().__init__()

    def forward(self, sdf_samples, inv_s):
        if not torch.is_tensor(sdf_samples) or not sdf_samples.is_floating_point():
            raise ValueError("sdf_samples must be a floating-point tensor")
        if sdf_samples.ndim < 1:
            raise ValueError("sdf_samples must have a sample axis")
        if sdf_samples.shape[-1] < 2:
            raise ValueError("sdf_samples must contain at least two SDF knots")
        if not torch.isfinite(sdf_samples).all():
            raise ValueError("sdf_samples must contain only finite values")
        if isinstance(inv_s, (int, float)) and not isinstance(inv_s, bool):
            inv_s = torch.as_tensor(inv_s, dtype=sdf_samples.dtype, device=sdf_samples.device)
        if not torch.is_tensor(inv_s) or not inv_s.is_floating_point() or inv_s.ndim != 0:
            raise ValueError("inv_s must be a floating-point scalar tensor")
        if inv_s.device != sdf_samples.device:
            raise ValueError("inv_s and sdf_samples must be on the same device")
        if not torch.isfinite(inv_s) or inv_s <= 0:
            raise ValueError("inv_s must be positive and finite")
        scaled = sdf_samples * inv_s
        log_cdf = -torch.nn.functional.softplus(-scaled)
        log_ratio = log_cdf[..., 1:] - log_cdf[..., :-1]
        return (-torch.expm1(log_ratio)).clamp_min(0.0)
