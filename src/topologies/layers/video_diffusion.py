import torch
from torch import nn


class VideoDiffusionStep(nn.Module):
    component_id = "video-diffusion"

    def __init__(self):
        super().__init__()

    def _scale(self, value, name, x):
        if isinstance(value, bool):
            raise TypeError(f"{name} must be numeric")
        try:
            scale = torch.as_tensor(value, device=x.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError(f"{name} must be numeric") from exc
        if scale.dtype == torch.bool or scale.is_complex():
            raise TypeError(f"{name} must be real-valued")
        scale = scale.to(dtype=x.dtype)
        if not torch.isfinite(scale).all() or torch.any(scale < 0):
            raise ValueError(f"{name} must be a finite non-negative scale")
        try:
            broadcast_shape = torch.broadcast_shapes(x.shape, scale.shape)
        except RuntimeError as exc:
            raise ValueError(f"{name} must be broadcastable to x") from exc
        if broadcast_shape != x.shape:
            raise ValueError(f"{name} must be broadcastable to x without expanding it")
        return scale

    def _video(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 5:
            raise ValueError("x must be a rank-5 video tensor")
        if not x.is_floating_point():
            raise ValueError("x must have a floating-point dtype")
        return x

    def _legacy(self, x, noise, alpha_bar):
        if not isinstance(noise, torch.Tensor) or noise.ndim != 5:
            raise ValueError("noise must be a rank-5 tensor")
        if noise.shape != x.shape:
            raise ValueError("noise must have the same shape as x")
        if not noise.is_floating_point():
            raise ValueError("noise must have a floating-point dtype")
        try:
            alpha_bar = torch.as_tensor(alpha_bar, device=x.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("alpha_bar must be numeric") from exc
        if alpha_bar.dtype == torch.bool or alpha_bar.is_complex():
            raise ValueError("alpha_bar must be real-valued")
        alpha_bar = alpha_bar.to(dtype=x.dtype)
        if not torch.isfinite(alpha_bar).all() or torch.any(alpha_bar < 0) or torch.any(alpha_bar > 1):
            raise ValueError("alpha_bar must be finite and between zero and one")
        noise = noise.to(device=x.device, dtype=x.dtype)
        return torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise

    def _direct(self, x, alpha_t, sigma_t, epsilon):
        alpha = self._scale(alpha_t, "alpha_t", x)
        sigma = self._scale(sigma_t, "sigma_t", x)
        if epsilon is None:
            epsilon = torch.randn_like(x)
        elif not isinstance(epsilon, torch.Tensor):
            raise ValueError("epsilon must be a tensor with the same shape as x")
        elif epsilon.shape != x.shape:
            raise ValueError("epsilon must have the same shape as x")
        elif not epsilon.is_floating_point():
            raise ValueError("epsilon must have a floating-point dtype")
        epsilon = epsilon.to(device=x.device, dtype=x.dtype)
        return alpha * x + sigma * epsilon

    def forward(self, x=None, noise=None, alpha_bar=None, epsilon=None, *, alpha_t=None, sigma_t=None, clean=None):
        named_direct = alpha_t is not None or sigma_t is not None
        legacy_call = clean is not None or (
            not named_direct
            and epsilon is None
            and isinstance(x, torch.Tensor)
            and isinstance(noise, torch.Tensor)
            and noise.shape == x.shape
            and alpha_bar is not None
        )
        if clean is not None:
            if x is not None:
                raise ValueError("provide only one of x and clean")
            x = clean
        x = self._video(x)
        if named_direct:
            if alpha_t is None or sigma_t is None:
                raise ValueError("alpha_t and sigma_t must be provided together")
            if noise is not None or alpha_bar is not None:
                raise ValueError("direct scales cannot be combined with noise or alpha_bar")
            return self._direct(x, alpha_t, sigma_t, epsilon)
        if legacy_call:
            if epsilon is not None:
                raise ValueError("legacy inputs cannot include epsilon")
            return self._legacy(x, noise, alpha_bar)
        if noise is not None and alpha_bar is not None:
            if epsilon is not None:
                return self._direct(x, noise, alpha_bar, epsilon)
            return self._direct(x, noise, alpha_bar, None)
        raise ValueError("provide noise and alpha_bar or alpha_t and sigma_t")
