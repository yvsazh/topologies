"""DDPM schedules and DDPM/DDIM image samplers."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_cumprod: torch.Tensor

    @classmethod
    def create(cls, num_timesteps: int, kind: str = "linear", beta_start: float = 1e-4,
               beta_end: float = .02, device: torch.device | str = "cpu") -> DiffusionSchedule:
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("require 0 < beta_start < beta_end < 1")
        if kind == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif kind == "cosine":
            points = torch.linspace(0, num_timesteps, num_timesteps + 1, device=device) / num_timesteps
            abar = torch.cos((points + .008) / 1.008 * math.pi / 2).square()
            abar = abar / abar[0]
            betas = (1 - abar[1:] / abar[:-1]).clamp(max=.999)
        else:
            raise ValueError("kind must be 'linear' or 'cosine'")
        alphas = 1 - betas
        return cls(betas, alphas, torch.cumprod(alphas, dim=0))

    def extract(self, values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        return values.to(t.device)[t.long()].reshape(t.shape[0], *((1,) * (ndim - 1)))

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.shape != (x0.shape[0],):
            raise ValueError("t must have shape [B]")
        a = self.extract(self.alpha_cumprod, t, x0.ndim)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise


def _predict_x0_eps(model: Callable, x: torch.Tensor, t: torch.Tensor, schedule: DiffusionSchedule,
                    prediction_type: str, conditioning=None):
    if prediction_type not in {"epsilon", "x0", "v"}:
        raise ValueError("prediction_type must be epsilon, x0, or v")
    pred = model(x, t) if conditioning is None else model(x, t, conditioning)
    a = schedule.extract(schedule.alpha_cumprod, t, x.ndim)
    if prediction_type == "epsilon":
        eps = pred
        x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()
    elif prediction_type == "x0":
        x0 = pred
        eps = (x - a.sqrt() * x0) / (1 - a).sqrt().clamp_min(1e-12)
    else:
        x0 = a.sqrt() * x - (1 - a).sqrt() * pred
        eps = a.sqrt() * pred + (1 - a).sqrt() * x
    return x0, eps


@torch.no_grad()
def sample_ddpm(model: Callable, shape: tuple[int, ...], schedule: DiffusionSchedule, *,
                prediction_type: str = "epsilon", device: torch.device | str = "cpu",
                generator: torch.Generator | None = None, conditioning=None,
                callback: Callable | None = None) -> torch.Tensor:
    """``callback(index, t, x_t, x0_pred, eps_pred)`` (optional) observes every step."""
    if not shape or shape[0] <= 0 or len(schedule.betas) <= 0:
        raise ValueError("shape must have a positive batch and schedule T must be > 0")
    x = torch.randn(shape, device=device, generator=generator)
    for index, step in enumerate(reversed(range(len(schedule.betas)))):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        x0, eps = _predict_x0_eps(model, x, t, schedule, prediction_type, conditioning)
        if callback is not None:
            callback(index, step, x, x0, eps)
        beta = schedule.extract(schedule.betas, t, x.ndim)
        alpha = schedule.extract(schedule.alphas, t, x.ndim)
        abar = schedule.extract(schedule.alpha_cumprod, t, x.ndim)
        mean = (x - beta / (1 - abar).sqrt() * eps) / alpha.sqrt()
        if step:
            x = mean + ((beta * (1 - schedule.extract(schedule.alpha_cumprod, t - 1, x.ndim)) / (1 - abar)).clamp_min(0).sqrt()) * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        else:
            x = mean
    return x


@torch.no_grad()
def sample_ddim(model: Callable, shape: tuple[int, ...], schedule: DiffusionSchedule, *,
                steps: int = 50, eta: float = 0.0, prediction_type: str = "epsilon",
                device: torch.device | str = "cpu", generator: torch.Generator | None = None,
                conditioning=None, callback: Callable | None = None) -> torch.Tensor:
    """``callback(index, t, x_t, x0_pred, eps_pred)`` (optional) observes every step."""
    total = len(schedule.betas)
    if total <= 0 or steps <= 0 or steps > total or eta < 0:
        raise ValueError("require T > 0, 1 <= steps <= T, and eta >= 0")
    if not shape or shape[0] <= 0:
        raise ValueError("shape must have a positive batch")
    indices = torch.linspace(total - 1, 0, steps, device=device).round().long().unique_consecutive()
    x = torch.randn(shape, device=device, generator=generator)
    for i, step_tensor in enumerate(indices):
        step = int(step_tensor)
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        x0, eps = _predict_x0_eps(model, x, t, schedule, prediction_type, conditioning)
        if callback is not None:
            callback(i, step, x, x0, eps)
        prev = int(indices[i + 1]) if i + 1 < len(indices) else -1
        a = schedule.alpha_cumprod[step].to(device)
        ap = schedule.alpha_cumprod[prev].to(device) if prev >= 0 else torch.ones((), device=device)
        sigma = eta * (((1 - ap) / (1 - a) * (1 - a / ap)).clamp_min(0).sqrt())
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) if eta else 0
        x = ap.sqrt() * x0 + (1 - ap - sigma.square()).clamp_min(0).sqrt() * eps + sigma * noise
    return x
