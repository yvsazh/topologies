import math
from numbers import Real

import torch
from torch import nn


class DishTS(nn.Module):
    component_id = "dish-ts"
    PRESETS = {
        "tiny": {"lookback": 4, "horizon": 2, "channels": 2, "negative_slope": 0.01, "scale_eps": 1e-6},
        "small": {"lookback": 24, "horizon": 24, "channels": 8, "negative_slope": 0.01, "scale_eps": 1e-6},
    }

    def __init__(self, lookback, horizon, channels, negative_slope=0.01, scale_eps=1e-6,
                 forecaster=None):
        super().__init__()
        dimensions = (lookback, horizon, channels)
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in dimensions):
            raise ValueError("lookback, horizon, and channels must be positive integers")
        if (not isinstance(negative_slope, Real) or isinstance(negative_slope, bool)
                or not math.isfinite(negative_slope) or negative_slope < 0):
            raise ValueError("negative_slope must be a finite nonnegative number")
        if (not isinstance(scale_eps, Real) or isinstance(scale_eps, bool)
                or not math.isfinite(scale_eps) or scale_eps <= 0):
            raise ValueError("scale_eps must be a finite positive number")
        if forecaster is not None and not isinstance(forecaster, nn.Module):
            raise ValueError("forecaster must be an nn.Module or None")
        self.lookback = lookback
        self.horizon = horizon
        self.channels = channels
        self.scale_eps = float(scale_eps)
        self.back_conet = nn.ModuleList(
            nn.Linear(lookback, 1, bias=False) for _ in range(channels)
        )
        self.hori_conet = nn.ModuleList(
            nn.Linear(lookback, 1, bias=False) for _ in range(channels)
        )
        self.level_activation = nn.LeakyReLU(float(negative_slope))
        self._uses_default_forecaster = forecaster is None
        self.forecaster = nn.Linear(lookback, horizon) if forecaster is None else forecaster

    def _validate_history(self, history):
        if not torch.is_tensor(history) or history.ndim != 3:
            raise ValueError("history must have shape (batch, lookback, channels)")
        if tuple(history.shape[1:]) != (self.lookback, self.channels):
            raise ValueError("history must have shape (batch, lookback, channels)")
        if not history.is_floating_point():
            raise ValueError("history must have a floating-point dtype")

    def coefficients(self, history):
        self._validate_history(history)
        back_levels = [
            layer(history[:, :, index]).unsqueeze(-1)
            for index, layer in enumerate(self.back_conet)
        ]
        hori_levels = [
            layer(history[:, :, index]).unsqueeze(-1)
            for index, layer in enumerate(self.hori_conet)
        ]
        phi_b = self.level_activation(torch.cat(back_levels, dim=-1))
        phi_h = self.level_activation(torch.cat(hori_levels, dim=-1))
        xi_b = (history - phi_b).square().mean(dim=1, keepdim=True).sqrt()
        xi_h = (history - phi_h).square().mean(dim=1, keepdim=True).sqrt()
        return {"phi_b": phi_b, "xi_b": xi_b, "phi_h": phi_h, "xi_h": xi_h}

    def forward(self, history, return_aux=False):
        self._validate_history(history)
        coefficients = self.coefficients(history)
        normalized = (history - coefficients["phi_b"]) / coefficients["xi_b"].clamp_min(self.scale_eps)
        if self._uses_default_forecaster:
            forecast = self.forecaster(normalized.transpose(1, 2)).transpose(1, 2)
        else:
            forecast = self.forecaster(normalized)
        expected = (history.shape[0], self.horizon, self.channels)
        if not torch.is_tensor(forecast) or tuple(forecast.shape) != expected:
            raise ValueError("forecaster output must have shape (batch, horizon, channels)")
        forecast = coefficients["xi_h"] * forecast + coefficients["phi_h"]
        if return_aux:
            return forecast, coefficients
        return forecast

    @staticmethod
    def prior_guidance_loss(prediction, target, phi_h, alpha):
        if (not isinstance(alpha, Real) or isinstance(alpha, bool)
                or not math.isfinite(alpha) or not 0 <= alpha <= 1):
            raise ValueError("alpha must be a finite number in [0, 1]")
        if not torch.is_tensor(prediction) or prediction.ndim != 3:
            raise ValueError("prediction must have shape (batch, horizon, channels)")
        if not torch.is_tensor(target) or target.shape != prediction.shape:
            raise ValueError("target must match prediction shape")
        if not torch.is_tensor(phi_h) or tuple(phi_h.shape) != (
            prediction.shape[0], 1, prediction.shape[2]
        ):
            raise ValueError("phi_h must have shape (batch, 1, channels)")
        forecast_error = (prediction - target).square().sum(dim=1)
        target_mean = target.mean(dim=1, keepdim=True)
        guidance_error = (target_mean - phi_h).square().squeeze(1)
        return (forecast_error + float(alpha) * guidance_error).mean()
