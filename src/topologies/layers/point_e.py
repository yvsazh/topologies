import math
from numbers import Real

import torch
from torch import nn


class PointE(nn.Module):
    component_id = "point-e"

    def __init__(self, num_timesteps=None, schedule="cosine", clip_denoised=False,
                 predictor=None, betas=None):
        super().__init__()
        if num_timesteps is not None and (
                isinstance(num_timesteps, bool) or not isinstance(num_timesteps, int)
                or num_timesteps < 2):
            raise ValueError("num_timesteps must be an integer of at least 2")
        if not isinstance(schedule, str) or schedule not in ("cosine", "linear"):
            raise ValueError("schedule must be 'cosine' or 'linear'")
        if type(clip_denoised) is not bool:
            raise ValueError("clip_denoised must be a bool")
        if predictor is not None and not isinstance(predictor, nn.Module):
            raise ValueError("predictor must be a torch.nn.Module")
        if betas is not None:
            if not isinstance(betas, (tuple, list)) or len(betas) < 2:
                raise ValueError("betas must be a sequence with at least 2 values")
            if num_timesteps is not None and num_timesteps != len(betas):
                raise ValueError("num_timesteps must match the number of betas")
            if any(isinstance(beta, bool) or not isinstance(beta, Real)
                   or not math.isfinite(float(beta)) or not 0.0 < float(beta) < 1.0
                   for beta in betas):
                raise ValueError("each beta must be a finite real value in (0, 1)")
            beta_values = [float(beta) for beta in betas]
            num_timesteps = len(beta_values)
        else:
            num_timesteps = 1024 if num_timesteps is None else num_timesteps
            if schedule == "cosine":
                def alpha_bar(time):
                    offset = 0.008
                    return math.cos((time + offset) / (1.0 + offset) * math.pi / 2.0) ** 2

                beta_values = [
                    min(1.0 - alpha_bar((index + 1) / num_timesteps)
                        / alpha_bar(index / num_timesteps), 0.999)
                    for index in range(num_timesteps)
                ]
            else:
                scale = 1000.0 / num_timesteps
                start = 0.0001 * scale
                end = 0.02 * scale
                beta_values = [
                    min(start + (end - start) * index / (num_timesteps - 1), 0.999)
                    for index in range(num_timesteps)
                ]

        beta_tensor = torch.tensor(beta_values, dtype=torch.float64)
        alpha_tensor = 1.0 - beta_tensor
        alpha_bar_tensor = torch.cumprod(alpha_tensor, dim=0)
        alpha_bar_previous = torch.cat((torch.ones(1, dtype=torch.float64), alpha_bar_tensor[:-1]))
        posterior_variance = beta_tensor * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar_tensor)
        posterior_log_variance_clipped = torch.log(torch.cat((
            posterior_variance[1:2], posterior_variance[1:]
        )))
        posterior_mean_x0 = beta_tensor * torch.sqrt(alpha_bar_previous) / (1.0 - alpha_bar_tensor)
        posterior_mean_xt = (1.0 - alpha_bar_previous) * torch.sqrt(alpha_tensor) / (
            1.0 - alpha_bar_tensor
        )

        self.num_timesteps = num_timesteps
        self.schedule = schedule
        self.clip_denoised = clip_denoised
        self.predictor = predictor if predictor is not None else nn.Conv1d(6, 12, kernel_size=1)
        self._default_predictor = predictor is None
        self.register_buffer("betas", beta_tensor)
        self.register_buffer("alpha_bar", alpha_bar_tensor)
        self.register_buffer("posterior_mean_x0", posterior_mean_x0)
        self.register_buffer("posterior_mean_xt", posterior_mean_xt)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)

    def forward(self, x_t, timesteps, noise=None, return_x0=True):
        if not torch.is_tensor(x_t) or x_t.ndim != 3 or x_t.shape[1] != 6 or x_t.shape[0] < 1 or x_t.shape[2] < 1:
            raise ValueError("x_t must be a nonempty tensor with shape [B, 6, K]")
        if not x_t.is_floating_point():
            raise ValueError("x_t must have a floating-point dtype")
        if type(return_x0) is not bool:
            raise ValueError("return_x0 must be a bool")

        try:
            timesteps = torch.as_tensor(timesteps, device=x_t.device)
        except (TypeError, ValueError):
            raise ValueError("timesteps must be integer indices with shape [B]") from None
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(x_t.shape[0])
        if timesteps.ndim != 1 or timesteps.shape[0] != x_t.shape[0] or timesteps.dtype not in {
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
            raise ValueError("timesteps must be integer indices with shape [B]")
        timesteps = timesteps.to(dtype=torch.long)
        if bool(((timesteps < 0) | (timesteps >= self.num_timesteps)).any()):
            raise ValueError("timestep values must be in the range [0, num_timesteps)")

        if self._default_predictor:
            model_output = self.predictor(x_t)
        else:
            model_output = self.predictor(x_t, timesteps)
        expected_shape = (x_t.shape[0], 12, x_t.shape[2])
        if not torch.is_tensor(model_output) or tuple(model_output.shape) != expected_shape:
            raise ValueError("predictor output must have shape [B, 12, K]")
        if not model_output.is_floating_point() or model_output.device != x_t.device:
            raise ValueError("predictor output must be floating-point and on x_t's device")
        if model_output.dtype != x_t.dtype:
            raise ValueError("predictor output must have the same dtype as x_t")

        if noise is not None and (
                not torch.is_tensor(noise) or noise.shape != x_t.shape
                or noise.dtype != x_t.dtype or noise.device != x_t.device):
            raise ValueError("noise must match x_t's shape, dtype, and device")
        noise = torch.randn_like(x_t) if noise is None else noise
        epsilon_hat, variance_values = model_output.chunk(2, dim=1)

        alpha_bar = self.alpha_bar.to(device=x_t.device, dtype=x_t.dtype)
        alpha_bar_t = alpha_bar.index_select(0, timesteps).reshape(-1, 1, 1)
        reciprocal_alpha_bar = torch.rsqrt(alpha_bar_t)
        x0_hat = reciprocal_alpha_bar * x_t - torch.sqrt(reciprocal_alpha_bar.square() - 1.0) * epsilon_hat
        if self.clip_denoised:
            x0_hat = x0_hat.clamp(-1.0, 1.0)

        coefficient_x0 = self.posterior_mean_x0.to(device=x_t.device, dtype=x_t.dtype).index_select(
            0, timesteps
        ).reshape(-1, 1, 1)
        coefficient_xt = self.posterior_mean_xt.to(device=x_t.device, dtype=x_t.dtype).index_select(
            0, timesteps
        ).reshape(-1, 1, 1)
        posterior_mean = coefficient_x0 * x0_hat + coefficient_xt * x_t

        minimum_log_variance = self.posterior_log_variance_clipped.to(
            device=x_t.device, dtype=x_t.dtype
        ).index_select(0, timesteps).reshape(-1, 1, 1)
        maximum_log_variance = torch.log(
            self.betas.to(device=x_t.device, dtype=x_t.dtype)
        ).index_select(0, timesteps).reshape(-1, 1, 1)
        variance_fraction = (variance_values + 1.0) / 2.0
        log_variance = variance_fraction * maximum_log_variance + (
            1.0 - variance_fraction
        ) * minimum_log_variance
        nonzero_timestep = (timesteps != 0).to(dtype=x_t.dtype).reshape(-1, 1, 1)
        x_prev = posterior_mean + nonzero_timestep * torch.exp(0.5 * log_variance) * noise
        if return_x0:
            return x_prev, x0_hat
        return x_prev
