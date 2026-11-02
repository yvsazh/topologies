import math

import torch
import torch.nn.functional as F
from torch import nn


class DeepAR(nn.Module):
    architecture_id = "deepar"
    PRESETS = {
        "tiny": {"feature_size": 0, "hidden_size": 8, "num_layers": 1, "forget_bias": 1.0},
        "small": {"feature_size": 0, "hidden_size": 40, "num_layers": 3, "forget_bias": 1.0},
        "base": {"feature_size": 0, "hidden_size": 120, "num_layers": 3, "forget_bias": 1.0},
    }

    def __init__(self, feature_size=0, hidden_size=40, num_layers=3, forget_bias=1.0):
        super().__init__()
        if isinstance(feature_size, bool) or not isinstance(feature_size, int) or feature_size < 0:
            raise ValueError("feature_size must be a nonnegative integer")
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if isinstance(forget_bias, bool) or not isinstance(forget_bias, (int, float)) or not math.isfinite(forget_bias):
            raise ValueError("forget_bias must be a finite number")
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forget_bias = float(forget_bias)
        self.lstm = nn.LSTM(feature_size + 1, hidden_size, num_layers, batch_first=True)
        self.mu_head = nn.Linear(hidden_size, 1)
        self.sigma_head = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            for layer in range(num_layers):
                input_bias = getattr(self.lstm, f"bias_ih_l{layer}")
                recurrent_bias = getattr(self.lstm, f"bias_hh_l{layer}")
                input_bias.zero_()
                recurrent_bias.zero_()
                input_bias[hidden_size : 2 * hidden_size].fill_(self.forget_bias)

    def _validate_targets(self, targets):
        if not isinstance(targets, torch.Tensor) or targets.ndim != 3 or targets.shape[-1] != 1:
            raise ValueError("targets must have shape [batch, time, 1]")
        if targets.shape[0] <= 0 or targets.shape[1] <= 0 or not targets.is_floating_point():
            raise ValueError("targets must have a nonempty batch and time dimension and a floating dtype")

    def _validate_covariates(self, covariates, batch_size, time_length, name):
        if self.feature_size == 0 and covariates is None:
            return
        if not isinstance(covariates, torch.Tensor) or covariates.ndim != 3:
            raise ValueError(f"{name} must have shape [batch, time, feature_size]")
        if covariates.shape != (batch_size, time_length, self.feature_size):
            raise ValueError(f"{name} must have shape [batch, time, {self.feature_size}]")
        if not covariates.is_floating_point():
            raise ValueError(f"{name} must have a floating dtype")

    def context_scale(self, targets, context_length=None):
        self._validate_targets(targets)
        if context_length is None:
            context_length = targets.shape[1]
        if isinstance(context_length, bool) or not isinstance(context_length, int):
            raise TypeError("context_length must be an integer")
        if context_length <= 0 or context_length > targets.shape[1]:
            raise ValueError("context_length must be between 1 and the sequence length")
        scale = 1.0 + targets[:, :context_length].mean(dim=1, keepdim=True)
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("context-derived scale must be finite and positive")
        return scale

    def gaussian_parameters(self, hidden, scale=None):
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden must have shape [batch, time, hidden_size]")
        mu = self.mu_head(hidden)
        sigma = F.softplus(self.sigma_head(hidden))
        if scale is not None:
            mu = mu * scale
            sigma = sigma * scale
        return mu, sigma

    @staticmethod
    def gaussian_nll(targets, mu, sigma, mask=None):
        if not all(isinstance(value, torch.Tensor) for value in (targets, mu, sigma)):
            raise ValueError("targets, mu, and sigma must be tensors")
        if targets.shape != mu.shape or mu.shape != sigma.shape:
            raise ValueError("targets, mu, and sigma must have matching shapes")
        if (sigma <= 0).any():
            raise ValueError("sigma must be positive")
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise ValueError("mask must be a tensor")
            if mask.shape == targets.shape[:-1]:
                mask = mask.unsqueeze(-1)
            try:
                mask = torch.broadcast_to(mask.to(dtype=torch.bool), targets.shape)
            except RuntimeError as exc:
                raise ValueError("mask must be broadcastable to the target shape") from exc
            targets = torch.where(mask, targets, mu)
        values = 0.5 * ((targets - mu) / sigma).square() + sigma.log() + 0.5 * math.log(2.0 * math.pi)
        if mask is not None:
            values = torch.where(mask, values, torch.zeros_like(values))
        return values.sum()

    def _recurrent_inputs(self, targets, covariates, scale):
        lagged = torch.cat((targets.new_zeros(targets.shape[0], 1, 1), targets[:, :-1]), dim=1) / scale
        if covariates is None:
            return lagged
        return torch.cat((lagged, covariates), dim=-1)

    def _zero_state(self, reference):
        shape = (self.num_layers, reference.shape[0], self.hidden_size)
        return reference.new_zeros(shape), reference.new_zeros(shape)

    def forward(self, targets, covariates=None, context_length=None):
        self._validate_targets(targets)
        self._validate_covariates(covariates, targets.shape[0], targets.shape[1], "covariates")
        scale = self.context_scale(targets, context_length)
        recurrent_inputs = self._recurrent_inputs(targets, covariates, scale)
        hidden, _ = self.lstm(recurrent_inputs, self._zero_state(targets))
        return self.gaussian_parameters(hidden, scale)

    def forecast(
        self,
        context_targets,
        prediction_length,
        context_covariates=None,
        prediction_covariates=None,
        generator=None,
    ):
        self._validate_targets(context_targets)
        if isinstance(prediction_length, bool) or not isinstance(prediction_length, int) or prediction_length <= 0:
            raise ValueError("prediction_length must be a positive integer")
        batch_size = context_targets.shape[0]
        context_length = context_targets.shape[1]
        self._validate_covariates(context_covariates, batch_size, context_length, "context_covariates")
        self._validate_covariates(prediction_covariates, batch_size, prediction_length, "prediction_covariates")
        scale = self.context_scale(context_targets)
        recurrent_inputs = self._recurrent_inputs(context_targets, context_covariates, scale)
        _, state = self.lstm(recurrent_inputs, self._zero_state(context_targets))
        lagged = context_targets[:, -1:] / scale
        samples = []
        for step in range(prediction_length):
            if prediction_covariates is None:
                recurrent_step = lagged
            else:
                recurrent_step = torch.cat((lagged, prediction_covariates[:, step : step + 1]), dim=-1)
            hidden, state = self.lstm(recurrent_step, state)
            mu, sigma = self.gaussian_parameters(hidden, scale)
            noise = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
            sample = mu + sigma * noise
            samples.append(sample)
            lagged = sample / scale
        return torch.cat(samples, dim=1)
