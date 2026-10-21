import math

import torch
from torch import nn


class ConsistencyModel(nn.Module):
    architecture_id = "consistency-model"
    PRESETS = {
        "tiny": {
            "feature_dim": 8,
            "hidden_dim": 32,
            "depth": 2,
            "sigma_data": 0.5,
            "epsilon": 0.002,
            "max_time": 80.0,
        },
        "small": {
            "feature_dim": 32,
            "hidden_dim": 128,
            "depth": 3,
            "sigma_data": 0.5,
            "epsilon": 0.002,
            "max_time": 80.0,
        },
    }

    def __init__(
        self,
        feature_dim=16,
        hidden_dim=64,
        depth=2,
        sigma_data=0.5,
        epsilon=0.002,
        max_time=80.0,
    ):
        super().__init__()
        for name, value in (
            ("feature_dim", feature_dim),
            ("hidden_dim", hidden_dim),
            ("depth", depth),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name, value in (("sigma_data", sigma_data), ("epsilon", epsilon), ("max_time", max_time)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise self._validation_error(name, "must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if sigma_data <= 0:
            raise ValueError("sigma_data must be positive.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if max_time <= 0:
            raise ValueError("max_time must be positive.")
        if epsilon >= max_time:
            raise ValueError("epsilon must be less than max_time.")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.sigma_data = float(sigma_data)
        self.epsilon = float(epsilon)
        self.max_time = float(max_time)
        layers = []
        in_features = feature_dim + 1
        for _ in range(depth):
            layers.extend((nn.Linear(in_features, hidden_dim), nn.ReLU()))
            in_features = hidden_dim
        layers.append(nn.Linear(hidden_dim, feature_dim))
        self.backbone = nn.Sequential(*layers)

    @staticmethod
    def _validation_error(name, message):
        return ValueError(f"{name} {message}.")

    def forward(self, x, t):
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("x must be a rank-2 tensor with shape [batch, feature_dim].")
        if x.shape[0] <= 0 or x.shape[1] != self.feature_dim:
            raise ValueError(f"x must have shape [batch, {self.feature_dim}] with a non-empty batch.")
        if not x.is_floating_point():
            raise ValueError("x must have a floating-point dtype.")
        try:
            times = torch.as_tensor(t, device=x.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("t must be a scalar or a rank-1 tensor with one time per example.") from exc
        if times.is_complex() or times.dtype == torch.bool:
            raise ValueError("t must contain real numeric times.")
        if times.ndim == 0:
            times = times.expand(x.shape[0])
        elif times.ndim != 1 or times.shape[0] != x.shape[0]:
            raise ValueError("t must be scalar or have shape [batch].")
        times = times.to(dtype=x.dtype)
        if not bool(torch.isfinite(times).all()):
            raise ValueError("t values must be finite.")
        if bool((times < self.epsilon).any()) or bool((times > self.max_time).any()):
            raise ValueError("t values must be within [epsilon, max_time].")
        prediction = self.backbone(torch.cat((x, times.unsqueeze(-1)), dim=-1))
        offset = times - self.epsilon
        sigma_squared = self.sigma_data**2
        c_skip = sigma_squared / (offset.square() + sigma_squared)
        c_out = self.sigma_data * offset / torch.sqrt(sigma_squared + times.square())
        return c_skip.unsqueeze(-1) * x + c_out.unsqueeze(-1) * prediction
