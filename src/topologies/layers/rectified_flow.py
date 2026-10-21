from __future__ import annotations

import torch
from torch import nn


class RectifiedFlowPath(nn.Module):
    component_id = "rectified-flow"

    def __init__(self, features: int, step_size: float = 0.25) -> None:
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive")
        if not 0.0 < step_size <= 1.0:
            raise ValueError("step_size must be in (0, 1]")
        self.features = features
        self.step_size = step_size

    def interpolate(
        self, noise: torch.Tensor, data: torch.Tensor, time: torch.Tensor | float
    ) -> torch.Tensor:
        self._validate_pair(noise, data)
        time = torch.as_tensor(time, dtype=noise.dtype, device=noise.device)
        while time.ndim < noise.ndim:
            time = time.unsqueeze(-1)
        return (1 - time) * noise + time * data

    def target_velocity(self, noise: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        self._validate_pair(noise, data)
        return data - noise

    def flow_matching_loss(
        self,
        predicted_velocity: torch.Tensor,
        noise: torch.Tensor,
        data: torch.Tensor,
    ) -> torch.Tensor:
        target = self.target_velocity(noise, data)
        if predicted_velocity.shape != target.shape:
            raise ValueError("predicted_velocity must match the paired endpoints")
        squared_error = (predicted_velocity - target).square()
        return squared_error.flatten(1).sum(dim=1).mean()

    def euler_step(self, state: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        self._validate_features(state)
        if state.shape != velocity.shape:
            raise ValueError("state and velocity must have identical shapes")
        return state + self.step_size * velocity

    def forward(
        self, noise: torch.Tensor, data: torch.Tensor, time: torch.Tensor | float = 0.0
    ) -> torch.Tensor:
        return self.interpolate(noise, data, time)

    def _validate_pair(self, noise: torch.Tensor, data: torch.Tensor) -> None:
        self._validate_features(noise)
        self._validate_features(data)
        if noise.shape != data.shape:
            raise ValueError("noise and data must have identical shapes")

    def _validate_features(self, value: torch.Tensor) -> None:
        if value.ndim == 0 or value.shape[-1] != self.features:
            raise ValueError(f"last dimension must have {self.features} features")
