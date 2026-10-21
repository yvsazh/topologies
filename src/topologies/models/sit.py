import torch
from torch import nn


class SiT(nn.Module):
    architecture_id = "sit"
    def __init__(self, features: int):
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive")
        self.features = features

    def interpolate(self, noise: torch.Tensor, data: torch.Tensor, time: torch.Tensor | float) -> torch.Tensor:
        self._validate_pair(noise, data)
        time = torch.as_tensor(time, dtype=noise.dtype, device=noise.device)
        while time.ndim < noise.ndim:
            time = time.unsqueeze(-1)
        return (1 - time) * data + time * noise

    def velocity_target(self, noise: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        self._validate_pair(noise, data)
        return noise - data

    def forward(self, noise: torch.Tensor, data: torch.Tensor, time: torch.Tensor | float = 0.0) -> torch.Tensor:
        return self.interpolate(noise, data, time)

    def _validate_pair(self, noise: torch.Tensor, data: torch.Tensor) -> None:
        if noise.ndim == 0 or data.ndim == 0 or noise.shape[-1] != self.features or data.shape[-1] != self.features:
            raise ValueError(f"last dimension must have {self.features} features")
        if noise.shape != data.shape:
            raise ValueError("noise and data must have identical shapes")
