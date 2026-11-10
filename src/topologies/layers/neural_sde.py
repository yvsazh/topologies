import math

import torch
from torch import nn


class NeuralSDEStep(nn.Module):
    component_id = "neural-sde"

    def __init__(self, dimension, step_size):
        super().__init__()
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if (
            isinstance(step_size, bool)
            or not isinstance(step_size, (int, float))
            or not math.isfinite(step_size)
            or step_size <= 0
        ):
            raise ValueError("step_size must be a positive finite number")
        self.dimension = dimension
        self.step_size = float(step_size)
        self.drift = nn.Linear(dimension, dimension, bias=False)
        self.diffusion = nn.Linear(dimension, dimension, bias=False)

    def forward(self, state, noise):
        if not isinstance(state, torch.Tensor):
            raise TypeError("state must be a torch.Tensor")
        if not isinstance(noise, torch.Tensor):
            raise TypeError("noise must be a torch.Tensor")
        if state.ndim < 1 or noise.ndim < 1:
            raise ValueError("state and noise must have at least one dimension")
        if state.shape != noise.shape:
            raise ValueError("state and noise must have the same shape")
        if state.shape[-1] != self.dimension:
            raise ValueError(f"state and noise final dimension must equal {self.dimension}")
        return (
            state
            + self.step_size * self.drift(state)
            + math.sqrt(self.step_size) * self.diffusion(state) * noise
        )
