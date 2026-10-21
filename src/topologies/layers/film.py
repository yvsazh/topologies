import torch
from torch import nn


class FiLM(nn.Module):
    component_id = "film"

    def __init__(self, channels, condition_dim):
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if not isinstance(condition_dim, int) or isinstance(condition_dim, bool) or condition_dim <= 0:
            raise ValueError("condition_dim must be a positive integer")
        self.channels = channels
        self.condition_dim = condition_dim
        self.generator = nn.Linear(condition_dim, 2 * channels)

    def forward(self, features, condition):
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim < 2:
            raise ValueError("features must have batch and channel dimensions")
        if features.shape[0] <= 0:
            raise ValueError("features batch size must be positive")
        if features.shape[1] != self.channels:
            raise ValueError(f"features channel dimension must be {self.channels}")
        if any(size <= 0 for size in features.shape[2:]):
            raise ValueError("feature dimensions must be positive")
        if not isinstance(condition, torch.Tensor):
            raise TypeError("condition must be a torch.Tensor")
        if condition.ndim != 2:
            raise ValueError("condition must have shape [batch, condition_dim]")
        if condition.shape[0] <= 0:
            raise ValueError("condition batch size must be positive")
        if condition.shape[1] != self.condition_dim:
            raise ValueError(f"condition width must be {self.condition_dim}")
        if condition.shape[0] != features.shape[0]:
            raise ValueError("features and condition must have the same batch size")
        modulation = self.generator(condition)
        gamma, beta = modulation.chunk(2, dim=1)
        shape = (features.shape[0], self.channels) + (1,) * (features.ndim - 2)
        return gamma.reshape(shape) * features + beta.reshape(shape)
