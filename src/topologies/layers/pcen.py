import torch
from torch import nn


class PCEN(nn.Module):
    component_id = "pcen"
    def __init__(self, channels=4, smoothing=0.5, alpha=0.98, delta=2.0, root=0.5, epsilon=1e-6):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError("smoothing must be between zero and one")
        if alpha < 0 or delta < 0 or root <= 0 or epsilon <= 0:
            raise ValueError("alpha and delta must be nonnegative and root and epsilon positive")
        self.channels = channels
        self.smoothing = smoothing
        self.alpha = alpha
        self.delta = delta
        self.root = root
        self.epsilon = epsilon

    def forward(self, values):
        if values.ndim != 3 or values.shape[1] != self.channels:
            raise ValueError("expected input with shape (batch, channels, time)")
        if values.shape[0] == 0 or values.shape[2] == 0:
            raise ValueError("batch and time dimensions must be non-empty")
        if torch.any(values < 0):
            raise ValueError("PCEN input must be nonnegative")
        energy = values
        smoothed = []
        state = energy[..., 0]
        for frame in energy.unbind(dim=-1):
            state = (1.0 - self.smoothing) * state + self.smoothing * frame
            smoothed.append(state)
        smooth = torch.stack(smoothed, dim=-1)
        normalized = energy / (self.epsilon + smooth).pow(self.alpha)
        return (normalized + self.delta).pow(self.root) - self.delta**self.root
