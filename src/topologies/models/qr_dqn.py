import math

import torch
from torch import nn


class QRDQN(nn.Module):
    architecture_id = "qr-dqn"
    def __init__(self, features=4, actions=3, quantiles=4, hidden=64, kappa=1.0):
        super().__init__()
        if features <= 0 or actions <= 0 or quantiles <= 0 or hidden <= 0:
            raise ValueError("features, actions, quantiles, and hidden must be positive")
        if not math.isfinite(kappa) or kappa <= 0:
            raise ValueError("kappa must be finite and positive")
        self.features = features
        self.actions = actions
        self.quantiles = quantiles
        self.kappa = float(kappa)
        self.network = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, actions * quantiles))
        levels = (torch.arange(quantiles, dtype=torch.float32) + 0.5) / quantiles
        self.register_buffer("quantile_levels", levels)

    def forward(self, observation):
        if observation.shape[-1] != self.features:
            raise ValueError(f"expected final observation dimension {self.features}")
        return self.network(observation).reshape(*observation.shape[:-1], self.actions, self.quantiles)

    def q_values(self, observation):
        return self(observation).mean(dim=-1)

    def greedy_action(self, observation):
        return self.q_values(observation).argmax(dim=-1)

    def greedy_action_from_quantiles(self, quantile_values):
        if quantile_values.shape[-2:] != (self.actions, self.quantiles):
            raise ValueError("quantile_values must end with action and quantile dimensions")
        return quantile_values.mean(dim=-1).argmax(dim=-1)

    def quantile_huber_loss(self, predictions, targets):
        if predictions.ndim < 1 or predictions.shape[-1] != self.quantiles:
            raise ValueError(f"predictions must have {self.quantiles} quantiles on the final dimension")
        if targets.shape == predictions.shape[:-1]:
            delta = targets.unsqueeze(-1).unsqueeze(-1) - predictions.unsqueeze(-1)
        elif targets.ndim == predictions.ndim and targets.shape[:-1] == predictions.shape[:-1]:
            delta = targets.unsqueeze(-2) - predictions.unsqueeze(-1)
        else:
            raise ValueError("targets must match prediction batch dimensions or provide target quantiles")
        absolute = delta.abs()
        huber = torch.where(absolute <= self.kappa, 0.5 * delta.square(), self.kappa * (absolute - 0.5 * self.kappa))
        quantile_shape = [1] * (delta.ndim - 2) + [self.quantiles, 1]
        levels = self.quantile_levels.to(device=predictions.device, dtype=predictions.dtype).view(quantile_shape)
        weight = (levels - (delta.detach() < 0).to(predictions.dtype)).abs()
        return (weight * huber).mean()

    def pinball_loss(self, predictions, targets):
        return self.quantile_huber_loss(predictions, targets)
