import math

import torch
from torch import nn
from torch.nn import functional as F


class DistributionalQ(nn.Module):
    architecture_id = "distributional-q"
    PRESETS = {
        "tiny": {"features": 4, "actions": 3, "atoms": 5, "v_min": -10.0, "v_max": 10.0, "hidden": 16},
        "small": {"features": 32, "actions": 4, "atoms": 51, "v_min": -10.0, "v_max": 10.0, "hidden": 256},
    }

    def __init__(self, features=4, actions=3, atoms=51, v_min=-10.0, v_max=10.0, hidden=64):
        super().__init__()
        dimensions = {"features": features, "actions": actions, "atoms": atoms, "hidden": hidden}
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if atoms < 2:
            raise ValueError("atoms must be at least two")
        if isinstance(v_min, bool) or not isinstance(v_min, (int, float)) or not math.isfinite(v_min):
            raise ValueError("v_min must be finite")
        if isinstance(v_max, bool) or not isinstance(v_max, (int, float)) or not math.isfinite(v_max):
            raise ValueError("v_max must be finite")
        if v_max <= v_min:
            raise ValueError("v_max must exceed v_min")
        self.features = features
        self.actions = actions
        self.atoms = atoms
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.network = nn.Sequential(
            nn.Linear(features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, actions * atoms),
        )
        self.register_buffer("support", torch.linspace(self.v_min, self.v_max, atoms))

    def forward(self, observation):
        if not isinstance(observation, torch.Tensor):
            raise TypeError("observation must be a torch.Tensor")
        if observation.ndim < 1 or observation.shape[-1] != self.features:
            raise ValueError(f"observation must end with {self.features} features")
        logits = self.network(observation).reshape(*observation.shape[:-1], self.actions, self.atoms)
        return F.softmax(logits, dim=-1)

    def q_values(self, observation):
        probabilities = self(observation)
        support = self.support.to(device=probabilities.device, dtype=probabilities.dtype)
        return (probabilities * support).sum(dim=-1)

    def greedy_action(self, observation):
        return self.q_values(observation).argmax(dim=-1)

    def greedy_action_from_probabilities(self, probabilities):
        self._validate_probabilities(probabilities, "probabilities")
        if probabilities.ndim < 2 or probabilities.shape[-2] != self.actions:
            raise ValueError("probabilities must end with action and atom dimensions")
        support = self.support.to(device=probabilities.device, dtype=probabilities.dtype)
        return (probabilities * support).sum(dim=-1).argmax(dim=-1)

    def project(self, rewards, discounts, next_probabilities):
        self._validate_probabilities(next_probabilities, "next_probabilities")
        expected_shape = next_probabilities.shape[:-1]
        rewards = torch.as_tensor(
            rewards, device=next_probabilities.device, dtype=next_probabilities.dtype
        )
        discounts = torch.as_tensor(
            discounts, device=next_probabilities.device, dtype=next_probabilities.dtype
        )
        if rewards.shape != expected_shape:
            raise ValueError("rewards must match the leading next_probabilities dimensions")
        if discounts.shape != expected_shape:
            raise ValueError("discounts must match the leading next_probabilities dimensions")
        if not torch.isfinite(rewards).all():
            raise ValueError("rewards must be finite")
        if not torch.isfinite(discounts).all() or (discounts < 0).any() or (discounts > 1).any():
            raise ValueError("discounts must be finite and in [0, 1]")
        support = self.support.to(device=next_probabilities.device, dtype=next_probabilities.dtype)
        delta = (self.v_max - self.v_min) / (self.atoms - 1)
        transformed = (
            rewards.unsqueeze(-1) + discounts.unsqueeze(-1) * support
        ).clamp(self.v_min, self.v_max)
        positions = (transformed - self.v_min) / delta
        lower = positions.floor().long()
        upper = positions.ceil().long()
        lower_weight = upper.to(positions.dtype) - positions
        upper_weight = positions - lower.to(positions.dtype)
        same_atom = lower == upper
        lower_weight = torch.where(same_atom, torch.ones_like(lower_weight), lower_weight)
        upper_weight = torch.where(same_atom, torch.zeros_like(upper_weight), upper_weight)
        projected = torch.zeros_like(next_probabilities)
        projected.scatter_add_(-1, lower, next_probabilities * lower_weight)
        projected.scatter_add_(-1, upper, next_probabilities * upper_weight)
        return projected

    def categorical_loss(self, predictions, targets):
        self._validate_probabilities(predictions, "predictions")
        self._validate_probabilities(targets, "targets")
        if predictions.shape != targets.shape:
            raise ValueError("predictions and targets must have the same shape")
        targets = targets.detach().to(device=predictions.device, dtype=predictions.dtype)
        floor = torch.finfo(predictions.dtype).tiny
        return -(targets * predictions.clamp_min(floor).log()).sum(dim=-1).mean()

    def _validate_probabilities(self, probabilities, name):
        if not isinstance(probabilities, torch.Tensor) or probabilities.ndim < 1:
            raise ValueError(f"{name} must be a tensor with a final atom dimension")
        if not probabilities.is_floating_point() or probabilities.shape[-1] != self.atoms:
            raise ValueError(f"{name} must be floating point with {self.atoms} atoms")
        if not torch.isfinite(probabilities).all():
            raise ValueError(f"{name} must be finite")
        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError(f"{name} values must be in [0, 1]")
        total = probabilities.sum(dim=-1)
        if not torch.allclose(total, torch.ones_like(total), atol=1e-5, rtol=1e-5):
            raise ValueError(f"{name} must sum to one over atoms")
