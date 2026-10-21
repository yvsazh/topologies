import math

import torch
from torch import nn


class BCQActionPerturbation(nn.Module):
    component_id = "bcq"

    def __init__(self, features=4, actions=2, hidden_features=8, phi=0.05):
        super().__init__()
        if features <= 0 or actions <= 0 or hidden_features <= 0:
            raise ValueError("features, actions, and hidden_features must be positive")
        phi = float(phi)
        if not math.isfinite(phi) or phi < 0.0:
            raise ValueError("phi must be finite and non-negative")
        self.features = features
        self.actions = actions
        self.phi = phi
        self.perturbation = nn.Sequential(
            nn.Linear(features + actions, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, actions),
        )

    def forward(self, observation, behavior_action):
        self._validate_inputs(observation, behavior_action)
        raw_perturbation = self.perturbation(torch.cat((observation, behavior_action), dim=-1))
        delta = self.phi * torch.tanh(raw_perturbation)
        return behavior_action + delta

    def _validate_inputs(self, observation, action):
        if observation.shape[-1] != self.features:
            raise ValueError(f"observation final dimension must be {self.features}")
        if action.shape[-1] != self.actions:
            raise ValueError(f"action final dimension must be {self.actions}")
        if observation.shape[:-1] != action.shape[:-1]:
            raise ValueError("observation and action batch dimensions must match")
