import math

import torch
from torch import nn


class AWR(nn.Module):
    architecture_id = "awr"

    def __init__(self, features, actions, hidden_features=256, beta=1.0, max_weight=20.0):
        super().__init__()
        if features <= 0 or actions <= 0 or hidden_features <= 0:
            raise ValueError("feature and action dimensions must be positive")
        if beta <= 0 or max_weight <= 0:
            raise ValueError("beta and max_weight must be positive")
        self.policy = nn.Sequential(
            nn.Linear(features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, actions),
        )
        self.value = nn.Sequential(
            nn.Linear(features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )
        self.beta = float(beta)
        self.max_weight = float(max_weight)

    def forward(self, observations):
        return self.policy(observations)

    def value_estimate(self, observations):
        return self.value(observations).squeeze(-1)

    def weights(self, advantages):
        log_max_weight = math.log(self.max_weight)
        return torch.exp(torch.clamp(advantages / self.beta, max=log_max_weight))

    def policy_loss(self, observations, actions, advantages):
        log_probs = self.policy(observations).log_softmax(dim=-1)
        actions = actions.long().unsqueeze(-1) if actions.ndim == 1 else actions.long()
        selected = log_probs.gather(-1, actions).squeeze(-1)
        return -(self.weights(advantages) * selected).mean()

    def value_loss(self, observations, returns):
        return torch.nn.functional.mse_loss(self.value_estimate(observations), returns)
