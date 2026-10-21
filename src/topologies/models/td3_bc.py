import torch
from torch import nn


class TD3BC(nn.Module):
    architecture_id = "td3-bc"

    def __init__(self, features=4, actions=2, hidden_features=8, alpha=2.5):
        super().__init__()
        if features <= 0 or actions <= 0 or hidden_features <= 0:
            raise ValueError("feature and action dimensions must be positive")
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        self.features = features
        self.actions = actions
        self.alpha = float(alpha)
        self.actor = nn.Sequential(
            nn.Linear(features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, actions),
            nn.Tanh(),
        )

    def forward(self, observations):
        return self.actor(observations)

    def cloning_loss(self, observations, dataset_actions):
        return (self(observations) - dataset_actions).square().mean()

    def actor_loss(self, observations, dataset_actions, q_values):
        return self.loss_components(observations, dataset_actions, q_values)["actor_loss"]

    def loss_components(self, observations, dataset_actions, q_values):
        cloning = self.cloning_loss(observations, dataset_actions)
        q_scale = q_values.detach().abs().mean().clamp_min(torch.finfo(q_values.dtype).eps)
        q_loss = -(self.alpha / q_scale) * q_values.mean()
        return {
            "q_loss": q_loss,
            "cloning_loss": cloning,
            "actor_loss": q_loss + cloning,
            "q_scale": q_scale,
        }
