import torch
from torch import nn


class DroQ(nn.Module):
    architecture_id = "droq"
    def __init__(self, observation_features, action_features, hidden_features=256, dropout=0.1, ensemble_size=2):
        super().__init__()
        if observation_features <= 0 or action_features <= 0 or hidden_features <= 0:
            raise ValueError("feature dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if ensemble_size < 2:
            raise ValueError("ensemble_size must be at least 2")
        self.observation_features = observation_features
        self.action_features = action_features
        self.ensemble_size = ensemble_size
        self.critics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(observation_features + action_features, hidden_features),
                nn.LayerNorm(hidden_features),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_features, hidden_features),
                nn.LayerNorm(hidden_features),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_features, 1),
            )
            for _ in range(ensemble_size)
        ])

    def forward(self, observation, action):
        if observation.shape[-1] != self.observation_features or action.shape[-1] != self.action_features:
            raise ValueError("observation or action feature dimension does not match constructor")
        joined = torch.cat((observation, action), dim=-1)
        return torch.stack([critic(joined).squeeze(-1) for critic in self.critics], dim=0)

    def target(self, observation, action, critic_indices=None):
        values = self(observation, action)
        if critic_indices is None:
            indices = torch.randperm(self.ensemble_size, device=values.device)[:2]
        else:
            indices = torch.as_tensor(critic_indices, device=values.device, dtype=torch.long)
            if indices.numel() != 2 or torch.unique(indices).numel() != 2:
                raise ValueError("critic_indices must select two distinct critics")
            if torch.any(indices < 0) or torch.any(indices >= self.ensemble_size):
                raise ValueError("critic index out of range")
        return values.index_select(0, indices).min(dim=0).values
