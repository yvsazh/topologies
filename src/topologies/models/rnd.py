import torch
from torch import nn


class RND(nn.Module):
    architecture_id = "rnd"
    def __init__(self, features=4, hidden_features=8):
        super().__init__()
        if features <= 0 or hidden_features <= 0:
            raise ValueError("features and hidden_features must be positive")
        self.target = nn.Sequential(nn.Linear(features, hidden_features), nn.ReLU(), nn.Linear(hidden_features, features))
        self.predictor = nn.Sequential(nn.Linear(features, hidden_features), nn.ReLU(), nn.Linear(hidden_features, features))
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs):
        target_features = self.target(inputs)
        predicted_features = self.predictor(inputs)
        return (predicted_features - target_features).square().mean(dim=-1)
