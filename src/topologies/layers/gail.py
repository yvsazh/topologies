import torch
from torch import nn
from torch.nn import functional as F


class GAILReward(nn.Module):
    component_id = "gail"
    def __init__(self, features=4, actions=2, hidden_features=8):
        super().__init__()
        if features <= 0 or actions <= 0 or hidden_features <= 0:
            raise ValueError("features, actions, and hidden_features must be positive")
        self.features = features
        self.actions = actions
        self.discriminator = nn.Sequential(
            nn.Linear(features + actions, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, 1),
        )

    def discriminator_logit(self, states, actions):
        if states.shape[:-1] != actions.shape[:-1]:
            raise ValueError("states and actions must have matching batch dimensions")
        if states.shape[-1] != self.features or actions.shape[-1] != self.actions:
            raise ValueError("state or action feature dimension does not match constructor")
        return self.discriminator(torch.cat((states, actions), dim=-1)).squeeze(-1)

    def forward(self, states, actions):
        return F.softplus(-self.discriminator_logit(states, actions))

    def reward(self, states, actions):
        return self(states, actions)
