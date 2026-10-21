import torch
from torch import nn

class QNetwork(nn.Module):
    architecture_id = 'q-network'

    def __init__(self, observation_dim, num_actions, hidden_dims=(64, 64), dueling=False):
        super().__init__()
        if observation_dim < 1 or num_actions < 2 or (not hidden_dims) or any((w < 1 for w in hidden_dims)):
            raise ValueError('positive dimensions, hidden widths, and at least two actions are required')
        layers, width = ([], observation_dim)
        for size in hidden_dims:
            layers.extend((nn.Linear(width, size), nn.ReLU()))
            width = size
        self.trunk = nn.Sequential(*layers)
        self.dueling = dueling
        self.value = nn.Linear(width, 1) if dueling else None
        self.advantage = nn.Linear(width, num_actions)

    def forward(self, observations):
        features = self.trunk(observations)
        advantage = self.advantage(features)
        return advantage if not self.dueling else self.value(features) + advantage - advantage.mean(dim=-1, keepdim=True)
