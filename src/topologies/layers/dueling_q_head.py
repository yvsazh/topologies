import torch
from torch import nn

class DuelingQHead(nn.Module):
    component_id = 'dueling-q-head'

    def __init__(self, in_features, num_actions):
        super().__init__()
        if min(in_features, num_actions) <= 0:
            raise ValueError('dimensions must be positive')
        self.value = nn.Linear(in_features, 1)
        self.advantage = nn.Linear(in_features, num_actions)

    def forward(self, x):
        a = self.advantage(x)
        return self.value(x) + a - a.mean(dim=-1, keepdim=True)
