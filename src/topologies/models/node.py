import torch
from torch import nn


class FixedStepNeuralODE(nn.Module):
    architecture_id = "node"
    def __init__(self, dim, hidden):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, dim))

    def forward(self, state, steps, dt):
        for _ in range(steps):
            state = state + dt * self.network(state)
        return state
