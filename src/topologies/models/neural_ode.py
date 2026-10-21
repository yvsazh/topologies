import torch
from torch import nn


class NeuralODE(nn.Module):
    architecture_id = "neural-ode"
    def __init__(self, state_size=4, hidden_size=None, steps=4, step_size=0.25):
        super().__init__()
        if state_size <= 0 or steps <= 0 or step_size <= 0:
            raise ValueError("state_size, steps, and step_size must be positive")
        hidden_size = state_size * 2 if hidden_size is None else hidden_size
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.state_size = state_size
        self.steps = steps
        self.step_size = step_size
        self.derivative = nn.Sequential(nn.Linear(state_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, state_size))

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=self.derivative[0].weight.dtype, device=self.derivative[0].weight.device)
        if state.shape[-1] != self.state_size:
            raise ValueError("last state dimension must match state_size")
        for _ in range(self.steps):
            state = state + self.step_size * self.derivative(state)
        return state
