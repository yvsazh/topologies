import torch
from torch import nn


class FFJORD(nn.Module):
    architecture_id = "ffjord"
    def __init__(self, dimension=4, hidden_dim=8, step_size=0.1):
        super().__init__()
        if dimension <= 0 or hidden_dim <= 0 or step_size <= 0:
            raise ValueError("dimension, hidden_dim, and step_size must be positive")
        self.dimension = dimension
        self.step_size = float(step_size)
        self.dynamics = nn.Sequential(
            nn.Linear(dimension, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dimension),
        )

    def forward(self, x, log_density=None, probe=None):
        if x.shape[-1] != self.dimension:
            raise ValueError("last input dimension does not match dimension")
        if not x.is_floating_point():
            raise TypeError("x must be floating point")
        original_shape = x.shape
        flat = x.reshape(-1, self.dimension)
        if probe is None:
            probe = torch.empty_like(flat).bernoulli_(0.5).mul_(2).sub_(1)
        else:
            probe = torch.as_tensor(probe, device=x.device, dtype=x.dtype).expand_as(flat)
        with torch.enable_grad():
            state = flat if flat.requires_grad else flat.detach().requires_grad_(True)
            velocity = self.dynamics(state)
            directional = (velocity * probe).sum()
            jvp = torch.autograd.grad(directional, state, create_graph=True)[0]
            trace_estimate = (jvp * probe).sum(dim=-1)
        transformed = (state + self.step_size * velocity).reshape(original_shape)
        change = (-self.step_size * trace_estimate).reshape(original_shape[:-1])
        if log_density is None:
            return transformed, change
        return transformed, log_density + change

    def inverse(self, y, iterations=100):
        if iterations < 1:
            raise ValueError("iterations must be positive")
        x = y
        for _ in range(iterations):
            x = y - self.step_size * self.dynamics(x)
        return x
