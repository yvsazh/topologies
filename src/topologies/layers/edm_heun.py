import torch
from torch import nn


class EDMHeun(nn.Module):
    component_id = "edm-heun"
    def __init__(self, features=4, step_size=0.1):
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.features = features
        self.step_size = step_size

    def forward(self, values, derivative_fn=None):
        if not torch.is_tensor(values):
            values = torch.as_tensor(values, dtype=torch.float32)
        if values.shape[-1] != self.features:
            raise ValueError("last input dimension must match features")
        if derivative_fn is None:
            derivative_fn = torch.neg
        first = derivative_fn(values)
        proposal = values + self.step_size * first
        second = derivative_fn(proposal)
        return values + (self.step_size * 0.5) * (first + second)
