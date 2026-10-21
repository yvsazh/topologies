import torch
from torch import nn


class LarsStep(nn.Module):
    component_id = "lars"
    def __init__(self, weight_decay=0.0, eps=1e-8, eta=1e-3):
        super().__init__()
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if eta <= 0:
            raise ValueError("eta must be positive")
        self.weight_decay = weight_decay
        self.eps = eps
        self.eta = eta

    def forward(self, parameter, gradient, lr):
        if parameter.shape != gradient.shape:
            raise ValueError("parameter and gradient must have matching shapes")
        parameter_norm = torch.linalg.vector_norm(parameter)
        gradient_norm = torch.linalg.vector_norm(gradient)
        trust = self.eta * parameter_norm / (
            gradient_norm + self.weight_decay * parameter_norm + self.eps
        )
        direction = gradient + self.weight_decay * parameter
        return parameter - lr * trust * direction
