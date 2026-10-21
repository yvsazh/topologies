import torch
from torch import nn


class PINN(nn.Module):
    architecture_id = "pinn"
    component_id = "pinn"

    def __init__(self, input_dim=1, output_dim=1, width=32, depth=3, activation=nn.Tanh):
        super().__init__()
        if input_dim < 1 or output_dim < 1 or width < 1 or depth < 1:
            raise ValueError("dimensions, width, and depth must be positive")
        layers = [nn.Linear(input_dim, width), activation()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), activation()))
        layers.append(nn.Linear(width, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, coordinates):
        return self.network(coordinates)

    def loss(self, collocation_coordinates, residual_fn, boundary_coordinates, boundary_values, boundary_weight=1.0):
        coordinates = collocation_coordinates.detach().clone().requires_grad_(True)
        field = self(coordinates)
        residual = residual_fn(coordinates, field)
        if not torch.is_tensor(residual):
            raise TypeError("residual_fn must return a tensor")
        boundary_error = self(boundary_coordinates) - boundary_values
        return residual.square().mean() + boundary_weight * boundary_error.square().mean()
