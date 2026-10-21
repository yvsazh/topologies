import torch
from torch import nn


def euler_lagrange_residual(lagrangian, q, velocity, acceleration):
    dL_dq = torch.autograd.grad(lagrangian.sum(), q, create_graph=True, retain_graph=True)[0]
    dL_dv = torch.autograd.grad(lagrangian.sum(), velocity, create_graph=True, retain_graph=True)[0]
    total = torch.zeros_like(dL_dv)
    for index in range(dL_dv.shape[-1]):
        derivative = torch.autograd.grad(dL_dv[..., index].sum(), q, create_graph=True, retain_graph=True)[0]
        total = total + derivative * velocity[..., index:index + 1]
        derivative = torch.autograd.grad(dL_dv[..., index].sum(), velocity, create_graph=True, retain_graph=True)[0]
        total = total + derivative * acceleration[..., index:index + 1]
    return total - dL_dq


class LNN(nn.Module):
    architecture_id = "lnn"
    component_id = "lnn"

    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("dimensions must be positive")
        self.input_dim = input_dim
        self.network = nn.Sequential(nn.Linear(2 * input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def lagrangian(self, q, velocity):
        return self.network(torch.cat((q, velocity), dim=-1))

    def forward(self, q, velocity=None, acceleration=None):
        if velocity is None:
            if q.shape[-1] != 2 * self.input_dim:
                raise ValueError("combined input must contain q and velocity")
            q, velocity = q.split(self.input_dim, dim=-1)
        if q.shape != velocity.shape or q.shape[-1] != self.input_dim:
            raise ValueError("state tensors must have matching shape ending in input_dim")
        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            velocity = velocity.detach().requires_grad_(True)
            lagrangian = self.lagrangian(q, velocity)
            dL_dq = torch.autograd.grad(lagrangian.sum(), q, create_graph=True, retain_graph=True)[0]
            dL_dv = torch.autograd.grad(lagrangian.sum(), velocity, create_graph=True, retain_graph=True)[0]
            mixed = torch.zeros_like(q)
            mass = []
            for index in range(self.input_dim):
                mixed_derivative = torch.autograd.grad(
                    dL_dv[..., index].sum(), q, create_graph=True, retain_graph=True,
                    allow_unused=True,
                )[0]
                if mixed_derivative is None:
                    mixed_derivative = torch.zeros_like(q)
                mixed = mixed + mixed_derivative * velocity[..., index:index + 1]
                row = torch.autograd.grad(
                    dL_dv[..., index].sum(), velocity, create_graph=True, retain_graph=True
                )[0]
                mass.append(row)
            matrix = torch.stack(mass, dim=-2)
            return torch.linalg.solve(matrix, (dL_dq - mixed).unsqueeze(-1)).squeeze(-1)
