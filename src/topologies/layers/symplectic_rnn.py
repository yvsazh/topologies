import math

import torch
from torch import nn


class SymplecticRNN(nn.Module):
    component_id = "symplectic-rnn"
    PRESETS = {
        "tiny": {
            "phase_dim": 2,
            "hidden_dim": 32,
            "num_hidden_layers": 1,
            "dt": 0.1,
            "steps": 1,
        },
        "small": {
            "phase_dim": 6,
            "hidden_dim": 512,
            "num_hidden_layers": 3,
            "dt": 1.0,
            "steps": 4,
        },
        "base": {
            "phase_dim": 20,
            "hidden_dim": 2048,
            "num_hidden_layers": 1,
            "dt": 0.1,
            "steps": 9,
        },
    }

    def __init__(
        self,
        phase_dim,
        hidden_dim=32,
        num_hidden_layers=1,
        dt=0.1,
        steps=1,
        hidden_activation="tanh",
    ):
        super().__init__()
        for name, value in (
            ("phase_dim", phase_dim),
            ("hidden_dim", hidden_dim),
            ("num_hidden_layers", num_hidden_layers),
            ("steps", steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be a finite positive number")
        activations = {"tanh": nn.Tanh, "relu": nn.ReLU}
        if not isinstance(hidden_activation, str) or hidden_activation not in activations:
            raise ValueError("hidden_activation must be 'tanh' or 'relu'")
        self.phase_dim = phase_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dt = float(dt)
        self.steps = steps
        self.hidden_activation = hidden_activation
        self.kinetic_net = self._energy_network(activations[hidden_activation])
        self.potential_net = self._energy_network(activations[hidden_activation])

    def _energy_network(self, activation):
        layers = []
        input_dim = self.phase_dim
        for _ in range(self.num_hidden_layers):
            layers.extend((nn.Linear(input_dim, self.hidden_dim), activation()))
            input_dim = self.hidden_dim
        layers.append(nn.Linear(input_dim, 1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            raise TypeError("state must be a tensor shaped [B, 2 * phase_dim]")
        if state.ndim != 2 or state.shape[-1] != 2 * self.phase_dim or state.shape[0] <= 0:
            raise ValueError(
                "last dimension must equal 2 * phase_dim and state must have positive B"
            )
        parameter = next(self.parameters(), None)
        if parameter is not None:
            state = state.to(device=parameter.device, dtype=parameter.dtype)
        track_gradients = torch.is_grad_enabled()
        with torch.enable_grad():
            if not state.requires_grad:
                state = state.detach().requires_grad_(True)
            q, p = state.split(self.phase_dim, dim=-1)
            trajectory = []
            for _ in range(self.steps):
                potential_gradient = torch.autograd.grad(
                    self.potential_net(q).sum(), q, create_graph=True
                )[0]
                p_half = p - (0.5 * self.dt) * potential_gradient
                kinetic_gradient = torch.autograd.grad(
                    self.kinetic_net(p_half).sum(), p_half, create_graph=True
                )[0]
                q = q + self.dt * kinetic_gradient
                potential_gradient = torch.autograd.grad(
                    self.potential_net(q).sum(), q, create_graph=True
                )[0]
                p = p_half - (0.5 * self.dt) * potential_gradient
                trajectory.append(torch.cat((q, p), dim=-1))
        result = torch.stack(trajectory, dim=1)
        return result if track_gradients else result.detach()
