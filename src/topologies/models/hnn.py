import torch
from torch import nn


class HNN(nn.Module):
    architecture_id = "hnn"
    component_id = "hnn"

    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        if input_dim <= 0 or input_dim % 2:
            raise ValueError("input_dim must be a positive even integer")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.input_dim = input_dim
        self.hamiltonian = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        parameter = next(self.hamiltonian.parameters(), None)
        if parameter is None:
            state = torch.as_tensor(state, dtype=torch.get_default_dtype())
        else:
            state = torch.as_tensor(state, dtype=parameter.dtype, device=parameter.device)
        if state.shape[-1] != self.input_dim:
            raise ValueError("last state dimension must equal input_dim")
        with torch.enable_grad():
            if not state.requires_grad:
                state = state.detach().requires_grad_(True)
            energy = self.hamiltonian(state)
            gradient = torch.autograd.grad(energy.sum(), state, create_graph=True)[0]
        half = self.input_dim // 2
        return torch.cat((gradient[..., half:], -gradient[..., :half]), dim=-1)
