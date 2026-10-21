import torch
from torch import nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    component_id = "mamba-block"
    def __init__(self, embed_dim=4, state_dim=2):
        super().__init__()
        if embed_dim < 1 or state_dim < 1:
            raise ValueError("embed_dim and state_dim must be positive")
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.delta_projection = nn.Linear(embed_dim, embed_dim)
        self.input_projection = nn.Linear(embed_dim, state_dim)
        self.output_projection = nn.Linear(embed_dim, state_dim)
        self.log_A = nn.Parameter(torch.zeros(embed_dim, state_dim))

    def forward(self, x):
        if x.ndim < 2:
            raise ValueError("x must include sequence and embedding dimensions")
        if x.shape[-1] != self.embed_dim:
            raise ValueError(f"expected final dimension {self.embed_dim}, got {x.shape[-1]}")
        delta = F.softplus(self.delta_projection(x))
        b_value = self.input_projection(x)
        c_value = self.output_projection(x)
        a_value = -torch.exp(self.log_A)
        state = x.new_zeros(*x.shape[:-2], self.embed_dim, self.state_dim)
        outputs = []
        for index in range(x.shape[-2]):
            token = x[..., index, :]
            dt = delta[..., index, :].unsqueeze(-1)
            b = b_value[..., index, :].unsqueeze(-2)
            c = c_value[..., index, :].unsqueeze(-2)
            state = torch.exp(dt * a_value) * state + dt * b * token.unsqueeze(-1)
            outputs.append((c * state).sum(dim=-1))
        if not outputs:
            return x.new_empty(*x.shape)
        return torch.stack(outputs, dim=-2)
