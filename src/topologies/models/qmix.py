import torch
from torch import nn


class QMIX(nn.Module):
    architecture_id = "qmix"
    PRESETS = {
        "tiny": {"n_agents": 2, "state_dim": 4, "mixing_embed_dim": 4},
        "small": {"n_agents": 8, "state_dim": 32, "mixing_embed_dim": 32},
    }

    def __init__(self, n_agents, state_dim, mixing_embed_dim=32):
        super().__init__()
        if isinstance(n_agents, bool) or not isinstance(n_agents, int) or n_agents <= 0:
            raise ValueError("n_agents must be a positive integer")
        if isinstance(state_dim, bool) or not isinstance(state_dim, int) or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        if (isinstance(mixing_embed_dim, bool) or not isinstance(mixing_embed_dim, int)
                or mixing_embed_dim <= 0):
            raise ValueError("mixing_embed_dim must be a positive integer")
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim
        self.hyper_w1 = nn.Linear(state_dim, n_agents * mixing_embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.elu = nn.ELU()

    def forward(self, agent_qs, state):
        if not isinstance(agent_qs, torch.Tensor) or not isinstance(state, torch.Tensor):
            raise TypeError("agent_qs and state must be torch tensors")
        if agent_qs.ndim < 1 or state.ndim < 1:
            raise ValueError("agent_qs and state must each have a feature dimension")
        if agent_qs.shape[-1] != self.n_agents:
            raise ValueError(f"agent_qs last dimension must be {self.n_agents}")
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"state last dimension must be {self.state_dim}")
        if agent_qs.shape[:-1] != state.shape[:-1]:
            raise ValueError("agent_qs and state leading dimensions must match")
        if not agent_qs.is_floating_point() or not state.is_floating_point():
            raise ValueError("agent_qs and state must have floating-point dtypes")

        leading_shape = state.shape[:-1]
        w1 = self.hyper_w1(state).abs().reshape(
            *leading_shape, self.n_agents, self.mixing_embed_dim
        )
        b1 = self.hyper_b1(state).unsqueeze(-2)
        w2 = self.hyper_w2(state).abs().reshape(
            *leading_shape, self.mixing_embed_dim, 1
        )
        b2 = self.hyper_b2(state).unsqueeze(-2)
        hidden = self.elu(torch.matmul(agent_qs.unsqueeze(-2), w1) + b1)
        return (torch.matmul(hidden, w2) + b2).squeeze(-2)
