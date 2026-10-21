import torch
from torch import nn


class DecisionTransformer(nn.Module):
    architecture_id = "decision-transformer"
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_heads=4, num_layers=2, max_length=1024):
        super().__init__()
        if min(state_dim, action_dim, hidden_dim, num_heads, num_layers, max_length) <= 0:
            raise ValueError("dimensions and layer counts must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length
        self.return_embedding = nn.Linear(1, hidden_dim)
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        self.return_layer_norm = nn.LayerNorm(hidden_dim)
        self.state_layer_norm = nn.LayerNorm(hidden_dim)
        self.action_layer_norm = nn.LayerNorm(hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=4 * hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, states, returns_to_go, actions=None, timesteps=None):
        states = torch.as_tensor(states, dtype=self.state_embedding.weight.dtype, device=self.state_embedding.weight.device)
        returns_to_go = torch.as_tensor(returns_to_go, dtype=states.dtype, device=states.device)
        if states.ndim != 3 or states.shape[-1] != self.state_dim:
            raise ValueError("states must have shape (batch, time, state_dim)")
        batch, time, _ = states.shape
        if time > self.max_length:
            raise ValueError("sequence exceeds max_length")
        if returns_to_go.ndim == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)
        if returns_to_go.shape != (batch, time, 1):
            raise ValueError("returns_to_go must have shape (batch, time) or (batch, time, 1)")
        if actions is None:
            actions = states.new_zeros(batch, time, self.action_dim)
        else:
            actions = torch.as_tensor(actions, dtype=states.dtype, device=states.device)
        if actions.shape != (batch, time, self.action_dim):
            raise ValueError("actions must have shape (batch, time, action_dim)")
        triplets = torch.stack(
            (
                self.return_layer_norm(self.return_embedding(returns_to_go)),
                self.state_layer_norm(self.state_embedding(states)),
                self.action_layer_norm(self.action_embedding(actions)),
            ),
            dim=2,
        )
        if timesteps is None:
            timestep_ids = torch.arange(time, device=states.device).unsqueeze(0).expand(batch, -1)
        else:
            timestep_ids = torch.as_tensor(timesteps, dtype=torch.long, device=states.device)
            if timestep_ids.ndim == 1 and timestep_ids.shape == (time,):
                timestep_ids = timestep_ids.unsqueeze(0).expand(batch, -1)
            if timestep_ids.shape != (batch, time):
                raise ValueError("timesteps must have shape (time,) or (batch, time)")
            if torch.any(timestep_ids < 0) or torch.any(timestep_ids >= self.max_length):
                raise ValueError("timestep ids must be within [0, max_length)")
        positions = self.position_embedding(timestep_ids)
        tokens = (triplets + positions.unsqueeze(2)).reshape(batch, time * 3, -1)
        mask = torch.triu(torch.ones(time * 3, time * 3, dtype=torch.bool, device=states.device), diagonal=1)
        encoded = self.transformer(tokens, mask=mask).reshape(batch, time, 3, -1)
        return self.action_head(encoded[:, :, 1])
