import torch
from torch import nn


class SlotAttention(nn.Module):


    component_id = "slot-attention"

    def __init__(self, input_dim, slot_dim, num_iterations=3, epsilon=1e-8):
        super().__init__()
        if input_dim <= 0 or slot_dim <= 0 or num_iterations <= 0:
            raise ValueError("input_dim, slot_dim, and num_iterations must be positive")
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(input_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.Linear(slot_dim, slot_dim * 2), nn.ReLU(), nn.Linear(slot_dim * 2, slot_dim))
        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        self.scale = slot_dim ** -0.5
        self.last_attention = None

    def forward(self, inputs, num_slots):
        inputs = torch.as_tensor(inputs, dtype=self.to_k.weight.dtype, device=self.to_k.weight.device)
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError("inputs must have shape (batch, inputs, input_dim)")
        if isinstance(num_slots, torch.Tensor) and num_slots.ndim == 0:
            num_slots = int(num_slots.item())
        if not isinstance(num_slots, int) or isinstance(num_slots, bool):
            raise ValueError("num_slots must be an integer")
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        batch = inputs.shape[0]
        mu = self.slots_mu.expand(batch, num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(batch, num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        keys, values = self.to_k(self.norm_inputs(inputs)), self.to_v(self.norm_inputs(inputs))
        attention = None
        for _ in range(self.num_iterations):
            previous = slots
            queries = self.to_q(self.norm_slots(slots))

            attention = torch.softmax(torch.einsum("bsd,bnd->bsn", queries, keys) * self.scale, dim=1)
            updates = attention / (attention.sum(dim=-1, keepdim=True) + self.epsilon)
            updates = torch.einsum("bsn,bnd->bsd", updates, values)
            slots = self.gru(updates.reshape(-1, self.slot_dim), previous.reshape(-1, self.slot_dim)).reshape(batch, num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))
        self.last_attention = attention
        return slots
