import torch
from torch import nn
from torch.nn import functional as F


class RWKVChannelMix(nn.Module):
    component_id = "rwkv-channel-mix"

    def __init__(self, embed_dim=4):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        self.embed_dim = embed_dim
        self.time_mix_key = nn.Parameter(torch.zeros(embed_dim))
        self.time_mix_receptance = nn.Parameter(torch.zeros(embed_dim))
        self.receptance = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError("input must have shape (batch, sequence, embed_dim)")
        previous = F.pad(x[:, :-1], (0, 0, 1, 0))
        key_mix = torch.sigmoid(self.time_mix_key).view(1, 1, -1)
        receptance_mix = torch.sigmoid(self.time_mix_receptance).view(1, 1, -1)
        key_input = key_mix * x + (1 - key_mix) * previous
        receptance_input = receptance_mix * x + (1 - receptance_mix) * previous
        receptance = torch.sigmoid(self.receptance(receptance_input))
        key = F.relu(self.key(key_input)).square()
        return receptance * self.value(key)
