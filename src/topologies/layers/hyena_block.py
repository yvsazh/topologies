import torch
from torch import nn
from torch.nn import functional as F


class HyenaBlock(nn.Module):
    component_id = "hyena-block"

    def __init__(self, embed_dim=4, filter_length=4):
        super().__init__()
        if embed_dim <= 0 or filter_length <= 0:
            raise ValueError("embed_dim and filter_length must be positive")
        self.embed_dim = embed_dim
        self.filter_length = filter_length
        self.input_projection = nn.Linear(embed_dim, embed_dim)
        self.gate_projection = nn.Linear(embed_dim, embed_dim)
        self.register_buffer("position_frequencies", 2.0 ** torch.arange(4, dtype=torch.float32), persistent=False)
        self.filter_network = nn.Sequential(
            nn.Linear(9, max(embed_dim, 8)),
            nn.SiLU(),
            nn.Linear(max(embed_dim, 8), embed_dim),
        )
        self.output_projection = nn.Linear(embed_dim, embed_dim)

    def implicit_filter(self):
        positions = torch.linspace(
            0.0,
            1.0,
            self.filter_length,
            device=self.position_frequencies.device,
            dtype=self.position_frequencies.dtype,
        )
        angles = 2.0 * torch.pi * positions[:, None] * self.position_frequencies[None, :]
        features = torch.cat((positions[:, None], angles.sin(), angles.cos()), dim=-1)
        parameter = next(self.filter_network.parameters())
        return self.filter_network(features.to(device=parameter.device, dtype=parameter.dtype))

    def causal_convolution(self, values):
        if values.ndim != 3 or values.shape[-1] != self.embed_dim:
            raise ValueError("values must have shape (batch, sequence, embed_dim)")
        if values.shape[1] == 0:
            return values
        signal = values.transpose(1, 2)
        padded = F.pad(signal, (self.filter_length - 1, 0))
        weights = self.implicit_filter().transpose(0, 1).flip(-1).unsqueeze(1)
        return F.conv1d(padded, weights, groups=self.embed_dim).transpose(1, 2)

    def forward(self, inputs):
        filtered = self.causal_convolution(self.input_projection(inputs))
        gate = self.gate_projection(inputs)
        return self.output_projection(filtered * gate)
