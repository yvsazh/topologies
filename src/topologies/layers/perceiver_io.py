import math

import torch
from torch import nn


class PerceiverIO(nn.Module):
    component_id = "perceiver-io"
    def __init__(self, latent_count, dimension, input_dim=None):
        super().__init__()
        input_dim = dimension if input_dim is None else input_dim
        if latent_count <= 0 or dimension <= 0 or input_dim <= 0:
            raise ValueError("latent_count, dimension, and input_dim must be positive")
        self.latents = nn.Parameter(torch.zeros(latent_count, dimension))
        self.dimension = dimension
        self.input_dim = input_dim
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(input_dim, dimension, bias=False)
        self.value = nn.Linear(input_dim, dimension, bias=False)
        self.output = nn.Linear(dimension, dimension, bias=False)

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1] == 0 or inputs.shape[-1] != self.input_dim:
            raise ValueError("inputs must have shape (batch, nonempty_tokens, input_dim)")
        latent_queries = self.latents.unsqueeze(0).expand(inputs.shape[0], -1, -1)
        queries = self.query(latent_queries)
        keys = self.key(inputs)
        values = self.value(inputs)
        scores = torch.softmax(queries @ keys.transpose(-2, -1) / math.sqrt(self.dimension), dim=-1)
        return self.output(scores @ values)
