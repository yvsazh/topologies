import torch
from torch import nn


class BLIP2(nn.Module):
    architecture_id = "blip-2"
    component_id = "blip-2"

    def __init__(self, image_dim=32, hidden_dim=32, num_queries=4):
        super().__init__()
        if min(image_dim, hidden_dim, num_queries) <= 0:
            raise ValueError("dimensions and query count must be positive")
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(image_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(image_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5

    def forward(self, image_tokens, return_attention=False):
        if image_tokens.ndim != 3:
            raise ValueError("image_tokens must have shape (batch, sequence, image_dim)")
        if image_tokens.shape[1] == 0:
            raise ValueError("image_tokens must contain at least one token")
        if image_tokens.shape[-1] != self.image_dim:
            raise ValueError("image token width does not match image_dim")

        queries = self.query_projection(self.query_tokens.expand(image_tokens.shape[0], -1, -1))
        keys = self.key_projection(image_tokens)
        values = self.value_projection(image_tokens)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attention = torch.softmax(scores, dim=-1)
        query_features = torch.matmul(attention, values)
        if return_attention:
            return query_features, attention
        return query_features
