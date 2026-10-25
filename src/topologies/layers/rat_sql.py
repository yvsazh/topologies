import math

import torch
from torch import nn


class RATSQLLayer(nn.Module):
    component_id = "rat-sql"

    def __init__(self, d_model=256, num_heads=8, ff_dim=1024, layer_norm_eps=1e-5):
        super().__init__()
        for name, value in (("d_model", d_model), ("num_heads", num_heads), ("ff_dim", ff_dim)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if (
            isinstance(layer_norm_eps, bool)
            or not isinstance(layer_norm_eps, (int, float))
            or not math.isfinite(layer_norm_eps)
            or layer_norm_eps <= 0
        ):
            raise ValueError("layer_norm_eps must be a finite positive number")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.attention_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )
        self.feed_forward_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x, relation_vectors):
        if (
            not isinstance(x, torch.Tensor)
            or x.ndim != 3
            or x.shape[0] == 0
            or x.shape[1] == 0
            or x.shape[2] != self.d_model
        ):
            raise ValueError("x must have shape (batch, nonempty_sequence, d_model)")
        if not x.is_floating_point():
            raise ValueError("x must have a floating-point dtype")
        expected_relation_shape = (
            x.shape[0],
            x.shape[1],
            x.shape[1],
            self.num_heads,
            self.head_dim,
        )
        if not isinstance(relation_vectors, torch.Tensor) or tuple(relation_vectors.shape) != expected_relation_shape:
            raise ValueError(
                "relation_vectors must have shape (batch, sequence, sequence, num_heads, head_dim)"
            )
        if not relation_vectors.is_floating_point():
            raise ValueError("relation_vectors must have a floating-point dtype")
        if relation_vectors.device != x.device or relation_vectors.dtype != x.dtype:
            raise ValueError("relation_vectors must have the same device and dtype as x")
        batch_size, sequence_length, _ = x.shape
        relations = relation_vectors.permute(0, 3, 1, 2, 4)
        queries = self.query(x).reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(x).reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(x).reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (queries.unsqueeze(-2) * (keys.unsqueeze(2) + relations)).sum(dim=-1) * self.scale
        attention = torch.softmax(scores, dim=-1)
        attended = (attention.unsqueeze(-1) * (values.unsqueeze(2) + relations)).sum(dim=-2)
        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, self.d_model)
        normalized = self.attention_norm(x + attended)
        return self.feed_forward_norm(normalized + self.feed_forward(normalized))
