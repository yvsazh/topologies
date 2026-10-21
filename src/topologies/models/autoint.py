import torch
from torch import nn


class AutoIntLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.residual = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, fields):
        batch, count, width = fields.shape
        shape = (batch, count, self.num_heads, self.head_dim)
        query = self.query(fields).reshape(shape).transpose(1, 2)
        key = self.key(fields).reshape(shape).transpose(1, 2)
        value = self.value(fields).reshape(shape).transpose(1, 2)
        scores = query @ key.transpose(-1, -2)
        attention = torch.softmax(scores, dim=-1)
        attended = (attention @ value).transpose(1, 2).contiguous().reshape(batch, count, width)
        return torch.relu(attended + self.residual(fields))


class AutoInt(nn.Module):
    architecture_id = "autoint"

    def __init__(self, num_fields=4, embedding_dim=8, num_heads=2, num_layers=2, output_dim=1):
        super().__init__()
        if min(num_fields, embedding_dim, num_heads, num_layers, output_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.interactions = nn.ModuleList(
            AutoIntLayer(embedding_dim, num_heads) for _ in range(num_layers)
        )
        self.scoring = nn.Linear(num_fields * embedding_dim, output_dim)

    def forward(self, fields):
        if not isinstance(fields, torch.Tensor) or fields.ndim != 3:
            raise ValueError("fields must have shape (batch, fields, embedding_dim)")
        if fields.shape[1] != self.num_fields or fields.shape[2] != self.embedding_dim:
            raise ValueError("field dimensions do not match constructor")
        for layer in self.interactions:
            fields = layer(fields)
        return torch.sigmoid(self.scoring(fields.reshape(fields.shape[0], -1)))
