import torch
from torch import nn


def pairwise_inner(fields):
    if fields.ndim < 3:
        raise ValueError("fields must have shape (..., fields, embedding_dim)")
    return torch.einsum("...fd,...gd->...fg", fields, fields)


class PNN(nn.Module):
    architecture_id = "pnn"
    def __init__(self, num_fields, embedding_dim, hidden_dims=(16,), output_dim=1, dropout=0.0):
        super().__init__()
        if num_fields < 2 or embedding_dim < 1 or output_dim < 1:
            raise ValueError("num_fields must be at least 2 and dimensions must be positive")
        if any(width < 1 for width in hidden_dims):
            raise ValueError("hidden dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        product_count = num_fields * (num_fields - 1) // 2
        widths = [num_fields * embedding_dim + product_count, *hidden_dims, output_dim]
        layers = []
        for index in range(len(widths) - 1):
            layers.append(nn.Linear(widths[index], widths[index + 1]))
            if index < len(widths) - 2:
                layers.append(nn.ReLU())
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("pair_i", torch.triu_indices(num_fields, num_fields, offset=1)[0])
        self.register_buffer("pair_j", torch.triu_indices(num_fields, num_fields, offset=1)[1])

    def forward(self, fields):
        if fields.ndim != 3 or fields.shape[1:] != (self.num_fields, self.embedding_dim):
            raise ValueError("fields must have shape (batch, num_fields, embedding_dim)")
        products = pairwise_inner(fields)[:, self.pair_i, self.pair_j]
        joined = torch.cat((fields.flatten(start_dim=1), products), dim=1)
        return self.mlp(joined)
