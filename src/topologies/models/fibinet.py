import torch
from torch import nn


class FiBiNET(nn.Module):
    architecture_id = "fibinet"
    def __init__(self, num_fields, embedding_dim, reduction_ratio=3, bilinear_type="field_all"):
        super().__init__()
        if num_fields < 2 or embedding_dim < 1 or reduction_ratio < 1:
            raise ValueError("num_fields must be at least 2; embedding_dim and reduction_ratio must be positive")
        if bilinear_type not in {"field_all", "field_each", "field_interaction"}:
            raise ValueError("bilinear_type must be field_all, field_each, or field_interaction")
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.bilinear_type = bilinear_type
        self.num_pairs = num_fields * (num_fields - 1) // 2
        hidden = max(1, num_fields // reduction_ratio)
        self.se = nn.Sequential(nn.Linear(num_fields, hidden), nn.ReLU(), nn.Linear(hidden, num_fields), nn.Sigmoid())
        if bilinear_type == "field_all":
            self.bilinear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        elif bilinear_type == "field_each":
            self.bilinear = nn.ModuleList(nn.Linear(embedding_dim, embedding_dim, bias=False) for _ in range(num_fields))
        else:
            self.bilinear = nn.ModuleList(nn.Linear(embedding_dim, embedding_dim, bias=False) for _ in range(self.num_pairs))
        pairs = torch.triu_indices(num_fields, num_fields, offset=1)
        self.register_buffer("pair_left", pairs[0], persistent=False)
        self.register_buffer("pair_right", pairs[1], persistent=False)

    def _interactions(self, fields):
        left = fields.index_select(1, self.pair_left)
        right = fields.index_select(1, self.pair_right)
        if self.bilinear_type == "field_all":
            projected = self.bilinear(left)
        elif self.bilinear_type == "field_each":
            projected = torch.stack(
                [self.bilinear[int(field_index)](left[:, pair_index])
                 for pair_index, field_index in enumerate(self.pair_left)],
                dim=1,
            )
        else:
            projected = torch.stack(
                [layer(left[:, pair_index]) for pair_index, layer in enumerate(self.bilinear)],
                dim=1,
            )
        return (projected * right).reshape(fields.shape[0], -1)

    def forward(self, fields):
        if fields.ndim != 3 or fields.shape[1:] != (self.num_fields, self.embedding_dim):
            raise ValueError(f"fields must have shape (batch, {self.num_fields}, {self.embedding_dim})")
        squeeze = fields.mean(dim=-1)
        gates = self.se(squeeze).unsqueeze(-1)
        weighted_fields = fields * gates
        return torch.cat((self._interactions(fields), self._interactions(weighted_fields)), dim=-1)
