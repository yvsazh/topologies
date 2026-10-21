import torch
from torch import nn

class DeepFM(nn.Module):
    architecture_id = 'deepfm'

    def __init__(self, field_cardinalities, embedding_dim=8, num_outputs=1, hidden_dims=(32, 16), dropout=0.0):
        super().__init__()
        if not field_cardinalities or any((int(n) < 1 for n in field_cardinalities)):
            raise ValueError('field_cardinalities must contain positive sizes')
        if embedding_dim < 1 or num_outputs < 1 or any((int(n) < 1 for n in hidden_dims)):
            raise ValueError('embedding, output, and hidden dimensions must be positive')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        self.field_cardinalities = tuple(map(int, field_cardinalities))
        self.embeddings = nn.ModuleList((nn.Embedding(n, embedding_dim) for n in self.field_cardinalities))
        self.first_order = nn.ModuleList((nn.Embedding(n, 1) for n in self.field_cardinalities))
        layers = []
        width = len(self.field_cardinalities) * embedding_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)))
            width = hidden
        layers.append(nn.Linear(width, num_outputs))
        self.deep = nn.Sequential(*layers)
        self.linear_bias = nn.Parameter(torch.zeros(num_outputs))
        self.num_outputs = num_outputs

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        if fields.ndim != 2 or fields.shape[1] != len(self.embeddings):
            raise ValueError('fields must have shape [batch, number_of_fields]')
        if fields.dtype not in (torch.int32, torch.int64):
            raise ValueError('categorical fields must be integer IDs')
        vectors = torch.stack([emb(fields[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)
        first = torch.stack([emb(fields[:, i]).squeeze(-1) for i, emb in enumerate(self.first_order)], dim=1).sum(1)
        fm = 0.5 * (vectors.sum(1).square() - vectors.square().sum(1)).sum(-1)
        return (first + fm).unsqueeze(-1) + self.deep(vectors.flatten(1)) + self.linear_bias
