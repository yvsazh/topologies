import torch
from torch import nn


class xDeepFM(nn.Module):
    architecture_id = "xdeepfm"
    def __init__(self, embedding_dim, field_count, cin_layers=(8, 8), deep_layers=(16, 8), output_dim=1, dropout=0.0):
        super().__init__()
        if embedding_dim <= 0 or field_count < 2 or output_dim <= 0:
            raise ValueError("embedding_dim and output_dim must be positive and field_count must be at least two")
        if not cin_layers or any(width <= 0 for width in cin_layers) or any(width <= 0 for width in deep_layers):
            raise ValueError("layer widths must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.embedding_dim = embedding_dim
        self.field_count = field_count
        self.cin_layers = tuple(cin_layers)
        previous = field_count
        self.cin = nn.ModuleList()
        for width in self.cin_layers:
            self.cin.append(nn.Conv1d(field_count * previous, width, kernel_size=1))
            previous = width
        self.cin_projection = nn.Linear(sum(self.cin_layers), output_dim)
        self.linear_projection = nn.Linear(field_count * embedding_dim, output_dim, bias=False)
        deep = []
        width = field_count * embedding_dim
        for layer_width in deep_layers:
            deep.extend((nn.Linear(width, layer_width), nn.ReLU(), nn.Dropout(dropout)))
            width = layer_width
        self.deep_network = nn.Sequential(*deep) if deep else nn.Identity()
        self.deep_projection = nn.Linear(width, output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, embeddings):
        if embeddings.ndim != 3 or embeddings.shape[1:] != (self.field_count, self.embedding_dim):
            raise ValueError("embeddings must have shape (batch, field_count, embedding_dim)")
        base = embeddings
        hidden = embeddings
        pooled = []
        for layer in self.cin:
            interactions = torch.einsum("bfd,bhd->bfhd", base, hidden)
            interactions = interactions.reshape(embeddings.shape[0], self.field_count * hidden.shape[1], self.embedding_dim)
            hidden = torch.relu(layer(interactions))
            pooled.append(hidden.sum(dim=-1))
        cin_output = self.cin_projection(torch.cat(pooled, dim=-1))
        flattened = embeddings.reshape(embeddings.shape[0], -1)
        linear_output = self.linear_projection(flattened)
        deep_output = self.deep_projection(self.deep_network(flattened))
        return linear_output + cin_output + deep_output + self.bias
