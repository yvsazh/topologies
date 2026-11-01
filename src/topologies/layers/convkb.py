import torch
from torch import nn


class ConvKB(nn.Module):
    component_id = "convkb"

    def __init__(self, embedding_dim, num_filters):
        super().__init__()
        if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool) or embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer")
        if not isinstance(num_filters, int) or isinstance(num_filters, bool) or num_filters < 1:
            raise ValueError("num_filters must be a positive integer")
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.conv = nn.Conv2d(1, num_filters, kernel_size=(1, 3), stride=1, padding=0, bias=True)
        self.output = nn.Linear(num_filters * embedding_dim, 1, bias=False)

    def forward(self, head, relation, tail):
        embeddings = (("head", head), ("relation", relation), ("tail", tail))
        for name, embedding in embeddings:
            if not isinstance(embedding, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if embedding.ndim != 2:
                raise ValueError(f"{name} must have shape [batch, embedding_dim]")
            if embedding.shape[1] != self.embedding_dim:
                raise ValueError(f"{name} width must match embedding_dim={self.embedding_dim}")
            if not embedding.is_floating_point():
                raise ValueError(f"{name} must use a floating-point dtype")
        if head.shape[0] != relation.shape[0] or head.shape[0] != tail.shape[0]:
            raise ValueError("head, relation, and tail must have the same batch size")
        if head.dtype != relation.dtype or head.dtype != tail.dtype:
            raise ValueError("head, relation, and tail must have the same dtype")
        if head.device != relation.device or head.device != tail.device:
            raise ValueError("head, relation, and tail must be on the same device")
        triples = torch.stack((head, relation, tail), dim=-1).unsqueeze(1)
        features = torch.relu(self.conv(triples)).squeeze(-1).flatten(start_dim=1)
        return self.output(features).squeeze(-1)
