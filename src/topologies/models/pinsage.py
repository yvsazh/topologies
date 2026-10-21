import torch
from torch import nn
from torch.nn import functional as F


class PinSAGE(nn.Module):
    architecture_id = "pinsage"

    def __init__(self, embedding_dim, output_dim=None, bias=True):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        output_dim = embedding_dim if output_dim is None else output_dim
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.update = nn.Linear(2 * embedding_dim, output_dim, bias=bias)

    def importance_pool(self, neighbor_embeddings, visit_counts):
        if neighbor_embeddings.ndim != 3:
            raise ValueError("neighbor_embeddings must have shape (batch, neighbors, features)")
        if visit_counts.shape != neighbor_embeddings.shape[:2]:
            raise ValueError("visit_counts must have shape (batch, neighbors)")
        if torch.any(visit_counts < 0):
            raise ValueError("visit_counts must be nonnegative")
        weights = visit_counts.to(device=neighbor_embeddings.device, dtype=neighbor_embeddings.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1)
        return torch.sum(neighbor_embeddings * weights.unsqueeze(-1), dim=1)

    def forward(self, node_embeddings, neighbor_ids, visit_counts, target_ids=None):
        if node_embeddings.ndim != 2 or node_embeddings.shape[1] != self.embedding_dim:
            raise ValueError("node_embeddings must have shape (nodes, embedding_dim)")
        if neighbor_ids.ndim != 2 or visit_counts.shape != neighbor_ids.shape:
            raise ValueError("neighbor_ids and visit_counts must have shape (batch, neighbors)")
        if neighbor_ids.dtype != torch.long:
            raise ValueError("neighbor_ids must use torch.long")
        neighbor_ids = neighbor_ids.to(device=node_embeddings.device)
        if neighbor_ids.numel() and (neighbor_ids.min() < 0 or neighbor_ids.max() >= node_embeddings.shape[0]):
            raise ValueError("neighbor id is outside node_embeddings")
        if target_ids is None:
            target_ids = torch.arange(neighbor_ids.shape[0], device=node_embeddings.device)
        else:
            target_ids = torch.as_tensor(target_ids, device=node_embeddings.device)
            if target_ids.dtype != torch.long:
                raise ValueError("target_ids must use torch.long")
        if target_ids.shape != (neighbor_ids.shape[0],):
            raise ValueError("target_ids must have one node id per neighborhood")
        if target_ids.numel() and (target_ids.min() < 0 or target_ids.max() >= node_embeddings.shape[0]):
            raise ValueError("target id is outside node_embeddings")
        targets = node_embeddings[target_ids]
        neighbors = node_embeddings[neighbor_ids]
        pooled = self.importance_pool(neighbors, visit_counts)
        updated = F.relu(self.update(torch.cat((targets, pooled), dim=-1)))
        return F.normalize(updated, p=2, dim=-1)
