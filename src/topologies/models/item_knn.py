import torch
from torch import nn


class ItemKNN(nn.Module):
    architecture_id = "item-knn"
    def __init__(self, n_items, n_users):
        super().__init__()
        if n_items <= 0 or n_users <= 0:
            raise ValueError("n_items and n_users must be positive")
        self.n_items = n_items
        self.n_users = n_users

    def item_similarity(self, interactions):
        interactions = torch.as_tensor(interactions, dtype=torch.float32)
        if interactions.shape != (self.n_users, self.n_items):
            raise ValueError("interactions must have shape (n_users, n_items)")
        item_vectors = interactions.transpose(0, 1)
        norms = item_vectors.norm(dim=1, keepdim=True)
        normalized = item_vectors / norms.clamp_min(torch.finfo(item_vectors.dtype).eps)
        return normalized @ normalized.transpose(0, 1)

    def forward(self, interactions):
        interactions = torch.as_tensor(interactions, dtype=torch.float32)
        similarity = self.item_similarity(interactions)
        return interactions @ similarity
