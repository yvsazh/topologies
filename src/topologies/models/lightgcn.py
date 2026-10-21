import torch
from torch import nn

PRESETS = {
    "tiny": {"n_users": 4, "n_items": 6, "embedding_dim": 4, "n_layers": 2},
    "small": {"n_users": 1000, "n_items": 2000, "embedding_dim": 64, "n_layers": 3},
}


class LightGCN(nn.Module):
    architecture_id = "lightgcn"
    PRESETS = PRESETS

    def __init__(self, n_users, n_items, interactions, embedding_dim=64, n_layers=3):
        super().__init__()
        dimensions = {
            "n_users": n_users,
            "n_items": n_items,
            "embedding_dim": embedding_dim,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers < 0:
            raise ValueError("n_layers must be a nonnegative integer")
        try:
            edge_pairs = torch.as_tensor(interactions)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("interactions must be an integer tensor with shape [2, E]") from exc
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if edge_pairs.ndim != 2 or edge_pairs.shape[0] != 2:
            raise ValueError("interactions must have shape [2, E]")
        if edge_pairs.dtype not in integer_types:
            raise ValueError("interactions must contain integer user and item IDs")
        edge_pairs = edge_pairs.to(dtype=torch.long, device="cpu")
        if edge_pairs.numel():
            if bool((edge_pairs[0] < 0).any()) or bool((edge_pairs[0] >= n_users).any()):
                raise ValueError("interaction user IDs must be in [0, n_users)")
            if bool((edge_pairs[1] < 0).any()) or bool((edge_pairs[1] >= n_items).any()):
                raise ValueError("interaction item IDs must be in [0, n_items)")
            if torch.unique(edge_pairs.t(), dim=0).shape[0] != edge_pairs.shape[1]:
                raise ValueError("interactions must contain unique user-item pairs")
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.register_buffer("interactions", edge_pairs.contiguous())

    def encode(self):
        user_ids, item_ids = self.interactions
        item_nodes = item_ids + self.n_users
        node_count = self.n_users + self.n_items
        initial = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_layers = initial
        embeddings = initial
        if user_ids.numel():
            degree = initial.new_zeros(node_count)
            edge_ones = initial.new_ones(user_ids.shape[0])
            degree = degree.index_add(0, user_ids, edge_ones)
            degree = degree.index_add(0, item_nodes, edge_ones)
            edge_weights = (degree[user_ids] * degree[item_nodes]).rsqrt()
        else:
            edge_weights = initial.new_empty((0,))
        for _ in range(self.n_layers):
            messages = initial.new_zeros((node_count, self.embedding_dim))
            messages = messages.index_add(
                0, user_ids, embeddings[item_nodes] * edge_weights.unsqueeze(-1)
            )
            messages = messages.index_add(
                0, item_nodes, embeddings[user_ids] * edge_weights.unsqueeze(-1)
            )
            embeddings = messages
            all_layers = all_layers + embeddings
        combined = all_layers / (self.n_layers + 1)
        return combined[:self.n_users], combined[self.n_users:]

    def forward(self, user_ids, item_ids):
        device = self.user_embedding.weight.device
        try:
            user_ids = torch.as_tensor(user_ids, device=device)
            item_ids = torch.as_tensor(item_ids, device=device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("user_ids and item_ids must be integer ID vectors") from exc
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if user_ids.dtype not in integer_types or item_ids.dtype not in integer_types:
            raise ValueError("user_ids and item_ids must contain integer IDs")
        if user_ids.ndim == 0:
            user_ids = user_ids.reshape(1)
        if item_ids.ndim == 0:
            item_ids = item_ids.reshape(1)
        if user_ids.ndim != 1 or item_ids.ndim != 1:
            raise ValueError("user_ids and item_ids must be one-dimensional")
        if user_ids.shape != item_ids.shape:
            raise ValueError("user_ids and item_ids must have the same shape")
        user_ids = user_ids.to(dtype=torch.long)
        item_ids = item_ids.to(dtype=torch.long)
        if user_ids.numel():
            if bool((user_ids < 0).any()) or bool((user_ids >= self.n_users).any()):
                raise ValueError("user IDs must be in [0, n_users)")
            if bool((item_ids < 0).any()) or bool((item_ids >= self.n_items).any()):
                raise ValueError("item IDs must be in [0, n_items)")
        user_embeddings, item_embeddings = self.encode()
        return (user_embeddings[user_ids] * item_embeddings[item_ids]).sum(dim=-1)
