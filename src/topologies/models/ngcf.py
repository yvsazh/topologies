import torch
from torch import nn
from torch.nn import functional as F


class _PropagationLayer(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.bilinear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.bilinear.weight)

    def forward(self, embeddings, normalized_adjacency):
        neighbors = normalized_adjacency @ embeddings
        messages = self.linear(embeddings + neighbors) + self.bilinear(embeddings * neighbors)
        return F.leaky_relu(messages)


class NGCF(nn.Module):
    architecture_id = "ngcf"
    def __init__(self, n_users, n_items, embedding_dim, n_layers=1):
        super().__init__()
        if min(n_users, n_items, embedding_dim) <= 0 or n_layers < 0:
            raise ValueError("sizes must be positive and n_layers nonnegative")
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.propagation_layers = nn.ModuleList(
            _PropagationLayer(embedding_dim) for _ in range(n_layers)
        )
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, interactions):
        interactions = torch.as_tensor(interactions, dtype=self.user_embedding.weight.dtype, device=self.user_embedding.weight.device)
        if interactions.shape != (self.n_users, self.n_items):
            raise ValueError("interactions must have shape (n_users, n_items)")
        user_zeros = interactions.new_zeros((self.n_users, self.n_users))
        item_zeros = interactions.new_zeros((self.n_items, self.n_items))
        adjacency = torch.cat((
            torch.cat((user_zeros, interactions), dim=1),
            torch.cat((interactions.T, item_zeros), dim=1),
        ), dim=0)
        degree = adjacency.sum(dim=1).clamp_min(1.0)
        normalized = adjacency / torch.sqrt(degree[:, None] * degree[None, :])
        embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_layers = [embeddings]
        for layer in self.propagation_layers:
            embeddings = layer(embeddings, normalized)
            all_layers.append(embeddings)
        result = torch.cat(all_layers, dim=-1)
        return result[:self.n_users], result[self.n_users:]

    def score(self, user_ids, item_ids, interactions):
        users, items = self(interactions)
        return (users[user_ids] * items[item_ids]).sum(dim=-1)
