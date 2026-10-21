import torch
from torch import nn
from torch.nn import functional as F


class LINE(nn.Module):
    architecture_id = "line"
    def __init__(self, num_nodes, embedding_dim=8, order=1):
        super().__init__()
        if num_nodes <= 0 or embedding_dim <= 0:
            raise ValueError("num_nodes and embedding_dim must be positive")
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        self.num_nodes = int(num_nodes)
        self.embedding_dim = int(embedding_dim)
        self.order = int(order)
        self.node_embeddings = nn.Embedding(self.num_nodes, self.embedding_dim)
        self.context_embeddings = nn.Embedding(self.num_nodes, self.embedding_dim) if self.order == 2 else None
        nn.init.xavier_uniform_(self.node_embeddings.weight)
        if self.context_embeddings is not None:
            nn.init.xavier_uniform_(self.context_embeddings.weight)

    def edge_logits(self, edges):
        edges = torch.as_tensor(edges, dtype=torch.long, device=self.node_embeddings.weight.device)
        if edges.ndim != 2 or edges.shape[-1] != 2:
            raise ValueError("edges must have shape (num_edges, 2)")
        source = self.node_embeddings(edges[:, 0])
        target_table = self.node_embeddings if self.order == 1 else self.context_embeddings
        target = target_table(edges[:, 1])
        return (source * target).sum(dim=-1)

    def forward(self, positive_edges, negative_edges=None):
        positive_logits = self.edge_logits(positive_edges)
        if positive_logits.numel() == 0:
            raise ValueError("positive_edges must not be empty")
        positive_loss = F.softplus(-positive_logits).mean()
        if negative_edges is None:
            return positive_loss
        negative_logits = self.edge_logits(negative_edges)
        if negative_logits.numel() == 0:
            raise ValueError("negative_edges must not be empty")
        negative_loss = F.softplus(negative_logits).mean() * (
            negative_logits.numel() / positive_logits.numel()
        )
        return positive_loss + negative_loss
