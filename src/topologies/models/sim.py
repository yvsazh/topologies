import torch
from torch import nn


class SIM(nn.Module):
    architecture_id = "sim"

    def __init__(self, embedding_dim=8, attention_dim=None, num_heads=2):
        super().__init__()
        if embedding_dim <= 0 or num_heads <= 0:
            raise ValueError("embedding_dim and num_heads must be positive")
        attention_dim = embedding_dim if attention_dim is None else attention_dim
        if attention_dim <= 0:
            raise ValueError("attention_dim must be positive")
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.query_projections = nn.ModuleList(
            nn.Linear(embedding_dim, attention_dim, bias=False)
            for _ in range(num_heads)
        )
        self.key_projections = nn.ModuleList(
            nn.Linear(embedding_dim, attention_dim, bias=False)
            for _ in range(num_heads)
        )

    def forward(self, history, history_categories, query, query_categories, return_attention=False):
        history = torch.as_tensor(history)
        query = torch.as_tensor(query, device=history.device, dtype=history.dtype)
        history_categories = torch.as_tensor(history_categories, device=history.device)
        query_categories = torch.as_tensor(query_categories, device=history.device)
        if history.ndim != 3 or query.ndim != 3:
            raise ValueError("history and query must have shape (batch, time, embedding_dim)")
        if history.shape[0] != query.shape[0] or history.shape[-1] != self.embedding_dim or query.shape[-1] != self.embedding_dim:
            raise ValueError("history and query batch/embedding dimensions do not match")
        if history_categories.shape != history.shape[:2] or query_categories.shape != query.shape[:2]:
            raise ValueError("category tensors must match their sequence dimensions")

        matching = history_categories[:, None, :] == query_categories[:, :, None]
        outputs = []
        attention_weights = []
        for query_projection, key_projection in zip(self.query_projections, self.key_projections):
            projected_query = query_projection(query)
            projected_history = key_projection(history)
            scores = (projected_query.unsqueeze(2) * projected_history.unsqueeze(1)).sum(dim=-1)
            scores = scores.masked_fill(~matching, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            weights = torch.where(matching, weights, torch.zeros_like(weights))
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
            outputs.append(torch.matmul(weights, history))
            attention_weights.append(weights)

        result = torch.cat(outputs, dim=-1)
        if return_attention:
            return result, torch.stack(attention_weights, dim=2)
        return result
