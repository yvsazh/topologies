import torch
from torch import nn


class MatchingNetworks(nn.Module):
    component_id = "matching-networks"

    def __init__(self, *args, **kwargs):
        super().__init__()
        if args or kwargs:
            raise ValueError("MatchingNetworks does not accept constructor arguments")

    def forward(self, query, support, support_labels):
        tensors = {
            "query": query,
            "support": support,
            "support_labels": support_labels,
        }
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")

        if query.ndim not in (2, 3):
            raise ValueError("query must have shape [Q, D] or [B, Q, D]")
        if support.ndim != query.ndim:
            raise ValueError("support must have the same rank as query")
        if support_labels.ndim != query.ndim:
            raise ValueError("support_labels must have the same rank as query")
        if query.shape[-1] == 0 or support.shape[-1] == 0:
            raise ValueError("query and support embedding dimensions must be non-empty")
        if support.shape[-2] == 0:
            raise ValueError("support must contain at least one embedding")
        if query.shape[-1] != support.shape[-1]:
            raise ValueError("query and support embedding dimensions must match")
        if support_labels.shape[-2] != support.shape[-2]:
            raise ValueError("support_labels must have one row per support embedding")
        if support_labels.shape[-1] == 0:
            raise ValueError("support_labels must contain at least one class")
        if query.ndim == 3 and (
            query.shape[0] != support.shape[0]
            or query.shape[0] != support_labels.shape[0]
        ):
            raise ValueError("batched query, support, and support_labels sizes must match")
        if not query.is_floating_point() or not support.is_floating_point():
            raise ValueError("query and support embeddings must use floating point dtypes")
        if query.dtype != support.dtype:
            raise ValueError("query and support embeddings must use the same dtype")
        if support_labels.is_complex():
            raise ValueError("support_labels must use a real numeric dtype")
        if query.device != support.device or query.device != support_labels.device:
            raise ValueError("query, support, and support_labels must be on the same device")
        if not torch.isfinite(query).all() or not torch.isfinite(support).all():
            raise ValueError("query and support embeddings must contain only finite values")
        if not torch.isfinite(support_labels).all():
            raise ValueError("support_labels must contain only finite values")
        if not torch.all((support_labels == 0) | (support_labels == 1)):
            raise ValueError("support_labels rows must be one-hot vectors")
        if not torch.all(support_labels.sum(dim=-1) == 1):
            raise ValueError("support_labels rows must contain exactly one active class")

        query_norm = torch.linalg.vector_norm(query, dim=-1, keepdim=True)
        support_norm = torch.linalg.vector_norm(support, dim=-1, keepdim=True)
        if not torch.isfinite(query_norm).all() or not torch.isfinite(support_norm).all():
            raise ValueError("query and support embedding norms must be finite")
        if torch.any(query_norm == 0) or torch.any(support_norm == 0):
            raise ValueError("query and support embeddings must have non-zero norms")

        normalized_query = query / query_norm
        normalized_support = support / support_norm
        cosine_similarity = normalized_query @ normalized_support.transpose(-1, -2)
        attention = torch.softmax(cosine_similarity, dim=-1)
        probabilities = attention @ support_labels.to(dtype=attention.dtype)
        if query.ndim == 3 and query.shape[-2] == 1:
            probabilities = probabilities.squeeze(-2)
        return probabilities
