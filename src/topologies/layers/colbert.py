import torch
from torch import nn


class MaxSim(nn.Module):
    component_id = "colbert"

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise ValueError("MaxSim does not accept constructor arguments")
        super().__init__()

    def forward(self, query_tokens, doc_tokens):
        if not torch.is_tensor(query_tokens):
            raise ValueError("query_tokens must be a torch.Tensor")
        if not torch.is_tensor(doc_tokens):
            raise ValueError("doc_tokens must be a torch.Tensor")
        if query_tokens.ndim not in (2, 3):
            raise ValueError("query_tokens must have rank 2 or 3")
        if doc_tokens.ndim not in (2, 3):
            raise ValueError("doc_tokens must have rank 2 or 3")
        if query_tokens.ndim != doc_tokens.ndim:
            raise ValueError("query_tokens and doc_tokens must have the same rank")
        if query_tokens.ndim == 3 and query_tokens.shape[0] == 0:
            raise ValueError("query_tokens batch size must be positive")
        if doc_tokens.ndim == 3 and doc_tokens.shape[0] == 0:
            raise ValueError("doc_tokens batch size must be positive")
        if query_tokens.shape[-2] == 0:
            raise ValueError("query_tokens must contain at least one token")
        if doc_tokens.shape[-2] == 0:
            raise ValueError("doc_tokens must contain at least one token")
        if query_tokens.shape[-1] == 0 or doc_tokens.shape[-1] == 0:
            raise ValueError("embedding width must be positive")
        if query_tokens.shape[-1] != doc_tokens.shape[-1]:
            raise ValueError("query_tokens and doc_tokens must have the same embedding width")
        if query_tokens.ndim == 3 and query_tokens.shape[0] != doc_tokens.shape[0]:
            raise ValueError("query_tokens and doc_tokens must have the same batch size")
        if query_tokens.device != doc_tokens.device:
            raise ValueError("query_tokens and doc_tokens must be on the same device")
        if query_tokens.dtype != doc_tokens.dtype:
            raise ValueError("query_tokens and doc_tokens must have the same dtype")
        if not query_tokens.is_floating_point() or not doc_tokens.is_floating_point():
            raise ValueError("query_tokens and doc_tokens must use floating-point dtypes")
        similarities = query_tokens @ doc_tokens.transpose(-1, -2)
        return similarities.max(dim=-1).values.sum(dim=-1)
