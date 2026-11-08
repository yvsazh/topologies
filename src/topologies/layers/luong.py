import torch
from torch import nn


class GeneralAttention(nn.Module):
    component_id = "luong"

    def __init__(self, query_dim, source_dim):
        super().__init__()
        if isinstance(query_dim, bool) or not isinstance(query_dim, int) or query_dim <= 0:
            raise ValueError("query_dim must be a positive integer")
        if isinstance(source_dim, bool) or not isinstance(source_dim, int) or source_dim <= 0:
            raise ValueError("source_dim must be a positive integer")
        self.query_dim = query_dim
        self.source_dim = source_dim
        self.weight = nn.Parameter(torch.empty(query_dim, source_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, query, source, return_attention=True):
        if type(return_attention) is not bool:
            raise ValueError("return_attention must be a boolean")
        if not isinstance(query, torch.Tensor) or query.ndim != 2:
            raise ValueError("query must be a tensor with shape [batch, query_dim]")
        if not isinstance(source, torch.Tensor) or source.ndim != 3:
            raise ValueError("source must be a tensor with shape [batch, positions, source_dim]")
        if query.shape[1] != self.query_dim:
            raise ValueError(f"query must have query_dim={self.query_dim}")
        if source.shape[2] != self.source_dim:
            raise ValueError(f"source must have source_dim={self.source_dim}")
        if query.shape[0] != source.shape[0]:
            raise ValueError("query and source must have the same batch size")
        if source.shape[1] == 0:
            raise ValueError("source must contain at least one position")
        if not query.is_floating_point() or not source.is_floating_point():
            raise ValueError("query and source must have floating-point dtypes")
        if query.dtype != source.dtype:
            raise ValueError("query and source must have the same dtype")
        if query.device != source.device:
            raise ValueError("query and source must be on the same device")
        if query.dtype != self.weight.dtype or query.device != self.weight.device:
            raise ValueError("query and source must match the module weight dtype and device")
        projected_query = torch.matmul(query, self.weight)
        scores = torch.bmm(source, projected_query.unsqueeze(-1)).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), source).squeeze(1)
        return (context, weights) if return_attention else context
