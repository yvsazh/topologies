import torch
from torch import nn
from torch.nn import functional as F


class NearestNeighborRetriever(nn.Module):
    component_id = "nearest-neighbor-retriever"
    def forward(self, query, keys, values):
        query = torch.as_tensor(query)
        keys = torch.as_tensor(keys, device=query.device, dtype=query.dtype)
        values = torch.as_tensor(values, device=query.device)
        if query.ndim < 1 or keys.ndim != query.ndim + 1 or values.ndim != keys.ndim:
            raise ValueError("query, keys, and values must have feature and row dimensions")
        if query.shape[-1] != keys.shape[-1] or keys.shape[-2] != values.shape[-2]:
            raise ValueError("query/key feature dimensions and key/value row counts must match")
        if keys.shape[-2] == 0:
            raise ValueError("keys must contain at least one row")
        if values.shape[:-2] != keys.shape[:-2] or query.shape[:-1] != keys.shape[:-2]:
            raise ValueError("query, keys, and values batch dimensions must match")
        normalized_query = F.normalize(query, p=2, dim=-1)
        normalized_keys = F.normalize(keys, p=2, dim=-1)
        indices = torch.sum(normalized_query.unsqueeze(-2) * normalized_keys, dim=-1).argmax(dim=-1)
        return torch.gather(values, -2, indices[..., None, None].expand(*indices.shape, 1, values.shape[-1])).squeeze(-2)
