import torch
from torch import nn


class SortPool(nn.Module):
    component_id = "sortpool"

    def __init__(self, in_channels, k):
        super().__init__()
        if in_channels <= 0 or k <= 0:
            raise ValueError("in_channels and k must be positive")
        self.in_channels = in_channels
        self.k = k

    def forward(self, x, edge_index=None, adjacency=None):
        if x.ndim != 2 or x.shape[1] != self.in_channels:
            raise ValueError("x must have shape (num_nodes, in_channels)")
        if adjacency is not None and edge_index is not None:
            raise ValueError("provide adjacency or edge_index, not both")
        n = x.shape[0]
        count = min(self.k, n)
        order = torch.argsort(x[:, -1], descending=True, stable=True)[:count]
        selected_x = x.index_select(0, order)
        pooled_x = torch.cat(
            (selected_x, x.new_zeros((self.k - count, self.in_channels))), dim=0
        )
        if adjacency is not None:
            if adjacency.shape != (n, n):
                raise ValueError("adjacency must have shape (num_nodes, num_nodes)")
            pooled_adjacency = adjacency.index_select(0, order).index_select(1, order)
            pooled_adjacency = torch.nn.functional.pad(
                pooled_adjacency, (0, self.k - count, 0, self.k - count)
            )
        elif edge_index is not None:
            edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=x.device)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("edge_index must have shape (2, num_edges)")
            if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= n):
                raise ValueError("edge_index contains an invalid node index")
            inverse = torch.full((n,), -1, dtype=torch.long, device=x.device)
            inverse[order] = torch.arange(count, device=x.device)
            mapped = inverse[edge_index]
            keep = (mapped >= 0).all(dim=0)
            pooled_adjacency = torch.zeros((self.k, self.k), dtype=x.dtype, device=x.device)
            if keep.any():
                pooled_adjacency[mapped[0, keep], mapped[1, keep]] = 1
        else:
            pooled_adjacency = None
        return pooled_x, order, pooled_adjacency
