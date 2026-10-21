import torch
from torch import nn


class SAGPool(nn.Module):
    def __init__(self, in_channels, ratio=0.5):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        self.in_channels = in_channels
        self.ratio = ratio
        self.score = nn.Linear(in_channels, 1, bias=False)

    def forward(self, x, edge_index=None):
        if x.ndim != 2 or x.shape[1] != self.in_channels or x.shape[0] == 0:
            raise ValueError("x must have shape (nodes, in_channels) with at least one node")
        nodes = x.shape[0]
        adjacency = x.new_zeros((nodes, nodes))
        adjacency.fill_diagonal_(1)
        if edge_index is not None:
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("edge_index must have shape (2, edges)")
            edge_index = edge_index.to(device=x.device, dtype=torch.long)
            if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= nodes):
                raise ValueError("edge_index contains an invalid node index")
            adjacency[edge_index[0], edge_index[1]] = 1
            adjacency[edge_index[1], edge_index[0]] = 1
        degree = adjacency.sum(dim=1).clamp_min(1)
        normalized = adjacency / degree.unsqueeze(1)
        scores = torch.tanh(self.score(normalized @ x).squeeze(-1))
        keep = max(1, int(self.ratio * nodes))
        indices = torch.topk(scores, keep, sorted=True).indices
        return x.index_select(0, indices) * scores.index_select(0, indices).unsqueeze(-1), indices, scores
