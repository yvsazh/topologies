import torch
from torch import nn
import torch.nn.functional as F


def contrast(left, right, temperature=0.5):
    if left.ndim != 2 or right.shape != left.shape:
        raise ValueError("views must have matching shape (nodes, features)")
    if left.shape[0] < 2:
        raise ValueError("InfoNCE requires at least two nodes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)

    def direction(anchor, other):
        inter_view = anchor @ other.T / temperature
        intra_view = anchor @ anchor.T / temperature
        intra_view = intra_view.masked_fill(
            torch.eye(anchor.shape[0], dtype=torch.bool, device=anchor.device), float("-inf")
        )
        positive = inter_view.diagonal()
        denominator = torch.logsumexp(torch.cat((inter_view, intra_view), dim=1), dim=1)
        return -(positive - denominator).mean()

    return (direction(left, right) + direction(right, left)) / 2


class GRACE(nn.Module):
    architecture_id = "grace"
    component_id = "grace"

    def __init__(self, in_channels, hidden_channels, out_channels=None, num_layers=2, temperature=0.5):
        super().__init__()
        if min(in_channels, hidden_channels, num_layers) <= 0:
            raise ValueError("channel counts and num_layers must be positive")
        if out_channels is None:
            out_channels = hidden_channels
        if out_channels <= 0 or temperature <= 0:
            raise ValueError("out_channels and temperature must be positive")
        self.temperature = float(temperature)
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.projector = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def encode(self, x, edge_index):
        if x.ndim != 2 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("expected x (nodes, features) and edge_index (2, edges)")
        n = x.shape[0]
        edges = torch.cat((edge_index.long(), torch.arange(n, device=x.device).repeat(2, 1)), dim=1)
        src, dst = edges
        degree = torch.bincount(dst, minlength=n).to(dtype=x.dtype).clamp_min_(1)
        norm = degree[src].rsqrt() * degree[dst].rsqrt()
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = torch.zeros_like(h).index_add_(0, dst, h[src] * norm.unsqueeze(-1))
            if i + 1 != len(self.layers):
                h = F.relu(h)
        return h

    def forward(self, x1, edge_index1, x2=None, edge_index2=None):
        if x2 is None:
            return self.encode(x1, edge_index1)
        if edge_index2 is None:
            raise ValueError("edge_index2 is required for a second view")
        left = self.projector(self.encode(x1, edge_index1))
        right = self.projector(self.encode(x2, edge_index2))
        return contrast(left, right, self.temperature)
