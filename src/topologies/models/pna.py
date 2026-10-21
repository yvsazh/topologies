import torch
from torch import nn


def principal(neighbors, degree, delta, eps=1e-8):
    if neighbors.ndim < 2 or neighbors.shape[0] == 0:
        raise ValueError("neighbors must contain at least one row")
    degree = torch.as_tensor(degree, dtype=neighbors.dtype, device=neighbors.device)
    delta = torch.as_tensor(delta, dtype=neighbors.dtype, device=neighbors.device)
    if torch.any(degree <= 0) or torch.any(delta <= 0):
        raise ValueError("degree and delta must be positive")
    log_degree = torch.log1p(degree)
    amp = log_degree / (delta + eps)
    atten = delta / (log_degree + eps)
    aggregations = (
        neighbors.mean(0),
        neighbors.amin(0),
        neighbors.amax(0),
        neighbors.std(0, unbiased=False),
    )
    return torch.cat(
        [torch.cat((value, value * amp, value * atten), dim=-1) for value in aggregations],
        dim=-1,
    )


class PNA(nn.Module):
    architecture_id = "pna"
    component_id = "pna"

    def __init__(self, in_channels, out_channels, delta=1.0):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or delta <= 0:
            raise ValueError("channel counts and delta must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer("delta", torch.tensor(float(delta)))
        self.mix = nn.Linear(12 * in_channels, out_channels)

    def forward(self, x, edge_index):
        if x.ndim != 2 or x.shape[1] != self.in_channels:
            raise ValueError("x must have shape (nodes, in_channels)")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, edges)")
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= x.shape[0]):
            raise ValueError("edge_index contains an invalid node index")
        output = x.new_zeros((x.shape[0], self.out_channels))
        delta = self.delta.to(dtype=x.dtype, device=x.device)
        for node in range(x.shape[0]):
            neighbors = x[edge_index[0, edge_index[1] == node]]
            if neighbors.shape[0]:
                degree = x.new_tensor(float(neighbors.shape[0]))
                stats = principal(neighbors, degree, delta)
                output[node] = self.mix(stats)
        return output
