import torch
from torch import nn
from torch.nn import functional as F


class GraphSAGE(nn.Module):
    architecture_id = "graphsage"
    PRESETS = {
        "tiny": {"input_dim": 4, "output_dim": 4},
        "small": {"input_dim": 32, "output_dim": 64},
    }

    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if isinstance(output_dim, bool) or not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection = nn.Linear(2 * input_dim, output_dim, bias=False)

    def forward(self, x, edge_index):
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("x must be a nonempty [num_nodes, input_dim] tensor")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"x must have feature width {self.input_dim}")
        if not x.is_floating_point():
            raise ValueError("x must have a floating point dtype")
        if x.device != self.projection.weight.device:
            raise ValueError("x and the model parameters must be on the same device")
        if x.dtype != self.projection.weight.dtype:
            raise ValueError("x and the model parameters must have the same dtype")
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2:
            raise ValueError("edge_index must be a [2, num_edges] tensor")
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.dtype != torch.long:
            raise ValueError("edge_index must have dtype torch.long")
        if edge_index.device != x.device:
            raise ValueError("edge_index and x must be on the same device")
        num_nodes = x.shape[0]
        sources, destinations = edge_index
        if torch.any(sources < 0) or torch.any(sources >= num_nodes):
            raise ValueError("edge_index contains a source index outside the node range")
        if torch.any(destinations < 0) or torch.any(destinations >= num_nodes):
            raise ValueError("edge_index contains a destination index outside the node range")
        counts = torch.bincount(destinations, minlength=num_nodes)
        if torch.any(counts == 0):
            raise ValueError("every node must have at least one incoming neighbor")
        neighbor_sum = torch.zeros_like(x).index_add(
            0, destinations, x.index_select(0, sources)
        )
        neighbor_mean = neighbor_sum / counts.to(dtype=x.dtype).unsqueeze(-1)
        combined = torch.cat((x, neighbor_mean), dim=-1)
        hidden = F.relu(self.projection(combined))
        return F.normalize(hidden, p=2, dim=-1, eps=1e-12)
