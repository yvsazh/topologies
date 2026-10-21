import torch
from torch import nn
from torch.nn import functional as F


def adaptive_adjacency(source, target):
    return torch.softmax(torch.relu(source @ target.transpose(-1, -2)), dim=-1)


class GraphWaveNet(nn.Module):
    architecture_id = "graph-wavenet"
    component_id = "graph-wavenet"

    def __init__(self, input_channels, hidden_channels, num_nodes, embedding_dim=8, dilation=1):
        super().__init__()
        if min(input_channels, hidden_channels, num_nodes, embedding_dim, dilation) <= 0:
            raise ValueError("channel counts, node count, embedding dimension, and dilation must be positive")
        self.num_nodes = num_nodes
        self.source_embedding = nn.Parameter(torch.randn(num_nodes, embedding_dim) * 0.1)
        self.target_embedding = nn.Parameter(torch.randn(num_nodes, embedding_dim) * 0.1)
        self.filter_conv = nn.Conv2d(input_channels, hidden_channels, (1, 2), dilation=(1, dilation))
        self.gate_conv = nn.Conv2d(input_channels, hidden_channels, (1, 2), dilation=(1, dilation))
        self.residual_conv = nn.Conv2d(hidden_channels, input_channels, 1)
        self.graph_projection = nn.Conv2d(input_channels, input_channels, 1)

    def forward(self, values):
        if values.ndim != 4 or values.shape[2] != self.num_nodes:
            raise ValueError("values must have shape (batch, channels, nodes, time)")
        adjacency = adaptive_adjacency(self.source_embedding, self.target_embedding)
        filtered = torch.tanh(self.filter_conv(values)) * torch.sigmoid(self.gate_conv(values))
        residual = self.residual_conv(filtered)
        graph = self.graph_projection(residual)
        graph = torch.einsum("bcnt,nm->bcmt", graph, adjacency)
        return graph + residual[..., -graph.shape[-1]:]
