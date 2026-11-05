import torch
from torch import nn


class GraphCast(nn.Module):
    architecture_id = "graphcast"
    PRESETS = {
        "tiny": {"latent_features": 16, "hidden_features": 32},
        "small": {"latent_features": 64, "hidden_features": 128},
        "base": {"latent_features": 512, "hidden_features": 512},
    }

    def __init__(self, latent_features=512, hidden_features=512):
        super().__init__()
        if type(latent_features) is not int or latent_features < 1:
            raise ValueError("latent_features must be a positive integer")
        if type(hidden_features) is not int or hidden_features < 1:
            raise ValueError("hidden_features must be a positive integer")

        self.latent_features = latent_features
        self.hidden_features = hidden_features
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * latent_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, latent_features),
        )
        self.edge_norm = nn.LayerNorm(latent_features)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * latent_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, latent_features),
        )
        self.node_norm = nn.LayerNorm(latent_features)

    def forward(self, nodes, edge_index, edges):
        if not isinstance(nodes, torch.Tensor) or nodes.ndim != 2 or nodes.shape[0] < 1:
            raise ValueError("nodes must be a nonempty rank-2 tensor with shape [N,D]")
        if not nodes.is_floating_point():
            raise ValueError("nodes must have a floating-point dtype")
        if nodes.shape[1] != self.latent_features:
            raise ValueError(f"nodes must have feature width {self.latent_features}")
        if not isinstance(edges, torch.Tensor) or edges.ndim != 2:
            raise ValueError("edges must be a rank-2 tensor with shape [E,D]")
        if not edges.is_floating_point():
            raise ValueError("edges must have a floating-point dtype")
        if edges.shape[1] != self.latent_features:
            raise ValueError(f"edges must have feature width {self.latent_features}")
        if edges.shape[0] < 0:
            raise ValueError("edges must have a nonnegative edge count")
        if edges.device != nodes.device:
            raise ValueError("nodes and edges must be on the same device")
        if edges.dtype != nodes.dtype:
            raise ValueError("nodes and edges must have the same dtype")
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2:
            raise ValueError("edge_index must be a rank-2 integer tensor with shape [2,E]")
        if edge_index.shape[0] != 2 or edge_index.shape[1] != edges.shape[0]:
            raise ValueError("edge_index must have shape [2,E] matching edges")
        if edge_index.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("edge_index must have an integer dtype")
        if edge_index.device != nodes.device:
            raise ValueError("edge_index, nodes, and edges must be on the same device")
        if nodes.dtype != self.edge_mlp[0].weight.dtype:
            raise ValueError("input tensors must use the model parameter dtype")
        if nodes.device != self.edge_mlp[0].weight.device:
            raise ValueError("input tensors must be on the model parameter device")

        edge_index = edge_index.to(dtype=torch.long)
        senders = edge_index[0]
        receivers = edge_index[1]
        if senders.numel() and ((senders < 0) | (senders >= nodes.shape[0])).any().item():
            raise ValueError("edge_index contains an out-of-range sender index")
        if receivers.numel() and ((receivers < 0) | (receivers >= nodes.shape[0])).any().item():
            raise ValueError("edge_index contains an out-of-range receiver index")

        sender_nodes = nodes.index_select(0, senders)
        receiver_nodes = nodes.index_select(0, receivers)
        edge_features = torch.cat((edges, sender_nodes, receiver_nodes), dim=-1)
        edge_updates = self.edge_norm(self.edge_mlp(edge_features))
        aggregates = nodes.new_zeros((nodes.shape[0], self.latent_features))
        aggregates.index_add_(0, receivers, edge_updates)
        node_features = torch.cat((nodes, aggregates), dim=-1)
        node_updates = self.node_norm(self.node_mlp(node_features))

        return nodes + node_updates, edges + edge_updates
