import torch
from torch import nn


class EGNN(nn.Module):
    architecture_id = "egnn"
    PRESETS = {
        "tiny": {"input_dim": 8, "hidden_dim": 16, "num_layers": 2, "edge_dim": 0},
        "small": {"input_dim": 16, "hidden_dim": 64, "num_layers": 4, "edge_dim": 0},
        "base": {"input_dim": 16, "hidden_dim": 128, "num_layers": 7, "edge_dim": 0},
    }

    def __init__(self, input_dim=16, hidden_dim=64, num_layers=4, edge_dim=0):
        super().__init__()
        if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if not isinstance(edge_dim, int) or isinstance(edge_dim, bool) or edge_dim < 0:
            raise ValueError("edge_dim must be a non-negative integer")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.edge_functions = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_dim + 1 + edge_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            for _ in range(num_layers)
        )
        self.coordinate_functions = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_layers)
        )
        self.node_functions = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        )

    def forward(self, node_features, coordinates, edge_attributes=None):
        if not isinstance(node_features, torch.Tensor) or not isinstance(coordinates, torch.Tensor):
            raise TypeError("node_features and coordinates must be tensors")
        if node_features.ndim != 3 or coordinates.ndim != 3:
            raise ValueError("node_features and coordinates must have shape (batch, nodes, features)")
        if node_features.shape[0] == 0 or node_features.shape[1] == 0 or node_features.shape[2] != self.input_dim:
            raise ValueError("node_features must have non-empty batch and node dimensions and the configured feature width")
        if coordinates.shape[0] != node_features.shape[0] or coordinates.shape[1] != node_features.shape[1]:
            raise ValueError("node_features and coordinates must have matching batch and node dimensions")
        if coordinates.shape[2] == 0:
            raise ValueError("coordinates must have a non-empty coordinate dimension")
        if not node_features.is_floating_point() or not coordinates.is_floating_point():
            raise ValueError("node_features and coordinates must have floating-point dtype")
        if node_features.device != coordinates.device or node_features.dtype != coordinates.dtype:
            raise ValueError("node_features and coordinates must share device and dtype")
        if self.edge_dim == 0:
            if edge_attributes is not None:
                raise ValueError("edge_attributes must be omitted when edge_dim is 0")
        else:
            if not isinstance(edge_attributes, torch.Tensor):
                raise ValueError("edge_attributes is required when edge_dim is positive")
            expected_shape = (node_features.shape[0], node_features.shape[1], node_features.shape[1], self.edge_dim)
            if tuple(edge_attributes.shape) != expected_shape:
                raise ValueError("edge_attributes must have shape (batch, nodes, nodes, edge_dim)")
            if not edge_attributes.is_floating_point():
                raise ValueError("edge_attributes must have floating-point dtype")
            if edge_attributes.device != node_features.device or edge_attributes.dtype != node_features.dtype:
                raise ValueError("edge_attributes must share device and dtype with node_features")

        batch_size, num_nodes, _ = node_features.shape
        hidden = self.input_projection(node_features)
        current_coordinates = coordinates
        non_self_edges = ~torch.eye(num_nodes, dtype=torch.bool, device=node_features.device)
        non_self_edges = non_self_edges.unsqueeze(0).unsqueeze(-1)

        for edge_function, coordinate_function, node_function in zip(
            self.edge_functions, self.coordinate_functions, self.node_functions
        ):
            node_i = hidden.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, self.hidden_dim)
            node_j = hidden.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, self.hidden_dim)
            relative_coordinates = current_coordinates.unsqueeze(2) - current_coordinates.unsqueeze(1)
            squared_distances = relative_coordinates.square().sum(dim=-1, keepdim=True)
            edge_inputs = [node_i, node_j, squared_distances]
            if self.edge_dim > 0:
                edge_inputs.append(edge_attributes)
            messages = edge_function(torch.cat(edge_inputs, dim=-1))
            messages = messages.masked_fill(~non_self_edges, 0.0)
            aggregated_messages = messages.sum(dim=2)
            hidden = hidden + node_function(torch.cat((hidden, aggregated_messages), dim=-1))
            if num_nodes > 1:
                coordinate_weights = coordinate_function(messages).masked_fill(~non_self_edges, 0.0)
                coordinate_updates = (relative_coordinates * coordinate_weights).sum(dim=2)
                current_coordinates = current_coordinates + coordinate_updates / (num_nodes - 1)

        return hidden, current_coordinates
