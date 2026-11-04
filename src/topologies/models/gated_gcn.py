import torch
from torch import nn
from torch.nn import functional as F


class GatedGCN(nn.Module):
    architecture_id = "gated-gcn"
    PRESETS = {
        "tiny": {"input_dim": 4, "hidden_dim": 8, "output_dim": 2, "num_layers": 2},
        "small": {"input_dim": 16, "hidden_dim": 32, "output_dim": 4, "num_layers": 4},
        "base": {"input_dim": 50, "hidden_dim": 50, "output_dim": 10, "num_layers": 6},
    }

    def __init__(self, input_dim, hidden_dim=50, output_dim=2, num_layers=6):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("hidden_dim", hidden_dim),
            ("output_dim", output_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(num_layers, bool)
            or not isinstance(num_layers, int)
            or num_layers <= 0
            or num_layers % 2
        ):
            raise ValueError("num_layers must be a positive even integer")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        center_projections = []
        message_projections = []
        gate_destination_projections = []
        gate_source_projections = []
        gate_biases = []
        batch_norms = []
        residual_projections = []
        for layer in range(num_layers):
            in_features = input_dim if layer == 0 else hidden_dim
            center_projections.append(nn.Linear(in_features, hidden_dim))
            message_projections.append(nn.Linear(in_features, hidden_dim, bias=False))
            gate_destination_projections.append(
                nn.Linear(in_features, hidden_dim, bias=False)
            )
            gate_source_projections.append(nn.Linear(in_features, hidden_dim, bias=False))
            gate_biases.append(nn.Parameter(torch.zeros(hidden_dim)))
            batch_norms.append(nn.BatchNorm1d(hidden_dim))
            if layer % 2 == 0:
                residual_projections.append(nn.Linear(in_features, hidden_dim))
        self.center_projections = nn.ModuleList(center_projections)
        self.message_projections = nn.ModuleList(message_projections)
        self.gate_destination_projections = nn.ModuleList(gate_destination_projections)
        self.gate_source_projections = nn.ModuleList(gate_source_projections)
        self.gate_biases = nn.ParameterList(gate_biases)
        self.batch_norms = nn.ModuleList(batch_norms)
        self.residual_projections = nn.ModuleList(residual_projections)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def _convolution(self, layer, x, edge_index):
        sources, destinations = edge_index
        center = self.center_projections[layer](x)
        if sources.numel() == 0:
            return center
        source_features = x.index_select(0, sources)
        destination_features = x.index_select(0, destinations)
        gates = torch.sigmoid(
            self.gate_destination_projections[layer](destination_features)
            + self.gate_source_projections[layer](source_features)
            + self.gate_biases[layer]
        )
        messages = self.message_projections[layer](source_features) * gates
        aggregate = torch.zeros_like(center).index_add(0, destinations, messages)
        return center + aggregate

    def forward(self, x, edge_index, edge_weight=None, batch=None, node_level=True):
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("x must be a nonempty [num_nodes, input_dim] tensor")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"x must have feature width {self.input_dim}")
        if not x.is_floating_point():
            raise ValueError("x must have a floating point dtype")
        parameter = self.classifier.weight
        if x.device != parameter.device:
            raise ValueError("x and the model parameters must be on the same device")
        if x.dtype != parameter.dtype:
            raise ValueError("x and the model parameters must have the same dtype")
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2:
            raise ValueError("edge_index must be a [2, num_edges] tensor")
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.dtype != torch.long:
            raise ValueError("edge_index must have dtype torch.long")
        if edge_index.device != x.device:
            raise ValueError("edge_index and x must be on the same device")
        if edge_weight is not None:
            raise ValueError("edge_weight is not supported")
        if not node_level:
            raise ValueError("GatedGCN returns node-level logits only")
        num_nodes = x.shape[0]
        sources, destinations = edge_index
        if torch.any(sources < 0) or torch.any(sources >= num_nodes):
            raise ValueError("edge_index contains a source index outside the node range")
        if torch.any(destinations < 0) or torch.any(destinations >= num_nodes):
            raise ValueError("edge_index contains a destination index outside the node range")
        hidden = x
        for block in range(self.num_layers // 2):
            layer = block * 2
            residual = self.residual_projections[block](hidden)
            hidden = F.relu(self.batch_norms[layer](self._convolution(layer, hidden, edge_index)))
            hidden = self.batch_norms[layer + 1](
                self._convolution(layer + 1, hidden, edge_index)
            )
            hidden = F.relu(hidden + residual)
        return self.classifier(hidden)
