from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F


class RGCN(nn.Module):
    architecture_id = "rgcn"
    PRESETS = {
        "tiny": {
            "in_features": 16,
            "num_classes": 4,
            "num_relations": 4,
            "hidden_features": 8,
            "num_layers": 2,
            "num_bases": 2,
        },
        "small": {
            "in_features": 32,
            "num_classes": 8,
            "num_relations": 16,
            "hidden_features": 16,
            "num_layers": 2,
            "num_bases": 8,
        },
    }

    def __init__(
        self,
        in_features,
        num_classes,
        num_relations,
        num_layers=2,
        hidden_features=16,
        num_bases=0,
    ):
        super().__init__()
        dimensions = (in_features, num_classes, num_relations, num_layers, hidden_features)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in dimensions):
            raise ValueError("feature counts, relation count, and layer count must be integers")
        if in_features <= 0 or num_classes <= 0 or num_relations <= 0:
            raise ValueError("in_features, num_classes, and num_relations must be positive")
        if num_layers <= 0 or hidden_features <= 0:
            raise ValueError("num_layers and hidden_features must be positive")
        if not isinstance(num_bases, int) or isinstance(num_bases, bool) or num_bases < 0:
            raise ValueError("num_bases must be a nonnegative integer")

        self.in_features = in_features
        self.num_classes = num_classes
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.hidden_features = hidden_features
        self.num_bases = num_bases
        layer_dimensions = [in_features] + [hidden_features] * (num_layers - 1) + [num_classes]
        self.self_weights = nn.ParameterList()
        self.relation_weights = nn.ParameterList()
        self.basis_matrices = nn.ParameterList()
        self.basis_coefficients = nn.ParameterList()

        for input_width, output_width in pairwise(layer_dimensions):
            self_weight = nn.Parameter(torch.empty(output_width, input_width))
            nn.init.xavier_uniform_(self_weight)
            self.self_weights.append(self_weight)
            if num_bases == 0:
                relation_weight = nn.Parameter(
                    torch.empty(num_relations, output_width, input_width)
                )
                nn.init.xavier_uniform_(relation_weight)
                self.relation_weights.append(relation_weight)
            else:
                basis = nn.Parameter(torch.empty(num_bases, output_width, input_width))
                coefficients = nn.Parameter(torch.empty(num_relations, num_bases))
                nn.init.xavier_uniform_(basis)
                nn.init.xavier_uniform_(coefficients)
                self.basis_matrices.append(basis)
                self.basis_coefficients.append(coefficients)

    def relation_matrices(self, layer_index):
        if not isinstance(layer_index, int) or isinstance(layer_index, bool):
            raise TypeError("layer_index must be an integer")
        if layer_index < 0 or layer_index >= self.num_layers:
            raise ValueError("layer_index is outside the model layers")
        if self.num_bases == 0:
            return self.relation_weights[layer_index]
        return torch.einsum(
            "rb,boi->roi",
            self.basis_coefficients[layer_index],
            self.basis_matrices[layer_index],
        )

    def forward(self, node_features, edge_index, edge_type):
        if not isinstance(node_features, torch.Tensor) or node_features.ndim != 2:
            raise ValueError("node_features must have shape (num_nodes, in_features)")
        if node_features.shape[1] != self.in_features:
            raise ValueError("node_features width must match in_features")
        if not torch.is_floating_point(node_features):
            raise ValueError("node_features must have a floating point dtype")
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, num_edges)")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_index must have an integer dtype")
        if not isinstance(edge_type, torch.Tensor) or edge_type.ndim != 1:
            raise ValueError("edge_type must have shape (num_edges,)")
        if edge_type.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_type must have an integer dtype")
        if edge_type.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_type must contain one relation ID per edge")
        if edge_index.device != node_features.device or edge_type.device != node_features.device:
            raise ValueError("node_features, edge_index, and edge_type must share a device")

        num_nodes = node_features.shape[0]
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
            raise ValueError("edge_index values must be within the node range")
        if edge_type.numel() and (edge_type.min() < 0 or edge_type.max() >= self.num_relations):
            raise ValueError("edge_type values must be within the relation range")

        source = edge_index[0].long()
        target = edge_index[1].long()
        relation_ids = edge_type.long()
        hidden = node_features
        for layer_index in range(self.num_layers):
            output = hidden @ self.self_weights[layer_index].transpose(0, 1)
            relation_weights = self.relation_matrices(layer_index)
            for relation in range(self.num_relations):
                edge_mask = relation_ids == relation
                if edge_mask.any():
                    relation_targets = target[edge_mask]
                    relation_sources = source[edge_mask]
                    messages = hidden[relation_sources] @ relation_weights[relation].transpose(0, 1)
                    relation_sum = hidden.new_zeros((num_nodes, relation_weights.shape[1]))
                    relation_sum = relation_sum.index_add(0, relation_targets, messages)
                    degree = torch.bincount(relation_targets, minlength=num_nodes).to(hidden.dtype)
                    output = output + relation_sum / degree.clamp_min(1).unsqueeze(1)
            if layer_index < self.num_layers - 1:
                hidden = F.relu(output)
            else:
                hidden = output
        return hidden
