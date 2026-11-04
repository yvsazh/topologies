import math

import torch
from torch import nn
from torch.nn import functional as F


class GAT(nn.Module):
    architecture_id = "gat"
    PRESETS = {
        "tiny": {"in_features": 4, "hidden_features": 2, "num_classes": 3, "heads": 2, "output_heads": 1, "dropout": 0.6, "negative_slope": 0.2},
        "small": {"in_features": 64, "hidden_features": 64, "num_classes": 16, "heads": 8, "output_heads": 1, "dropout": 0.6, "negative_slope": 0.2},
    }

    def __init__(self, in_features, num_classes, hidden_features=8, heads=8, output_heads=1, dropout=0.6, negative_slope=0.2):
        super().__init__()
        dimensions = {
            "in_features": in_features,
            "num_classes": num_classes,
            "hidden_features": hidden_features,
            "heads": heads,
            "output_heads": output_heads,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a finite number in [0, 1)")
        if isinstance(negative_slope, bool) or not isinstance(negative_slope, (int, float)) or not math.isfinite(negative_slope) or negative_slope <= 0:
            raise ValueError("negative_slope must be a finite positive number")
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.heads = heads
        self.output_heads = output_heads
        self.dropout = float(dropout)
        self.negative_slope = float(negative_slope)
        self.hidden_projection = nn.Parameter(torch.empty(heads, hidden_features, in_features))
        self.hidden_attention = nn.Parameter(torch.empty(heads, 2 * hidden_features))
        self.output_projection = nn.Parameter(torch.empty(output_heads, num_classes, heads * hidden_features))
        self.output_attention = nn.Parameter(torch.empty(output_heads, 2 * num_classes))
        for projection in self.hidden_projection:
            nn.init.xavier_uniform_(projection)
        for projection in self.output_projection:
            nn.init.xavier_uniform_(projection)
        for head in range(heads):
            nn.init.xavier_uniform_(self.hidden_attention[head:head + 1])
        for head in range(output_heads):
            nn.init.xavier_uniform_(self.output_attention[head:head + 1])

    def _aggregate(self, x, edge_index, projection, attention):
        node_count = x.shape[0]
        projected = torch.einsum("nf,hof->nho", F.dropout(x, p=self.dropout, training=self.training), projection)
        feature_count = projected.shape[-1]
        left = (projected * attention[:, :feature_count]).sum(dim=-1)
        right = (projected * attention[:, feature_count:]).sum(dim=-1)
        source, target = edge_index
        keep = source != target
        loops = torch.arange(node_count, device=x.device)
        source = torch.cat((source[keep], loops))
        target = torch.cat((target[keep], loops))
        scores = F.leaky_relu(left[target] + right[source], negative_slope=self.negative_slope)
        indices = target.unsqueeze(1).expand_as(scores)
        maximum = scores.new_full((node_count, scores.shape[1]), -torch.inf)
        maximum.scatter_reduce_(0, indices, scores, reduce="amax", include_self=True)
        exponentials = (scores - maximum[target]).exp()
        normalizer = scores.new_zeros((node_count, scores.shape[1])).index_add(0, target, exponentials)
        coefficients = exponentials / normalizer[target]
        coefficients = F.dropout(coefficients, p=self.dropout, training=self.training)
        messages = projected[source] * coefficients.unsqueeze(-1)
        return projected.new_zeros((node_count, projected.shape[1], feature_count)).index_add(0, target, messages)

    def forward(self, x, edge_index, edge_weight=None, batch=None, node_level=True):
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError("x must have shape [nodes, in_features]")
        if not x.is_floating_point() or x.shape[0] == 0:
            raise ValueError("x must contain floating features for at least one node")
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if edge_index.dtype not in integer_types:
            raise ValueError("edge_index must contain integer node indices")
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        if edge_index.numel() and (bool((edge_index < 0).any()) or bool((edge_index >= x.shape[0]).any())):
            raise ValueError("edge_index contains an invalid node index")
        if edge_weight is not None:
            raise ValueError("edge_weight is unsupported for unweighted GAT attention")
        if node_level is not True:
            raise ValueError("GAT supports node_level=True only")
        hidden = F.elu(self._aggregate(x, edge_index, self.hidden_projection, self.hidden_attention))
        hidden = hidden.reshape(x.shape[0], self.heads * self.hidden_features)
        output = self._aggregate(hidden, edge_index, self.output_projection, self.output_attention)
        return output.mean(dim=1)
