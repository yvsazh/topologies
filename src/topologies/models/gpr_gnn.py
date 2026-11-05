import math

import torch
from torch import nn
from torch.nn import functional as F


class GPRGNN(nn.Module):
    architecture_id = "gpr-gnn"
    PRESETS = {
        "tiny": {
            "in_features": 4,
            "num_classes": 2,
            "hidden_features": 8,
            "K": 2,
            "alpha": 0.1,
            "initialization": "ppr",
            "dropout": 0.0,
            "dprate": 0.0,
        },
        "small": {
            "in_features": 16,
            "num_classes": 8,
            "hidden_features": 64,
            "K": 10,
            "alpha": 0.1,
            "initialization": "ppr",
            "dropout": 0.5,
            "dprate": 0.5,
        },
        "base": {
            "in_features": 64,
            "num_classes": 16,
            "hidden_features": 128,
            "K": 20,
            "alpha": 0.1,
            "initialization": "ppr",
            "dropout": 0.5,
            "dprate": 0.5,
        },
    }

    def __init__(
        self,
        in_features,
        num_classes,
        hidden_features=64,
        K=10,
        alpha=0.1,
        initialization="ppr",
        dropout=0.5,
        dprate=0.5,
    ):
        super().__init__()
        dimensions = {
            "in_features": in_features,
            "num_classes": num_classes,
            "hidden_features": hidden_features,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if isinstance(K, bool) or not isinstance(K, int) or K < 0:
            raise ValueError("K must be a nonnegative integer")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be a finite number in [0, 1]")
        if initialization not in {"ppr", "delta0", "deltaK", "random"}:
            raise ValueError("initialization must be one of ppr, delta0, deltaK, or random")
        for name, value in (("dropout", dropout), ("dprate", dprate)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be a finite number in [0, 1)")
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.K = K
        self.dropout = float(dropout)
        self.dprate = float(dprate)
        self.lin1 = nn.Linear(in_features, hidden_features)
        self.lin2 = nn.Linear(hidden_features, num_classes)
        if initialization == "ppr":
            gamma = [alpha * (1 - alpha) ** hop for hop in range(K)]
            gamma.append((1 - alpha) ** K)
            gamma = torch.tensor(gamma, dtype=torch.float32)
        elif initialization == "delta0":
            gamma = torch.zeros(K + 1, dtype=torch.float32)
            gamma[0] = 1.0
        elif initialization == "deltaK":
            gamma = torch.zeros(K + 1, dtype=torch.float32)
            gamma[K] = 1.0
        else:
            gamma = torch.randn(K + 1, dtype=torch.float32)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x, edge_index, edge_weight=None, batch=None, node_level=True):
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError("x must have shape [N, in_features]")
        if not x.is_floating_point() or x.shape[0] == 0:
            raise ValueError("x must contain floating features for at least one node")
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if edge_index.dtype not in integer_types:
            raise ValueError("edge_index must contain integer node indices")
        edge_index = edge_index.to(device=x.device, dtype=torch.long)
        node_count = x.shape[0]
        if edge_index.numel() and (bool((edge_index < 0).any()) or bool((edge_index >= node_count).any())):
            raise ValueError("edge_index contains an invalid node index")
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.shape[1])
        else:
            valid_weight_types = integer_types + (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            if not isinstance(edge_weight, torch.Tensor) or edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
                raise ValueError("edge_weight must have shape [E]")
            if edge_weight.dtype not in valid_weight_types:
                raise ValueError("edge_weight must contain real numeric values")
            edge_weight = edge_weight.to(device=x.device, dtype=x.dtype)
            if not bool(torch.isfinite(edge_weight).all()) or bool((edge_weight < 0).any()):
                raise ValueError("edge_weight must contain finite nonnegative values")
        if node_level is not True:
            raise ValueError("GPRGNN supports node_level=True only")
        source, target = edge_index
        loops = torch.arange(node_count, device=x.device)
        source = torch.cat((source, loops))
        target = torch.cat((target, loops))
        edge_weight = torch.cat((edge_weight, x.new_ones(node_count)))
        degree = x.new_zeros(node_count).index_add_(0, target, edge_weight)
        inverse_degree = degree.rsqrt()
        normalization = edge_weight * inverse_degree[source] * inverse_degree[target]
        hidden = F.dropout(x, p=self.dropout, training=self.training)
        hidden = F.relu(self.lin1(hidden))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = self.lin2(hidden)
        hidden = F.dropout(hidden, p=self.dprate, training=self.training)
        state = hidden
        logits = self.gamma[0] * state
        for hop in range(1, self.K + 1):
            state = state.new_zeros(state.shape).index_add_(
                0, target, state[source] * normalization[:, None]
            )
            logits = logits + self.gamma[hop] * state
        return logits
