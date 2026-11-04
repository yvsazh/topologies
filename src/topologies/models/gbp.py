import math

import torch
from torch import nn


class GBP(nn.Module):
    architecture_id = "gbp"
    PRESETS = {
        "tiny": {
            "in_features": 4,
            "hidden_dim": 8,
            "num_classes": 3,
            "propagation_steps": 2,
            "alpha": 0.1,
            "r": 0.5,
            "dropout": 0.0,
        },
        "small": {
            "in_features": 16,
            "hidden_dim": 64,
            "num_classes": 8,
            "propagation_steps": 4,
            "alpha": 0.1,
            "r": 0.5,
            "dropout": 0.5,
        },
    }

    def __init__(
        self,
        in_features,
        num_classes,
        hidden_dim=64,
        propagation_steps=4,
        alpha=0.1,
        r=0.5,
        dropout=0.5,
        coefficients=None,
        bias=False,
    ):
        for name, value in (
            ("in_features", in_features),
            ("num_classes", num_classes),
            ("hidden_dim", hidden_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(propagation_steps, bool) or not isinstance(propagation_steps, int)
                or propagation_steps < 0):
            raise ValueError("propagation_steps must be a nonnegative integer")
        if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
                or not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0):
            raise ValueError("alpha must be a finite number strictly between zero and one")
        if (isinstance(r, bool) or not isinstance(r, (int, float))
                or not math.isfinite(float(r)) or not 0.0 <= float(r) <= 1.0):
            raise ValueError("r must be a finite number between zero and one")
        if (isinstance(dropout, bool) or not isinstance(dropout, (int, float))
                or not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0):
            raise ValueError("dropout must be a finite number in [0, 1)")
        if bias is not True and bias is not False:
            raise ValueError("bias must be a boolean")
        if coefficients is None:
            coefficients = [float(alpha) * (1.0 - float(alpha)) ** hop
                            for hop in range(propagation_steps + 1)]
        try:
            weights = torch.as_tensor(coefficients, dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("coefficients must be a finite sequence of weights") from error
        if (weights.ndim != 1 or weights.numel() != propagation_steps + 1
                or not torch.isfinite(weights).all() or torch.any(weights < 0)
                or weights.sum() > 1.0 + 1e-12):
            raise ValueError(
                "coefficients must be finite, nonnegative, one per hop, and sum to at most one"
            )
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.propagation_steps = propagation_steps
        self.alpha = float(alpha)
        self.r = float(r)
        self.dropout = float(dropout)
        self.register_buffer("coefficients", weights.to(dtype=torch.float32))
        self.fc1 = nn.Linear(in_features, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=bias)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, features, adjacency):
        if not torch.is_tensor(features) or not torch.is_tensor(adjacency):
            raise ValueError("features and adjacency must be torch.Tensor values")
        if features.ndim != 2 or adjacency.ndim != 2:
            raise ValueError("features and adjacency must be matrices")
        if features.shape[1] != self.in_features:
            raise ValueError(f"features last dimension must match in_features ({self.in_features})")
        if (adjacency.shape[0] != adjacency.shape[1]
                or adjacency.shape[0] != features.shape[0]):
            raise ValueError("adjacency must be square and match the node count")
        if not features.is_floating_point():
            raise ValueError("features must have a floating-point dtype")
        if features.dtype != self.fc1.weight.dtype or features.device != self.fc1.weight.device:
            raise ValueError("features must match the model parameter dtype and device")
        adjacency = adjacency.to(device=features.device, dtype=features.dtype)
        if not torch.isfinite(adjacency).all() or torch.any(adjacency < 0):
            raise ValueError("adjacency values must be finite and nonnegative")
        degree = adjacency.sum(dim=1)
        if torch.any(degree <= 0):
            raise ValueError("each node must have positive degree; supply self-loops for isolated nodes")
        transition = (
            degree.pow(self.r - 1.0).unsqueeze(1)
            * adjacency
            * degree.pow(-self.r).unsqueeze(0)
        )
        propagated = features
        features = self.coefficients[0].to(dtype=features.dtype) * propagated
        for hop in range(1, self.propagation_steps + 1):
            propagated = transition @ propagated
            features = features + self.coefficients[hop].to(dtype=propagated.dtype) * propagated
        hidden = torch.relu(self.fc1(self.dropout_layer(features)))
        hidden = self.dropout_layer(hidden)
        return self.fc2(hidden)
