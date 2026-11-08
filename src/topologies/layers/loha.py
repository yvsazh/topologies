import torch
from torch import nn


class LoHaLinear(nn.Module):
    component_id = "loha"

    def __init__(self, in_features, out_features, rank1, rank2, activation="identity"):
        super().__init__()
        for name, value in (
            ("in_features", in_features),
            ("out_features", out_features),
            ("rank1", rank1),
            ("rank2", rank2),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if activation not in ("identity", "tanh"):
            raise ValueError("activation must be 'identity' or 'tanh'")
        self.in_features = in_features
        self.out_features = out_features
        self.rank1 = rank1
        self.rank2 = rank2
        self.activation = activation
        self.x1 = nn.Parameter(torch.empty(out_features, rank1))
        self.y1 = nn.Parameter(torch.empty(in_features, rank1))
        self.x2 = nn.Parameter(torch.empty(out_features, rank2))
        self.y2 = nn.Parameter(torch.empty(in_features, rank2))
        for factor in (self.x1, self.y1, self.x2, self.y2):
            nn.init.kaiming_normal_(factor, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim < 1:
            raise ValueError("x must have at least one dimension")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"x final dimension must be {self.in_features}")
        weight1 = self.x1 @ self.y1.transpose(0, 1)
        weight2 = self.x2 @ self.y2.transpose(0, 1)
        if self.activation == "tanh":
            weight1 = torch.tanh(weight1)
            weight2 = torch.tanh(weight2)
        weight = weight1 * weight2
        return torch.matmul(x, weight.transpose(0, 1))
