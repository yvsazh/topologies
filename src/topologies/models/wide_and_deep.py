import torch
from torch import nn


class WideAndDeep(nn.Module):
    architecture_id = "wide-and-deep"
    def __init__(self, input_dim, hidden_dims=(64, 32), num_classes=1, cross_features=None):
        super().__init__()
        if input_dim < 1 or num_classes < 1:
            raise ValueError("input_dim and num_classes must be positive")
        if any(width < 1 for width in hidden_dims):
            raise ValueError("hidden dimensions must be positive")
        self.input_dim = input_dim
        self.cross_features = tuple(cross_features or ())
        if any(len(pair) != 2 or min(pair) < 0 or max(pair) >= input_dim for pair in self.cross_features):
            raise ValueError("cross feature indices must be valid pairs")
        self.wide = nn.Linear(input_dim + len(self.cross_features), num_classes)
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.deep = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"expected input shape [batch, {self.input_dim}]")
        cross_terms = [x[:, i] * x[:, j] for i, j in self.cross_features]
        wide_input = torch.cat((x, torch.stack(cross_terms, dim=1)), dim=1) if cross_terms else x
        return self.wide(wide_input) + self.deep(x)
