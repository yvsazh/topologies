import math

import torch
from torch import nn


class DeepSurv(nn.Module):
    component_id = "deepsurv"
    PRESETS = {
        "tiny": {"hidden_dims": (4,), "activation": "selu", "dropout": 0.0, "batch_norm": False},
        "small": {"hidden_dims": (17, 17, 17), "activation": "relu", "dropout": 0.0, "batch_norm": False},
    }

    @staticmethod
    def _raise_value_error(message):
        raise ValueError(message)

    def __init__(self, input_dim, hidden_dims=(), activation="relu", batch_norm=False,
                 dropout=0.0):
        super().__init__()
        if not isinstance(input_dim, int) or isinstance(input_dim, bool) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if not isinstance(hidden_dims, (tuple, list)):
            self._raise_value_error("hidden_dims must be a tuple or list of positive integers")
        if any(not isinstance(width, int) or isinstance(width, bool) or width <= 0
               for width in hidden_dims):
            raise ValueError("hidden_dims must contain only positive integers")
        if activation not in ("relu", "selu"):
            raise ValueError("activation must be 'relu' or 'selu'")
        if not isinstance(batch_norm, bool):
            self._raise_value_error("batch_norm must be a boolean")
        if (not isinstance(dropout, (int, float)) or isinstance(dropout, bool)
                or not math.isfinite(dropout) or not 0.0 <= dropout < 1.0):
            raise ValueError("dropout must be a finite number in [0, 1)")

        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = float(dropout)
        self.hidden_layers = nn.ModuleList()
        width = input_dim
        for hidden_width in self.hidden_dims:
            linear = nn.Linear(width, hidden_width)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers = [linear, nn.ReLU() if activation == "relu" else nn.SELU()]
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_width))
            if dropout:
                layers.append(nn.Dropout(dropout))
            self.hidden_layers.append(nn.Sequential(*layers))
            width = hidden_width
        self.output_layer = nn.Linear(width, 1)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            self._raise_value_error("x must be a two-dimensional tensor")
        if x.ndim != 2:
            raise ValueError("x must be a two-dimensional tensor with shape [B, input_dim]")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"x must have input_dim={self.input_dim} features")
        if not x.is_floating_point():
            raise ValueError("x must be a floating-point tensor")
        hidden = x
        for layer in self.hidden_layers:
            hidden = layer(hidden)
        return self.output_layer(hidden).squeeze(-1)

    @staticmethod
    def negative_log_partial_likelihood(risk_scores, event_times, event_indicators):
        named = (("risk_scores", risk_scores), ("event_times", event_times),
                 ("event_indicators", event_indicators))
        for name, value in named:
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional tensor")
        if not risk_scores.is_floating_point():
            raise ValueError("risk_scores must be a floating-point tensor")
        if not (risk_scores.shape[0] == event_times.shape[0] == event_indicators.shape[0]):
            raise ValueError("risk_scores, event_times, and event_indicators must have the same number of rows")
        if risk_scores.numel() == 0:
            raise ValueError("the cohort must be non-empty")
        if not torch.all((event_indicators == 0) | (event_indicators == 1)):
            raise ValueError("event_indicators must be binary")
        observed = event_indicators.bool()
        if not observed.any():
            raise ValueError("the cohort must contain at least one observed event")
        event_risks = risk_scores[observed]
        event_times_observed = event_times[observed]
        log_denominators = torch.stack([
            torch.logsumexp(risk_scores[event_times >= time], dim=0)
            for time in event_times_observed
        ])
        return (log_denominators - event_risks).sum()
