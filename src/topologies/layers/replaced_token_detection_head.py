import math

import torch
from torch import nn


class ReplacedTokenDetectionHead(nn.Module):
    component_id = "replaced-token-detection-head"
    PRESETS = {"tiny": {"hidden_size": 32}, "small": {"hidden_size": 128},
               "base": {"hidden_size": 256}}

    def __init__(self, hidden_size: int, initializer_range: float = 0.02,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if (isinstance(initializer_range, bool) or not isinstance(initializer_range, (int, float))
                or not math.isfinite(initializer_range) or initializer_range <= 0):
            raise ValueError("initializer_range must be positive and finite")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) \
                or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        self.hidden_size = hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)
        for layer in (self.dense, self.output):
            nn.init.trunc_normal_(layer.weight, std=initializer_range,
                                  a=-2 * initializer_range, b=2 * initializer_range)
            nn.init.zeros_(layer.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden must have shape [batch, time, hidden_size]")
        return self.output(self.dropout(self.activation(self.dense(hidden)))).squeeze(-1)
