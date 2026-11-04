import math

import torch
from torch import nn


class FourierFeatures(nn.Module):
    architecture_id = "fourier-features"
    PRESETS = {
        "tiny": {
            "input_dim": 2,
            "output_dim": 3,
            "num_frequencies": 16,
            "sigma": 10.0,
            "hidden_dim": 32,
            "num_layers": 2,
            "activation": "relu",
            "output_activation": "sigmoid",
        },
        "small": {
            "input_dim": 2,
            "output_dim": 3,
            "num_frequencies": 64,
            "sigma": 10.0,
            "hidden_dim": 128,
            "num_layers": 4,
            "activation": "relu",
            "output_activation": "sigmoid",
        },
        "base": {
            "input_dim": 2,
            "output_dim": 3,
            "num_frequencies": 256,
            "sigma": 10.0,
            "hidden_dim": 256,
            "num_layers": 4,
            "activation": "relu",
            "output_activation": "sigmoid",
        },
    }

    def __init__(
        self,
        input_dim,
        output_dim,
        num_frequencies=256,
        sigma=10.0,
        hidden_dim=256,
        num_layers=4,
        activation="relu",
        output_activation="sigmoid",
    ):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("output_dim", output_dim),
            ("num_frequencies", num_frequencies),
            ("hidden_dim", hidden_dim),
            ("num_layers", num_layers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if sigma.__class__ not in (int, float):
            raise ValueError("sigma must be a positive finite number")
        try:
            sigma = float(sigma)
        except OverflowError:
            raise ValueError("sigma must be a positive finite number") from None
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be a positive finite number")
        if activation != "relu":
            raise ValueError("activation must be 'relu'")
        if output_activation not in ("sigmoid", "identity"):
            raise ValueError("output_activation must be 'sigmoid' or 'identity'")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_frequencies = num_frequencies
        self.sigma = sigma
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.output_activation = output_activation
        self.register_buffer("B", torch.randn(num_frequencies, input_dim) * self.sigma)

        widths = [2 * num_frequencies]
        if num_layers > 1:
            widths.extend([hidden_dim] * (num_layers - 1))
        widths.append(output_dim)
        self.layers = nn.ModuleList(
            nn.Linear(widths[index], widths[index + 1])
            for index in range(num_layers)
        )

    def encode(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim < 1:
            raise ValueError("x must be a tensor with a final coordinate dimension")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"x must have final dimension {self.input_dim}")
        if not x.is_floating_point():
            raise ValueError("x must have a floating-point dtype")
        projection = torch.matmul(x, self.B.transpose(0, 1)) * (2.0 * math.pi)
        return torch.cat((torch.cos(projection), torch.sin(projection)), dim=-1)

    def forward(self, x):
        values = self.encode(x)
        for index, layer in enumerate(self.layers):
            values = layer(values)
            if index < len(self.layers) - 1:
                values = torch.relu(values)
        if self.output_activation == "sigmoid":
            values = torch.sigmoid(values)
        return values
