import math

import torch
from torch import nn


class Highway(nn.Module):
    architecture_id = "highway"
    PRESETS = {
        "tiny": {
            "input_size": 8,
            "hidden_size": 8,
            "num_classes": 4,
            "num_layers": 2,
            "activation": "relu",
            "gate_bias_init": -1.0,
        },
        "small": {
            "input_size": 784,
            "hidden_size": 50,
            "num_classes": 10,
            "num_layers": 9,
            "activation": "relu",
            "gate_bias_init": -1.0,
        },
        "base": {
            "input_size": 784,
            "hidden_size": 128,
            "num_classes": 10,
            "num_layers": 32,
            "activation": "relu",
            "gate_bias_init": -1.0,
        },
    }

    def __init__(
        self,
        input_size=784,
        hidden_size=50,
        num_classes=10,
        num_layers=9,
        activation="relu",
        gate_bias_init=-1.0,
    ):
        super().__init__()
        dimensions = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_classes": num_classes,
            "num_layers": num_layers,
        }
        for name, value in dimensions.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if activation not in ("relu", "tanh"):
            raise ValueError("activation must be 'relu' or 'tanh'")
        if (
            isinstance(gate_bias_init, bool)
            or not isinstance(gate_bias_init, (int, float))
            or not math.isfinite(gate_bias_init)
        ):
            raise ValueError("gate_bias_init must be a finite number")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.activation = activation
        self.gate_bias_init = float(gate_bias_init)
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.transform_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        for _ in range(num_layers):
            transform = nn.Linear(hidden_size, hidden_size)
            nn.init.kaiming_normal_(transform.weight, nonlinearity=activation)
            gate = nn.Linear(hidden_size, hidden_size)
            with torch.no_grad():
                gate.bias.fill_(self.gate_bias_init)
            self.transform_layers.append(transform)
            self.gate_layers.append(gate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim < 1:
            raise ValueError("x must have at least one dimension with input features last")
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"x last dimension must equal input_size ({self.input_size})"
            )

        hidden = self.input_projection(x)
        for transform_layer, gate_layer in zip(
            self.transform_layers, self.gate_layers
        ):
            transformed = transform_layer(hidden)
            if self.activation == "relu":
                transformed = torch.relu(transformed)
            else:
                transformed = torch.tanh(transformed)
            gate = torch.sigmoid(gate_layer(hidden))
            hidden = transformed * gate + hidden * (1.0 - gate)
        return self.classifier(hidden)
