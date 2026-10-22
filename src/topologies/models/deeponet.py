from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import nn


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _value_error(message):
    return ValueError(message)


def _hidden_dims(values, name):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise _value_error(f"{name} must be a sequence of positive integers")
    return tuple(_positive_int(value, name) for value in values)


def _activation(name):
    activations = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if not isinstance(name, str) or name not in activations:
        raise ValueError(f"activation must be one of {tuple(activations)}")
    return activations[name]


def _network(input_dim, hidden_dims, output_dim, activation, *, branch_output):
    dimensions = (input_dim, *hidden_dims, output_dim)
    layers = []
    for index, (in_dim, out_dim) in enumerate(pairwise(dimensions)):
        is_output = index == len(dimensions) - 2
        layers.append(nn.Linear(in_dim, out_dim, bias=not (branch_output and is_output)))
        if not is_output or not branch_output:
            layers.append(activation())
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    architecture_id = "deeponet"
    PRESETS = {
        "tiny": {
            "branch_input_dim": 16,
            "coordinate_dim": 1,
            "latent_dim": 12,
            "branch_hidden_dims": [8],
            "trunk_hidden_dims": [8],
            "activation": "tanh",
            "use_output_bias": True,
        },
        "small": {
            "branch_input_dim": 100,
            "coordinate_dim": 1,
            "latent_dim": 40,
            "branch_hidden_dims": [40, 40],
            "trunk_hidden_dims": [40, 40, 40],
            "activation": "tanh",
            "use_output_bias": True,
        },
        "base": {
            "branch_input_dim": 100,
            "coordinate_dim": 1,
            "latent_dim": 100,
            "branch_hidden_dims": [100, 100],
            "trunk_hidden_dims": [100, 100, 100],
            "activation": "tanh",
            "use_output_bias": True,
        },
    }

    def __init__(
        self,
        branch_input_dim,
        coordinate_dim,
        latent_dim,
        branch_hidden_dims=(64, 64),
        trunk_hidden_dims=(64, 64),
        activation="tanh",
        use_output_bias=True,
    ):
        super().__init__()
        self.branch_input_dim = _positive_int(branch_input_dim, "branch_input_dim")
        self.coordinate_dim = _positive_int(coordinate_dim, "coordinate_dim")
        self.latent_dim = _positive_int(latent_dim, "latent_dim")
        self.branch_hidden_dims = _hidden_dims(branch_hidden_dims, "branch_hidden_dims")
        self.trunk_hidden_dims = _hidden_dims(trunk_hidden_dims, "trunk_hidden_dims")
        activation_type = _activation(activation)
        if not isinstance(use_output_bias, bool):
            raise _value_error("use_output_bias must be a boolean")
        self.activation = activation
        self.use_output_bias = use_output_bias
        self.branch_net = _network(
            self.branch_input_dim,
            self.branch_hidden_dims,
            self.latent_dim,
            activation_type,
            branch_output=True,
        )
        self.trunk_net = _network(
            self.coordinate_dim,
            self.trunk_hidden_dims,
            self.latent_dim,
            activation_type,
            branch_output=False,
        )
        if use_output_bias:
            self.output_bias = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("output_bias", None)

    def forward(self, function_samples, coordinates):
        if not isinstance(function_samples, torch.Tensor):
            raise _value_error("function_samples must be a tensor with shape [batch, branch_input_dim]")
        if function_samples.ndim != 2:
            raise ValueError("function_samples must have shape [batch, branch_input_dim]")
        if function_samples.shape[0] == 0:
            raise ValueError("function_samples batch size must be positive")
        if function_samples.shape[1] != self.branch_input_dim:
            raise ValueError(f"function_samples width must equal branch_input_dim={self.branch_input_dim}")
        if not function_samples.is_floating_point():
            raise ValueError("function_samples must have a floating point dtype")
        if not isinstance(coordinates, torch.Tensor):
            raise _value_error("coordinates must be a tensor with shape [queries, coordinate_dim] or [batch, queries, coordinate_dim]")
        if coordinates.ndim not in (2, 3):
            raise ValueError("coordinates must have shape [queries, coordinate_dim] or [batch, queries, coordinate_dim]")
        if coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError(f"coordinates width must equal coordinate_dim={self.coordinate_dim}")
        if coordinates.shape[-2] == 0:
            raise ValueError("coordinates must contain at least one query")
        if not coordinates.is_floating_point():
            raise ValueError("coordinates must have a floating point dtype")
        if coordinates.device != function_samples.device or coordinates.dtype != function_samples.dtype:
            raise ValueError("function_samples and coordinates must share device and dtype")
        branch_features = self.branch_net(function_samples)
        if coordinates.ndim == 2:
            trunk_features = self.trunk_net(coordinates)
            output = torch.einsum("bp,qp->bq", branch_features, trunk_features)
        else:
            if coordinates.shape[0] != function_samples.shape[0]:
                raise ValueError("per-sample coordinates and function_samples must have matching batch sizes")
            trunk_features = self.trunk_net(coordinates)
            output = torch.einsum("bp,bqp->bq", branch_features, trunk_features)
        output = output.unsqueeze(-1)
        if self.output_bias is not None:
            output = output + self.output_bias
        return output
