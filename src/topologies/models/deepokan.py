import math
from collections.abc import Sequence
from itertools import pairwise
from numbers import Real

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


def _grid_value(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _rbf_features(inputs, centers, beta):
    return torch.exp(-((inputs.unsqueeze(-1) - centers) / beta).square()).flatten(start_dim=-2)


def _build_stack(input_dim, hidden_dims, output_dim, num_centers, grid_min, grid_max, learnable_centers):
    dimensions = (input_dim, *hidden_dims, output_dim)
    layers = nn.ModuleList()
    centers = nn.ParameterList()
    for in_dim, out_dim in pairwise(dimensions):
        layers.append(nn.Linear(in_dim * num_centers, out_dim, bias=False))
        grid = torch.linspace(grid_min, grid_max, num_centers)
        centers.append(nn.Parameter(grid, requires_grad=learnable_centers))
    return layers, centers


def _encode(inputs, layers, centers, beta):
    for layer, grid in zip(layers, centers):
        inputs = layer(_rbf_features(inputs, grid, beta))
    return inputs


class DeepOKAN(nn.Module):
    architecture_id = "deepokan"
    PRESETS = {
        "tiny": {
            "branch_input_dim": 3,
            "coordinate_dim": 1,
            "latent_dim": 4,
            "branch_hidden_dims": [4],
            "trunk_hidden_dims": [4],
            "num_centers": 4,
            "grid_min": -1.0,
            "grid_max": 1.0,
            "learnable_centers": True,
        },
        "small": {
            "branch_input_dim": 6,
            "coordinate_dim": 2,
            "latent_dim": 16,
            "branch_hidden_dims": [16],
            "trunk_hidden_dims": [16],
            "num_centers": 8,
            "grid_min": -1.0,
            "grid_max": 1.0,
            "learnable_centers": True,
        },
    }

    def __init__(
        self,
        branch_input_dim,
        coordinate_dim,
        latent_dim,
        branch_hidden_dims,
        trunk_hidden_dims,
        num_centers,
        grid_min,
        grid_max,
        learnable_centers=True,
    ):
        super().__init__()
        self.branch_input_dim = _positive_int(branch_input_dim, "branch_input_dim")
        self.coordinate_dim = _positive_int(coordinate_dim, "coordinate_dim")
        self.latent_dim = _positive_int(latent_dim, "latent_dim")
        self.branch_hidden_dims = _hidden_dims(branch_hidden_dims, "branch_hidden_dims")
        self.trunk_hidden_dims = _hidden_dims(trunk_hidden_dims, "trunk_hidden_dims")
        self.num_centers = _positive_int(num_centers, "num_centers")
        if self.num_centers < 2:
            raise ValueError("num_centers must be at least 2")
        self.grid_min = _grid_value(grid_min, "grid_min")
        self.grid_max = _grid_value(grid_max, "grid_max")
        if self.grid_max <= self.grid_min:
            raise ValueError("grid_max must be greater than grid_min")
        if not isinstance(learnable_centers, bool):
            raise _value_error("learnable_centers must be a boolean")
        self.learnable_centers = learnable_centers
        self.beta = (self.grid_max - self.grid_min) / (self.num_centers - 1)
        self.branch_layers, self.branch_centers = _build_stack(
            self.branch_input_dim,
            self.branch_hidden_dims,
            self.latent_dim,
            self.num_centers,
            self.grid_min,
            self.grid_max,
            self.learnable_centers,
        )
        self.trunk_layers, self.trunk_centers = _build_stack(
            self.coordinate_dim,
            self.trunk_hidden_dims,
            self.latent_dim,
            self.num_centers,
            self.grid_min,
            self.grid_max,
            self.learnable_centers,
        )
        self.output_bias = nn.Parameter(torch.zeros(()))

    def forward(self, function_samples, coordinates):
        if not isinstance(function_samples, torch.Tensor) or function_samples.ndim != 2:
            raise ValueError("function_samples must have shape [batch, branch_input_dim]")
        if function_samples.shape[0] == 0:
            raise ValueError("function_samples batch size must be positive")
        if function_samples.shape[1] != self.branch_input_dim:
            raise ValueError(f"function_samples width must equal branch_input_dim={self.branch_input_dim}")
        if not function_samples.is_floating_point():
            raise ValueError("function_samples must have a floating point dtype")
        if not isinstance(coordinates, torch.Tensor) or coordinates.ndim not in (2, 3):
            raise ValueError("coordinates must have shape [queries, coordinate_dim] or [batch, queries, coordinate_dim]")
        if coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError(f"coordinates width must equal coordinate_dim={self.coordinate_dim}")
        if coordinates.shape[-2] == 0:
            raise ValueError("coordinates must contain at least one query")
        if not coordinates.is_floating_point():
            raise ValueError("coordinates must have a floating point dtype")
        if coordinates.device != function_samples.device or coordinates.dtype != function_samples.dtype:
            raise ValueError("function_samples and coordinates must share device and dtype")
        branch = _encode(function_samples, self.branch_layers, self.branch_centers, self.beta)
        if coordinates.ndim == 2:
            trunk = _encode(coordinates, self.trunk_layers, self.trunk_centers, self.beta)
            output = branch @ trunk.transpose(0, 1)
        else:
            if coordinates.shape[0] != function_samples.shape[0]:
                raise ValueError("per-sample coordinates and function_samples must have matching batch sizes")
            trunk = _encode(coordinates, self.trunk_layers, self.trunk_centers, self.beta)
            output = torch.einsum("br,bqr->bq", branch, trunk)
        output = output.unsqueeze(-1)
        return output + self.output_bias
