import math

import torch
from torch import nn


class DeepRitz(nn.Module):
    architecture_id = "deep-ritz"
    PRESETS = {
        "tiny": {"input_dim": 2, "hidden_dim": 4, "depth": 1},
        "small": {"input_dim": 2, "hidden_dim": 10, "depth": 4},
    }

    def __init__(
        self,
        input_dim=2,
        hidden_dim=10,
        depth=4,
        reaction_coefficient=0.0,
        boundary_weight=500.0,
    ):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("hidden_dim", hidden_dim),
            ("depth", depth),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._validate_nonnegative_real("reaction_coefficient", reaction_coefficient)
        self._validate_nonnegative_real("boundary_weight", boundary_weight)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.reaction_coefficient = float(reaction_coefficient)
        self.boundary_weight = float(boundary_weight)
        self.input_projection = (
            nn.Linear(input_dim, hidden_dim) if input_dim > hidden_dim else None
        )
        self.blocks = nn.ModuleList(
            nn.ModuleList((nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)))
            for _ in range(depth)
        )
        self.output = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _validate_nonnegative_real(name, value):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be a finite nonnegative number")

    @staticmethod
    def _activation(x):
        return torch.clamp_min(x.pow(3), 0.0)

    def forward(self, coordinates):
        if not isinstance(coordinates, torch.Tensor) or coordinates.ndim != 2:
            raise ValueError("coordinates must be a rank-2 tensor with shape [batch, input_dim]")
        if coordinates.shape[0] == 0:
            raise ValueError("coordinates must contain at least one point")
        if coordinates.shape[1] != self.input_dim:
            raise ValueError(f"coordinates last dimension must equal input_dim={self.input_dim}")
        if not coordinates.is_floating_point():
            raise ValueError("coordinates must have a floating-point dtype")
        if self.input_projection is not None:
            state = self.input_projection(coordinates)
        elif self.input_dim < self.hidden_dim:
            state = torch.nn.functional.pad(coordinates, (0, self.hidden_dim - self.input_dim))
        else:
            state = coordinates
        for first, second in self.blocks:
            state = self._activation(second(self._activation(first(state)))) + state
        return self.output(state)

    @staticmethod
    def _weights(weights, count, reference, name):
        if weights is None:
            return reference.new_full((count,), 1.0 / count)
        result = torch.as_tensor(weights, dtype=reference.dtype, device=reference.device)
        if result.ndim == 2 and result.shape == (count, 1):
            result = result[:, 0]
        if result.ndim != 1 or result.shape[0] != count:
            raise ValueError(f"{name} must have shape [{count}] or [{count}, 1]")
        if not torch.isfinite(result).all():
            raise ValueError(f"{name} must contain only finite values")
        return result

    @staticmethod
    def _forcing(forcing, count, reference):
        if forcing is None:
            return reference.new_zeros((count,))
        result = torch.as_tensor(forcing, dtype=reference.dtype, device=reference.device)
        if result.ndim == 0:
            result = result.expand(count)
        elif result.ndim == 2 and result.shape == (count, 1):
            result = result[:, 0]
        if result.ndim != 1 or result.shape[0] != count:
            raise ValueError(f"forcing must be a scalar or have shape [{count}] or [{count}, 1]")
        if not torch.isfinite(result).all():
            raise ValueError("forcing must contain only finite values")
        return result

    def energy(
        self,
        interior_coordinates,
        forcing=None,
        interior_weights=None,
        boundary_coordinates=None,
        boundary_weights=None,
        boundary_weight=None,
    ):
        if not isinstance(interior_coordinates, torch.Tensor) or interior_coordinates.ndim != 2:
            raise ValueError("interior_coordinates must have shape [batch, input_dim]")
        if interior_coordinates.shape[0] == 0:
            raise ValueError("interior_coordinates must contain at least one point")
        coordinates = (
            interior_coordinates
            if interior_coordinates.requires_grad
            else interior_coordinates.detach().requires_grad_(True)
        )
        with torch.enable_grad():
            field = self(coordinates)
            coordinate_gradient = torch.autograd.grad(
                field,
                coordinates,
                grad_outputs=torch.ones_like(field),
                create_graph=True,
            )[0]
            source = self._forcing(forcing, coordinates.shape[0], field)
            density = (
                0.5 * coordinate_gradient.square().sum(dim=-1)
                + 0.5 * self.reaction_coefficient * field[:, 0].square()
                - source * field[:, 0]
            )
            weights = self._weights(
                interior_weights, coordinates.shape[0], density, "interior_weights"
            )
            result = (weights * density).sum()
            if boundary_coordinates is not None:
                if not isinstance(boundary_coordinates, torch.Tensor) or boundary_coordinates.ndim != 2:
                    raise ValueError("boundary_coordinates must have shape [batch, input_dim]")
                if boundary_coordinates.shape[0] == 0:
                    raise ValueError("boundary_coordinates must contain at least one point")
                boundary_field = self(boundary_coordinates)
                weights = self._weights(
                    boundary_weights,
                    boundary_coordinates.shape[0],
                    boundary_field,
                    "boundary_weights",
                )
                beta = self.boundary_weight if boundary_weight is None else boundary_weight
                self._validate_nonnegative_real("boundary_weight", beta)
                result = result + float(beta) * (weights * boundary_field[:, 0].square()).sum()
            elif boundary_weight is not None:
                self._validate_nonnegative_real("boundary_weight", boundary_weight)
        return result
