import torch
from torch import nn


class WAN(nn.Module):
    architecture_id = "wan"
    PRESETS = {
        "tiny": {"hidden_dim": 8, "depth": 1},
        "small": {"hidden_dim": 16, "depth": 2},
    }

    def __init__(self, input_dim, hidden_dim=8, depth=1):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("hidden_dim", hidden_dim),
            ("depth", depth),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.primal = self._make_field(input_dim, hidden_dim, depth)
        self.test = self._make_field(input_dim, hidden_dim, depth)

    @staticmethod
    def _make_field(input_dim, hidden_dim, depth):
        layers = []
        width = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(width, hidden_dim), nn.Tanh()))
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _source_values(source, batch_size, reference):
        try:
            values = torch.as_tensor(source, dtype=reference.dtype, device=reference.device)
        except (TypeError, ValueError) as error:
            raise ValueError("source must be a scalar or have shape [batch] or [batch, 1]") from error
        if values.ndim == 0:
            values = values.expand(batch_size)
        elif values.ndim == 2 and values.shape == (batch_size, 1):
            values = values[:, 0]
        if values.ndim != 1 or values.shape[0] != batch_size:
            raise ValueError("source must be a scalar or have shape [batch] or [batch, 1]")
        if not torch.isfinite(values).all():
            raise ValueError("source must contain only finite values")
        return values.unsqueeze(-1)

    def forward(self, coordinates, source):
        if not isinstance(coordinates, torch.Tensor) or coordinates.ndim != 2:
            raise ValueError("coordinates must be a rank-2 tensor with shape [batch, input_dim]")
        if coordinates.shape[0] == 0:
            raise ValueError("coordinates must contain at least one point")
        if coordinates.shape[1] != self.input_dim:
            raise ValueError(f"coordinates last dimension must equal input_dim={self.input_dim}")
        if not coordinates.is_floating_point():
            raise ValueError("coordinates must have a floating-point dtype")
        if not torch.isfinite(coordinates).all():
            raise ValueError("coordinates must contain only finite values")
        coordinates = (
            coordinates if coordinates.requires_grad else coordinates.detach().requires_grad_(True)
        )
        with torch.enable_grad():
            field = self.primal(coordinates)
            test_field = self.test(coordinates)
            field_gradient = torch.autograd.grad(
                field.sum(), coordinates, create_graph=True
            )[0]
            test_gradient = torch.autograd.grad(
                test_field.sum(), coordinates, create_graph=True
            )[0]
            source_values = self._source_values(source, coordinates.shape[0], field)
            integrand = (field_gradient * test_gradient).sum(dim=-1, keepdim=True)
            integrand = integrand - source_values * test_field
        return field, integrand
