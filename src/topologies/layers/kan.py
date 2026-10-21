import torch
from torch import nn
from torch.nn import functional as F


class KANEdge(nn.Module):
    component_id = "kan"

    def __init__(self, grid, coefficients, degree=3, base_scale=1.0, spline_scale=1.0):
        super().__init__()
        grid = torch.as_tensor(grid, dtype=torch.get_default_dtype())
        coefficients = torch.as_tensor(coefficients, dtype=grid.dtype)
        if grid.ndim != 1 or grid.numel() < 2 or not bool(torch.all(grid[1:] > grid[:-1])):
            raise ValueError("grid must be a strictly increasing one-dimensional sequence")
        if degree != 3:
            raise ValueError("this source-scoped edge implements cubic B-splines (degree=3)")
        if coefficients.ndim != 1 or coefficients.numel() != grid.numel() + degree - 1:
            raise ValueError("cubic spline needs len(grid) + 2 coefficient values")
        if not torch.isfinite(coefficients).all():
            raise ValueError("coefficients must be finite")
        interior = grid[1:-1]
        knot_vector = torch.cat((grid[:1].expand(degree + 1), interior, grid[-1:].expand(degree + 1)))
        self.register_buffer("grid", grid)
        self.register_buffer("knot_vector", knot_vector)
        self.degree = degree
        self.coefficients = nn.Parameter(coefficients.clone())
        self.base_scale = nn.Parameter(torch.tensor(float(base_scale), dtype=grid.dtype))
        self.spline_scale = nn.Parameter(torch.tensor(float(spline_scale), dtype=grid.dtype))

    def _basis(self, x):
        knots = self.knot_vector
        basis = ((x.unsqueeze(-1) >= knots[:-1]) & (x.unsqueeze(-1) < knots[1:])).to(x.dtype)
        for order in range(1, self.degree + 1):
            count = knots.numel() - order - 1
            left_start = knots[:count]
            left_end = knots[order : order + count]
            right_start = knots[1 : count + 1]
            right_end = knots[order + 1 : order + count + 1]
            left_denominator = left_end - left_start
            right_denominator = right_end - right_start
            left_weight = torch.where(
                left_denominator > 0,
                (x.unsqueeze(-1) - left_start) / left_denominator.clamp_min(torch.finfo(x.dtype).tiny),
                0.0,
            )
            right_weight = torch.where(
                right_denominator > 0,
                (right_end - x.unsqueeze(-1)) / right_denominator.clamp_min(torch.finfo(x.dtype).tiny),
                0.0,
            )
            basis = left_weight * basis[..., :count] + right_weight * basis[..., 1 : count + 1]
        at_right_edge = x == self.grid[-1]
        if bool(at_right_edge.any()):
            last = torch.zeros_like(basis)
            last[..., -1] = 1.0
            basis = torch.where(at_right_edge.unsqueeze(-1), last, basis)
        return basis

    def forward(self, x):
        x = torch.as_tensor(x, dtype=self.coefficients.dtype, device=self.coefficients.device)
        if bool(torch.any((x < self.grid[0]) | (x > self.grid[-1]))):
            raise ValueError("x is outside the fixed spline grid")
        spline = torch.matmul(self._basis(x), self.coefficients)
        return self.base_scale * F.silu(x) + self.spline_scale * spline
