import torch
from torch import nn


class NeuralSplineFlow(nn.Module):
    architecture_id = "neural-spline-flows"
    def __init__(self, dimensions=4, bins=4, bound=3.0, min_bin=1e-3, min_derivative=1e-3):
        super().__init__()
        if dimensions < 1 or bins < 2 or bound <= 0:
            raise ValueError("dimensions and bins must be positive; bins must be at least two and bound positive")
        self.dimensions = dimensions
        self.bins = bins
        self.bound = float(bound)
        self.min_bin = float(min_bin)
        self.min_derivative = float(min_derivative)
        self.width_logits = nn.Parameter(torch.zeros(dimensions, bins))
        self.height_logits = nn.Parameter(torch.zeros(dimensions, bins))
        self.derivative_logits = nn.Parameter(torch.zeros(dimensions, bins - 1))

    def _knots(self, x):
        widths = self.min_bin + (2 * self.bound - self.bins * self.min_bin) * torch.softmax(self.width_logits, dim=-1)
        heights = self.min_bin + (2 * self.bound - self.bins * self.min_bin) * torch.softmax(self.height_logits, dim=-1)
        derivatives = self.min_derivative + torch.nn.functional.softplus(self.derivative_logits)
        derivatives = torch.cat((torch.ones(self.dimensions, 1, device=x.device, dtype=x.dtype), derivatives, torch.ones(self.dimensions, 1, device=x.device, dtype=x.dtype)), dim=-1)
        widths = widths.to(dtype=x.dtype, device=x.device)
        heights = heights.to(dtype=x.dtype, device=x.device)
        derivatives = derivatives.to(dtype=x.dtype, device=x.device)
        xk = torch.cat((torch.full((self.dimensions, 1), -self.bound, device=x.device, dtype=x.dtype), -self.bound + torch.cumsum(widths, dim=-1)), dim=-1)
        yk = torch.cat((torch.full((self.dimensions, 1), -self.bound, device=x.device, dtype=x.dtype), -self.bound + torch.cumsum(heights, dim=-1)), dim=-1)
        return xk, yk, derivatives

    def _transform(self, value, inverse):
        if value.shape[-1] != self.dimensions:
            raise ValueError(f"expected last dimension {self.dimensions}")
        xk, yk, derivatives = self._knots(value)
        knots = yk if inverse else xk
        outputs = xk if inverse else yk
        idx = torch.sum(value.unsqueeze(-1) >= knots[..., 1:], dim=-1).clamp(max=self.bins - 1)
        gather = idx.unsqueeze(-1)
        x0 = torch.gather(knots.expand(value.shape[:-1] + knots.shape), -1, gather).squeeze(-1)
        y0 = torch.gather(outputs.expand(value.shape[:-1] + outputs.shape), -1, gather).squeeze(-1)
        x1 = torch.gather(knots.expand(value.shape[:-1] + knots.shape), -1, gather + 1).squeeze(-1)
        y1 = torch.gather(outputs.expand(value.shape[:-1] + outputs.shape), -1, gather + 1).squeeze(-1)
        d0 = torch.gather(derivatives.expand(value.shape[:-1] + derivatives.shape), -1, gather).squeeze(-1)
        d1 = torch.gather(derivatives.expand(value.shape[:-1] + derivatives.shape), -1, gather + 1).squeeze(-1)
        width = x1 - x0
        height = y1 - y0
        slope = height / width
        if inverse:
            delta = value - y0
            a = delta * (d0 + d1 - 2 * slope) + height * (slope - d0)
            b = height * d0 - delta * (d0 + d1 - 2 * slope)
            c = -slope * delta
            disc = (b.square() - 4 * a * c).clamp_min(0)
            theta = (2 * c) / (-b - torch.sqrt(disc)).where((-b - torch.sqrt(disc)).abs() > 1e-12, torch.full_like(b, -1e-12))
            theta = torch.where(a.abs() < 1e-10, -c / b, theta).clamp(0, 1)
        else:
            theta = ((value - x0) / width).clamp(0, 1)
        numerator = height * (slope * theta.square() + d0 * theta * (1 - theta))
        denominator = slope + (d1 + d0 - 2 * slope) * theta * (1 - theta)
        transformed = y0 + numerator / denominator
        derivative_numerator = slope.square() * (d1 * theta.square() + 2 * slope * theta * (1 - theta) + d0 * (1 - theta).square())
        derivative = derivative_numerator / denominator.square()
        if inverse:
            transformed = x0 + theta * width
            derivative = derivative.reciprocal()
        outside = (value < -self.bound) | (value > self.bound)
        result = torch.where(outside, value, transformed)
        logdet = torch.where(outside, torch.zeros_like(value), torch.log(derivative.clamp_min(1e-12)))
        return result, logdet.sum(dim=-1)

    def forward(self, value, inverse=False, return_logdet=False):
        result, logdet = self._transform(value, inverse)
        return (result, logdet) if return_logdet else result

    def inverse(self, value, return_logdet=False):
        result, logdet = self._transform(value, True)
        return (result, logdet) if return_logdet else result
