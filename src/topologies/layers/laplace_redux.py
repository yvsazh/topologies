import math
from numbers import Real

import torch
from torch import nn


class LaplaceRedux(nn.Module):
    component_id = "laplace-redux"

    def __init__(self, prior_precision):
        super().__init__()
        if isinstance(prior_precision, bool) or not isinstance(prior_precision, Real):
            precision = math.nan
        else:
            try:
                precision = float(prior_precision)
            except OverflowError:
                precision = math.inf
        if not math.isfinite(precision) or precision <= 0:
            raise ValueError("prior_precision must be finite and greater than zero")
        self.prior_precision = precision

    def forward(self, map_mean, curvature_diag, epsilon=None):
        if not isinstance(map_mean, torch.Tensor) or not map_mean.is_floating_point():
            raise ValueError("map_mean must be a floating-point tensor")
        if not isinstance(curvature_diag, torch.Tensor) or not curvature_diag.is_floating_point():
            raise ValueError("curvature_diag must be a floating-point tensor")
        if curvature_diag.shape != map_mean.shape:
            raise ValueError("curvature_diag must have the same shape as map_mean")
        if curvature_diag.device != map_mean.device or curvature_diag.dtype != map_mean.dtype:
            raise ValueError("curvature_diag must have the same device and dtype as map_mean")
        if not torch.isfinite(map_mean).all() or not torch.isfinite(curvature_diag).all():
            raise ValueError("map_mean and curvature_diag must contain only finite values")
        if (curvature_diag < 0).any():
            raise ValueError("curvature_diag must be nonnegative")
        if epsilon is None:
            epsilon = torch.randn_like(map_mean)
        elif not isinstance(epsilon, torch.Tensor) or not epsilon.is_floating_point():
            raise ValueError("epsilon must be a floating-point tensor")
        if epsilon.shape != map_mean.shape:
            raise ValueError("epsilon must have the same shape as map_mean")
        if epsilon.device != map_mean.device or epsilon.dtype != map_mean.dtype:
            raise ValueError("epsilon must have the same device and dtype as map_mean")
        if not torch.isfinite(epsilon).all():
            raise ValueError("epsilon must contain only finite values")
        posterior_variance = torch.reciprocal(curvature_diag + self.prior_precision)
        return map_mean + torch.sqrt(posterior_variance) * epsilon
