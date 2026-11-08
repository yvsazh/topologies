import math

import torch
from torch import nn


def valid_distance(value):
    try:
        distance = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("distance must be a finite non-negative number") from error
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("distance must be a finite non-negative number")
    return distance


def mahalanobis_distance(values, mean, precision):
    values = torch.as_tensor(values, dtype=torch.float64)
    mean = torch.as_tensor(mean, dtype=torch.float64, device=values.device)
    precision = torch.as_tensor(precision, dtype=torch.float64, device=values.device)
    if values.ndim not in (1, 2) or mean.ndim != 1:
        raise ValueError("values must be a vector or matrix and mean a vector")
    dimension = mean.numel()
    if values.shape[-1] != dimension or precision.shape != (dimension, dimension):
        raise ValueError("values, mean, and precision dimensions must match")
    if not torch.isfinite(values).all() or not torch.isfinite(mean).all() or not torch.isfinite(precision).all():
        raise ValueError("inputs must be finite")
    if not torch.allclose(precision, precision.transpose(-1, -2)):
        raise ValueError("precision must be symmetric")
    delta = values - mean
    squared = torch.einsum("...i,ij,...j->...", delta, precision, delta)
    if not torch.isfinite(squared).all() or torch.any(squared < 0):
        raise ValueError("computed squared distance must be finite and non-negative")
    distances = torch.sqrt(squared)
    for item in distances.reshape(-1):
        valid_distance(item.item())
    return distances


class Mahalanobis(nn.Module):
    component_id = "mahalanobis"

    def __init__(self, mean=None, precision=None, *, class_means=None, covariance=None):
        super().__init__()
        confidence_requested = class_means is not None or covariance is not None
        if confidence_requested:
            if mean is not None or precision is not None or class_means is None or covariance is None:
                raise ValueError("provide class_means and covariance together")
            self._initialize_confidence(class_means, covariance)
            return
        if mean is None or precision is None:
            raise ValueError("provide mean and precision or class_means and covariance")
        try:
            mean_tensor = torch.as_tensor(mean)
            precision_tensor = torch.as_tensor(precision)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("mean and precision must be compatible tensors or arrays") from error
        if mean_tensor.ndim == 2:
            self._initialize_confidence(mean_tensor, precision_tensor)
            return
        self._initialize_distance(mean_tensor, precision_tensor)

    def _initialize_distance(self, mean, precision):
        mean_tensor = torch.as_tensor(mean, dtype=torch.float64)
        precision_tensor = torch.as_tensor(precision, dtype=torch.float64)
        if mean_tensor.ndim != 1 or precision_tensor.shape != (mean_tensor.numel(), mean_tensor.numel()):
            raise ValueError("mean and precision dimensions must match")
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(precision_tensor).all():
            raise ValueError("mean and precision must be finite")
        if not torch.allclose(precision_tensor, precision_tensor.transpose(-1, -2)):
            raise ValueError("precision must be symmetric")
        self.score_mode = "distance"
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("precision", precision_tensor)

    def _initialize_confidence(self, class_means, covariance):
        try:
            class_means = torch.as_tensor(class_means)
            covariance = torch.as_tensor(covariance)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("class_means and covariance must be finite floating-point tensors or arrays") from error
        if not class_means.is_floating_point() or not covariance.is_floating_point():
            raise ValueError("class_means and covariance must be floating-point tensors")
        if class_means.ndim != 2 or class_means.shape[0] == 0 or class_means.shape[1] == 0:
            raise ValueError("class_means must have shape [num_classes, feature_dim] with positive dimensions")
        if covariance.ndim != 2 or covariance.shape[0] == 0 or covariance.shape[1] == 0:
            raise ValueError("covariance must have shape [feature_dim, feature_dim] with positive dimensions")
        feature_dim = class_means.shape[1]
        if covariance.shape != (feature_dim, feature_dim):
            raise ValueError("class_means and covariance must have matching feature dimensions")
        if class_means.device != covariance.device:
            raise ValueError("class_means and covariance must be on the same device")
        if not torch.isfinite(class_means).all() or not torch.isfinite(covariance).all():
            raise ValueError("class_means and covariance must contain only finite values")

        dtype = torch.promote_types(class_means.dtype, covariance.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        means = class_means.detach().to(dtype=dtype).clone()
        covariance_value = covariance.detach().to(dtype=dtype).clone()
        if not torch.allclose(covariance_value, covariance_value.transpose(-1, -2), rtol=1e-5, atol=1e-6):
            raise ValueError("covariance must be symmetric")

        covariance_cpu = covariance_value.to(device="cpu", dtype=torch.float64)
        cholesky, info = torch.linalg.cholesky_ex(covariance_cpu)
        if torch.any(info != 0):
            raise ValueError("covariance must be positive definite")
        precision_value = torch.cholesky_inverse(cholesky).to(device=means.device, dtype=dtype)

        self.score_mode = "confidence"
        self.register_buffer("class_means", means)
        self.register_buffer("covariance", covariance_value)
        self.register_buffer("precision", precision_value)

    def forward(self, x):
        if self.score_mode == "distance":
            return mahalanobis_distance(x, self.mean, self.precision)
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise ValueError("x must be a floating-point tensor with shape [batch, feature_dim]")
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.class_means.shape[1]:
            raise ValueError("x must have shape [batch, feature_dim] matching class_means")
        if x.device != self.class_means.device:
            raise ValueError("x and the stored statistics must be on the same device")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        dtype = torch.promote_types(torch.promote_types(x.dtype, self.class_means.dtype), torch.float32)
        differences = x.to(dtype=dtype).unsqueeze(1) - self.class_means.to(dtype=dtype).unsqueeze(0)
        precision = self.precision.to(dtype=dtype)
        quadratic = torch.einsum("bci,ij,bcj->bc", differences, precision, differences)
        return -quadratic.min(dim=1).values
