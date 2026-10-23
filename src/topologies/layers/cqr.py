import math
from numbers import Real

import torch
from torch import nn


class ConformalizedQuantileRegression(nn.Module):
    component_id = "cqr"

    def __init__(self, alpha=0.1):
        super().__init__()
        if (not isinstance(alpha, Real) or isinstance(alpha, bool)
                or not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0):
            raise ValueError("alpha must be a finite number in (0, 1)")
        self.alpha = float(alpha)
        self.register_buffer("correction", torch.zeros((), dtype=torch.float64))
        self.register_buffer("_calibrated", torch.tensor(False, dtype=torch.bool))

    def calibrate(self, calibration_quantiles, calibration_targets):
        if (not isinstance(calibration_quantiles, torch.Tensor)
                or calibration_quantiles.ndim != 2 or calibration_quantiles.shape[1] != 2
                or calibration_quantiles.shape[0] == 0
                or not calibration_quantiles.is_floating_point()):
            raise ValueError("calibration_quantiles must be a nonempty floating tensor shaped [M, 2]")
        if not isinstance(calibration_targets, torch.Tensor):
            raise TypeError("calibration_targets must be a tensor shaped [M] or [M, 1]")
        if calibration_targets.ndim == 2 and calibration_targets.shape[1] == 1:
            calibration_targets = calibration_targets.squeeze(-1)
        if calibration_targets.ndim != 1 or calibration_targets.shape[0] != calibration_quantiles.shape[0]:
            raise ValueError("calibration_targets must match calibration_quantiles along M")
        if not calibration_targets.is_floating_point():
            raise ValueError("calibration_targets must be floating point")
        targets = calibration_targets.to(device=calibration_quantiles.device,
                                         dtype=calibration_quantiles.dtype)
        if (not bool(torch.isfinite(calibration_quantiles).all())
                or not bool(torch.isfinite(targets).all())):
            raise ValueError("calibration inputs must contain only finite values")
        scores = torch.maximum(calibration_quantiles[:, 0] - targets,
                               targets - calibration_quantiles[:, 1])
        rank = math.ceil((scores.numel() + 1) * (1.0 - self.alpha))
        if rank > scores.numel():
            correction = scores.new_tensor(float("inf"))
        else:
            correction = torch.kthvalue(scores, rank).values
        with torch.no_grad():
            self.correction.copy_(correction.to(device=self.correction.device,
                                                dtype=self.correction.dtype))
            self._calibrated.fill_(True)
        return self

    def forward(self, query_quantiles, calibration_quantiles=None, calibration_targets=None):
        if calibration_quantiles is not None or calibration_targets is not None:
            if calibration_quantiles is None or calibration_targets is None:
                raise ValueError("both calibration_quantiles and calibration_targets are required")
            self.calibrate(calibration_quantiles, calibration_targets)
        if not isinstance(query_quantiles, torch.Tensor) or query_quantiles.ndim < 1:
            raise ValueError("query_quantiles must be a tensor shaped [..., 2]")
        if query_quantiles.shape[-1] != 2 or not query_quantiles.is_floating_point():
            raise ValueError("query_quantiles must be a floating tensor shaped [..., 2]")
        if not bool(torch.isfinite(query_quantiles).all()):
            raise ValueError("query_quantiles must contain only finite values")
        if not bool(self._calibrated):
            raise ValueError("calibrate the module before requesting prediction intervals")
        correction = self.correction.to(device=query_quantiles.device, dtype=query_quantiles.dtype)
        lower = query_quantiles[..., 0] - correction
        upper = query_quantiles[..., 1] + correction
        return torch.stack((lower, upper), dim=-1)
