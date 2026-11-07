import math

import torch
from torch import nn
from torch.nn import functional as F


class GPTQLinear(nn.Module):
    component_id = "gptq"

    def __init__(self, in_features, out_features, bits=4, block_size=128, damping=0.01,
                 bias=True):
        super().__init__()
        if not isinstance(in_features, int) or isinstance(in_features, bool) or in_features <= 0:
            raise ValueError("in_features must be a positive integer")
        if not isinstance(out_features, int) or isinstance(out_features, bool) or out_features <= 0:
            raise ValueError("out_features must be a positive integer")
        if not isinstance(bits, int) or isinstance(bits, bool) or not 2 <= bits <= 8:
            raise ValueError("bits must be an integer from 2 through 8")
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if isinstance(damping, bool) or not isinstance(damping, (int, float)) or \
                not math.isfinite(damping) or damping < 0:
            raise ValueError("damping must be a finite nonnegative number")
        if bias is not True and bias is not False:
            raise ValueError("bias must be a boolean")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.block_size = block_size
        self.damping = float(damping)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def quantize(self, calibration_inputs):
        if not isinstance(calibration_inputs, torch.Tensor) or calibration_inputs.ndim != 2:
            raise ValueError("calibration_inputs must be a 2D tensor")
        if calibration_inputs.shape[0] == 0 or calibration_inputs.shape[1] != self.in_features:
            raise ValueError("calibration_inputs must have shape [n_calibration, in_features]")
        if not torch.isfinite(calibration_inputs).all():
            raise ValueError("calibration_inputs must contain only finite values")
        if not torch.isfinite(self.weight).all():
            raise ValueError("weight must contain only finite values")
        dtype = torch.float64 if self.weight.dtype == torch.float64 else torch.float32
        inputs = calibration_inputs.detach().to(device=self.weight.device, dtype=dtype)
        if inputs.square().sum() <= 0:
            raise ValueError("calibration_inputs must have nonzero energy")
        weights = self.weight.detach().to(dtype=dtype).clone()
        hessian = 2.0 * inputs.mT.matmul(inputs)
        mean_diagonal = hessian.diagonal().mean()
        if not torch.isfinite(mean_diagonal) or mean_diagonal <= 0:
            raise ValueError("calibration_inputs must produce a finite positive Hessian")
        hessian.diagonal().add_(self.damping * mean_diagonal)
        try:
            inverse = torch.linalg.inv(hessian)
            upper = torch.linalg.cholesky(inverse).mT
        except RuntimeError as error:
            raise ValueError("damped calibration Hessian must be positive definite") from error
        if not torch.isfinite(upper).all() or torch.any(upper.diagonal() <= 0):
            raise ValueError("damped calibration Hessian must be positive definite")
        levels = (1 << self.bits) - 1
        minima = weights.amin(dim=1)
        maxima = weights.amax(dim=1)
        scales = (maxima - minima) / levels
        safe_scales = torch.where(scales > 0, scales, torch.ones_like(scales))
        quantized = torch.empty_like(weights)
        n_features = self.in_features
        for start in range(0, n_features, self.block_size):
            stop = min(start + self.block_size, n_features)
            errors = torch.empty((self.out_features, stop - start), device=weights.device,
                                 dtype=weights.dtype)
            for column in range(start, stop):
                normalized = (weights[:, column] - minima) / safe_scales
                indices = normalized.round().clamp_(0, levels)
                values = minima + scales * indices
                quantized[:, column] = values
                error = (weights[:, column] - values) / upper[column, column]
                errors[:, column - start] = error
                weights[:, column:stop] -= error[:, None] * upper[column, column:stop]
            if stop < n_features:
                weights[:, stop:] -= errors.matmul(upper[start:stop, stop:])
        with torch.no_grad():
            self.weight.copy_(quantized.to(dtype=self.weight.dtype))
        return self.weight

    def forward(self, inputs):
        if inputs.shape[-1] != self.in_features:
            raise ValueError("the last input dimension must match in_features")
        return F.linear(inputs, self.weight, self.bias)
