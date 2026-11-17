import math

import torch
from torch import nn
from torch.nn import functional as F


class SparseGPT(nn.Module):
    component_id = "sparsegpt"

    def __init__(
        self,
        in_features,
        out_features,
        sparsity=0.5,
        block_size=128,
        mask_block_size=128,
        damping=0.01,
        n=None,
        m=None,
        bias=True,
    ):
        super().__init__()
        if not isinstance(in_features, int) or isinstance(in_features, bool) or in_features <= 0:
            raise ValueError("in_features must be a positive integer")
        if not isinstance(out_features, int) or isinstance(out_features, bool) or out_features <= 0:
            raise ValueError("out_features must be a positive integer")
        if isinstance(sparsity, bool) or not isinstance(sparsity, (int, float)) or \
                not math.isfinite(sparsity) or not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must be a finite number in [0, 1)")
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if not isinstance(mask_block_size, int) or isinstance(mask_block_size, bool) or \
                mask_block_size <= 0:
            raise ValueError("mask_block_size must be a positive integer")
        if isinstance(damping, bool) or not isinstance(damping, (int, float)) or \
                not math.isfinite(damping) or damping < 0:
            raise ValueError("damping must be a finite nonnegative number")
        if bias is not True and bias is not False:
            raise ValueError("bias must be a boolean")
        if (n is None) != (m is None):
            raise ValueError("n and m must be supplied together")
        if n is not None:
            if not isinstance(n, int) or isinstance(n, bool) or not isinstance(m, int) or \
                    isinstance(m, bool) or not 0 < n < m:
                raise ValueError("n and m must be integers satisfying 0 < n < m")
            if in_features % m != 0:
                raise ValueError("in_features must be divisible by m for n:m sparsity")
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = float(sparsity)
        self.block_size = block_size
        self.mask_block_size = mask_block_size
        self.damping = float(damping)
        self.n = n
        self.m = m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.register_buffer("mask", torch.ones(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        self.mask.fill_(1)

    def _select_mask(self, scores):
        mask = torch.ones_like(scores)
        flat_scores = scores.reshape(-1)
        prune_count = int(self.sparsity * flat_scores.numel())
        if prune_count:
            order = torch.argsort(flat_scores, stable=True)
            mask.reshape(-1)[order[:prune_count]] = 0
        return mask

    def _select_nm_mask(self, scores):
        mask = torch.ones_like(scores)
        order = torch.argsort(scores, dim=1, stable=True)
        mask.scatter_(1, order[:, :self.n], 0)
        return mask

    def prune(self, calibration_inputs):
        if not isinstance(calibration_inputs, torch.Tensor) or calibration_inputs.ndim < 1:
            raise ValueError("calibration_inputs must be a tensor with a final feature dimension")
        if calibration_inputs.shape[-1] != self.in_features:
            raise ValueError("the final calibration dimension must match in_features")
        if calibration_inputs.numel() == 0 or not torch.isfinite(calibration_inputs).all():
            raise ValueError("calibration_inputs must be nonempty and finite")
        if not torch.isfinite(self.weight).all():
            raise ValueError("weight must contain only finite values")
        compute_dtype = torch.float64 if self.weight.dtype == torch.float64 else torch.float32
        inputs = calibration_inputs.detach().reshape(-1, self.in_features).to(
            device=self.weight.device, dtype=compute_dtype
        )
        weights = self.weight.detach().to(dtype=compute_dtype).clone()
        hessian = inputs.mT.matmul(inputs)
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
        mask = torch.ones_like(weights)
        features = self.in_features
        with torch.no_grad():
            for start in range(0, features, self.block_size):
                stop = min(start + self.block_size, features)
                errors = torch.zeros((self.out_features, stop - start), device=weights.device,
                                     dtype=weights.dtype)
                for column in range(start, stop):
                    if self.n is None:
                        if column % self.mask_block_size == 0:
                            mask_stop = min(column + self.mask_block_size, features)
                            denominator = upper.diagonal()[column:mask_stop].square()
                            scores = weights[:, column:mask_stop].square() / denominator
                            mask[:, column:mask_stop] = self._select_mask(scores)
                    elif column % self.m == 0:
                        mask_stop = column + self.m
                        denominator = upper.diagonal()[column:mask_stop].square()
                        scores = weights[:, column:mask_stop].square() / denominator
                        mask[:, column:mask_stop] = self._select_nm_mask(scores)
                    diagonal = upper[column, column]
                    error = (1.0 - mask[:, column]) * weights[:, column] / diagonal
                    errors[:, column - start] = error
                    weights[:, column:stop] -= error.unsqueeze(1) * upper[column, column:stop]
                if stop < features:
                    weights[:, stop:] -= errors.matmul(upper[start:stop, stop:])
            weights.mul_(mask)
            self.weight.copy_(weights.to(dtype=self.weight.dtype))
            self.mask.copy_(mask.to(dtype=self.mask.dtype))
        return self.mask

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor) or inputs.ndim < 1:
            raise ValueError("inputs must be a tensor with a final feature dimension")
        if inputs.shape[-1] != self.in_features:
            raise ValueError("the final input dimension must match in_features")
        return F.linear(inputs, self.weight * self.mask, self.bias)
