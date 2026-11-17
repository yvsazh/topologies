import math

import torch
from torch import nn


class ShampooStep(nn.Module):
    component_id = "shampoo"

    def __init__(self, eps):
        super().__init__()
        if type(eps) not in (int, float):
            raise ValueError("eps must be a finite positive real number")
        try:
            eps = float(eps)
        except OverflowError as error:
            raise ValueError("eps must be a finite positive real number") from error
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a finite positive real number")
        self.eps = eps
        self.register_buffer("left_accumulator", None)
        self.register_buffer("right_accumulator", None)

    def _inverse_fourth_root(self, matrix):
        symmetric = (matrix + matrix.mT) * 0.5
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
        eigenvalues = eigenvalues.clamp_min(self.eps)
        scaled_vectors = eigenvectors * eigenvalues.pow(-0.25).unsqueeze(0)
        return scaled_vectors @ eigenvectors.mT

    def forward(self, parameter, gradient, lr):
        if not isinstance(parameter, torch.Tensor) or parameter.ndim != 2:
            raise ValueError("parameter must be a two-dimensional tensor")
        if parameter.shape[0] == 0 or parameter.shape[1] == 0:
            raise ValueError("parameter dimensions must be non-empty")
        if not parameter.is_floating_point():
            raise ValueError("parameter must have a floating-point dtype")
        if not isinstance(gradient, torch.Tensor) or gradient.ndim != 2:
            raise ValueError("gradient must be a two-dimensional tensor")
        if gradient.shape != parameter.shape:
            raise ValueError("gradient and parameter must have the same shape")
        if not gradient.is_floating_point():
            raise ValueError("gradient must have a floating-point dtype")
        if gradient.device != parameter.device or gradient.dtype != parameter.dtype:
            raise ValueError("gradient and parameter must share device and dtype")
        if type(lr) not in (int, float):
            raise ValueError("learning rate must be a finite non-negative real number")
        try:
            learning_rate = float(lr)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("learning rate must be a finite non-negative real number") from error
        if not math.isfinite(learning_rate) or learning_rate < 0.0:
            raise ValueError("learning rate must be a finite non-negative real number")
        if not torch.isfinite(parameter).all() or not torch.isfinite(gradient).all():
            raise ValueError("parameter and gradient must contain only finite values")

        rows, columns = parameter.shape
        partial_batch = False
        if self.left_accumulator is not None:
            if self.left_accumulator.device != parameter.device:
                raise ValueError("parameter device differs from initialized state; move the module with .to()")
            state_rows = self.left_accumulator.shape[0]
            state_columns = self.right_accumulator.shape[0]
            if (state_rows, state_columns) != (rows, columns):
                partial_batch = rows == 1 and columns == state_columns and state_rows > 1
                if not partial_batch:
                    raise ValueError("parameter shape cannot change after state initialization")

        work_dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
        detached_gradient = gradient.detach().to(dtype=work_dtype)
        with torch.no_grad():
            if self.left_accumulator is None:
                left_identity = torch.eye(rows, device=parameter.device, dtype=work_dtype)
                right_identity = torch.eye(columns, device=parameter.device, dtype=work_dtype)
                left_accumulator = self.eps * left_identity
                right_accumulator = self.eps * right_identity
            else:
                left_accumulator = self.left_accumulator.to(dtype=work_dtype)
                right_accumulator = self.right_accumulator.to(dtype=work_dtype)
            if partial_batch:
                padded_gradient = detached_gradient.new_zeros((left_accumulator.shape[0], columns))
                padded_gradient[:rows] = detached_gradient
                left_accumulator = left_accumulator + padded_gradient @ padded_gradient.mT
                left_for_step = left_accumulator[:rows, :rows]
            else:
                left_accumulator = left_accumulator + detached_gradient @ detached_gradient.mT
                left_for_step = left_accumulator
            right_accumulator = right_accumulator + detached_gradient.mT @ detached_gradient
            if self.training:
                self.left_accumulator = left_accumulator
                self.right_accumulator = right_accumulator

        left_preconditioner = self._inverse_fourth_root(left_for_step)
        right_preconditioner = self._inverse_fourth_root(right_accumulator)
        direction = left_preconditioner @ detached_gradient @ right_preconditioner
        return parameter - learning_rate * direction.to(dtype=parameter.dtype)
