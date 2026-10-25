import math

import torch
from torch import nn


def _is_valid_beta2(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and \
        0 <= value < 1 and math.isfinite(value)


def _is_valid_epsilon(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_tensor(value):
    return isinstance(value, torch.Tensor)


def _is_bool(value):
    return isinstance(value, bool)


class AdafactorFactoredSecondMomentStep(nn.Module):
    component_id = "adafactor"

    def __init__(self, beta2=None, epsilon=None):
        super().__init__()
        if not _is_valid_beta2(beta2):
            raise ValueError("beta2 must be a finite number in [0, 1)")
        if not _is_valid_epsilon(epsilon):
            raise ValueError("epsilon must be a finite non-negative number")
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)

    def forward(self, gradient, row_state, column_state, step, return_state=True):
        if not _is_bool(return_state):
            raise ValueError("return_state must be a boolean")
        if not isinstance(gradient, torch.Tensor) or gradient.ndim not in (2, 3):
            raise ValueError("gradient must be a two-dimensional matrix or a three-dimensional batch")
        if not gradient.is_floating_point():
            raise ValueError("gradient must have a real floating-point dtype")
        if not _is_tensor(row_state):
            raise ValueError("row_state must be a tensor")
        if not _is_tensor(column_state):
            raise ValueError("column_state must be a tensor")
        batched_rows = gradient.ndim == 2 and row_state.ndim == 2 and \
            column_state.ndim == 2 and row_state.shape[0] == gradient.shape[0] and \
            column_state.shape == gradient.shape
        if gradient.ndim == 2 and not batched_rows:
            if row_state.ndim != 1:
                raise ValueError("row_state must be a one-dimensional tensor")
            if row_state.shape != (gradient.shape[0],):
                raise ValueError("row_state shape must match the gradient row count")
            if column_state.ndim != 1:
                raise ValueError("column_state must be a one-dimensional tensor")
            if column_state.shape != (gradient.shape[1],):
                raise ValueError("column_state shape must match the gradient column count")
        elif gradient.ndim == 3:
            expected_row_shape = (gradient.shape[0], gradient.shape[1])
            expected_column_shape = (gradient.shape[0], gradient.shape[2])
            if row_state.shape != expected_row_shape:
                raise ValueError("row_state shape must match the gradient batch and row counts")
            if column_state.shape != expected_column_shape:
                raise ValueError("column_state shape must match the gradient batch and column counts")
        elif not batched_rows:
            raise ValueError("two-dimensional gradients require one-dimensional states")
        if not row_state.is_floating_point() or not column_state.is_floating_point():
            raise ValueError("row_state and column_state must have real floating-point dtypes")
        if row_state.device != gradient.device or column_state.device != gradient.device:
            raise ValueError("gradient and states must be on the same device")
        if row_state.dtype != gradient.dtype or column_state.dtype != gradient.dtype:
            raise ValueError("gradient and states must have the same dtype")
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError("step must be a positive one-indexed integer")

        unbatched = gradient.ndim == 2 and not batched_rows
        if unbatched:
            gradient = gradient.unsqueeze(0)
            row_state = row_state.unsqueeze(0)
            column_state = column_state.unsqueeze(0)
        elif batched_rows:
            gradient = gradient.unsqueeze(1).expand(-1, row_state.shape[1], -1)

        squared_gradient = gradient.square()
        updated_row = self.beta2 * row_state + (1.0 - self.beta2) * squared_gradient.sum(dim=2)
        updated_column = self.beta2 * column_state + (1.0 - self.beta2) * squared_gradient.sum(dim=1)
        row_total = updated_row.sum(dim=1)
        safe_row_total = torch.where(row_total != 0, row_total, torch.ones_like(row_total))
        factored_second_moment = updated_row.unsqueeze(2) * updated_column.unsqueeze(1)
        factored_second_moment = factored_second_moment / safe_row_total[:, None, None]
        factored_second_moment = factored_second_moment / (1.0 - self.beta2**step)
        denominator = factored_second_moment.sqrt() + self.epsilon
        safe_denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
        preconditioned = gradient / safe_denominator
        if unbatched:
            preconditioned = preconditioned.squeeze(0)
            updated_row = updated_row.squeeze(0)
            updated_column = updated_column.squeeze(0)
        if return_state:
            return preconditioned, updated_row, updated_column
        return preconditioned
