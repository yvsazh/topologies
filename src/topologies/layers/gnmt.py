import math
import numbers

import torch
from torch import nn


class LengthNormalizedScore(nn.Module):
    component_id = "gnmt"

    def __init__(self):
        super().__init__()

    def _raise_value_error(self, message):
        raise ValueError(message)

    def forward(self, log_probability, length, alpha):
        if not isinstance(log_probability, torch.Tensor) or not log_probability.is_floating_point():
            raise ValueError("log_probability must be a floating-point tensor")

        if isinstance(length, torch.Tensor):
            length_values = length
            if length_values.dtype == torch.bool or length_values.is_complex():
                raise ValueError("length must contain integer token counts")
            if length_values.is_floating_point():
                if not torch.isfinite(length_values).all().item():
                    raise ValueError("length must contain finite token counts")
                if torch.ne(length_values, torch.floor(length_values)).any().item():
                    raise ValueError("length must contain integer token counts")
        elif isinstance(length, numbers.Real) and not isinstance(length, bool):
            if not math.isfinite(float(length)):
                raise ValueError("length must contain finite token counts")
            if int(length) != length:
                raise ValueError("length must contain integer token counts")
            length_values = torch.as_tensor(length, device=log_probability.device)
        else:
            self._raise_value_error("length must be a nonnegative integer or integer-valued tensor")

        if torch.lt(length_values, 0).any().item():
            raise ValueError("length must be nonnegative")

        try:
            torch.broadcast_shapes(log_probability.shape, length_values.shape)
        except RuntimeError as error:
            raise ValueError("length must be broadcastable to log_probability") from error

        if isinstance(alpha, torch.Tensor):
            if alpha.dtype == torch.bool or alpha.is_complex():
                raise ValueError("alpha must be a finite real tensor")
            if not torch.isfinite(alpha).all().item():
                raise ValueError("alpha must be finite")
        elif isinstance(alpha, numbers.Real) and not isinstance(alpha, bool):
            if not math.isfinite(float(alpha)):
                raise ValueError("alpha must be finite")
        else:
            self._raise_value_error("alpha must be a finite scalar")

        try:
            torch.broadcast_shapes(log_probability.shape, length_values.shape, alpha.shape if isinstance(alpha, torch.Tensor) else ())
        except RuntimeError as error:
            raise ValueError("alpha must be broadcastable to log_probability and length") from error

        length_values = length_values.to(device=log_probability.device, dtype=log_probability.dtype)
        alpha_value = torch.as_tensor(
            alpha, device=log_probability.device, dtype=log_probability.dtype
        )
        length_penalty = torch.pow((5.0 + length_values) / 6.0, alpha_value)
        return log_probability / length_penalty
