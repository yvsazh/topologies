"""Global-norm gradient clipping utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence


def grad_clip(grads: Sequence[object], max_norm: float = 1.0) -> float:
    """Clip gradients in place by their combined L2 norm.

    Returns the multiplier applied to each gradient (1.0 when no clipping
    was needed). Gradients may be PyTorch-like tensors supporting ``norm``
    and ``mul_``, or nested Python sequences of numeric values.
    """
    if max_norm < 0:
        raise ValueError("max_norm must be non-negative")

    norms = []
    for grad in grads:
        norm_method = getattr(grad, "norm", None)
        if callable(norm_method):
            value = norm_method(2).item()
        else:
            value = _sequence_squared_norm(grad) ** 0.5
        norms.append(float(value))

    global_norm = math.sqrt(math.fsum(norm * norm for norm in norms))
    if global_norm == 0 or global_norm <= max_norm:
        return 1.0

    scale = max_norm / global_norm
    for grad in grads:
        multiply_in_place = getattr(grad, "mul_", None)
        if callable(multiply_in_place):
            multiply_in_place(scale)
        else:
            _scale_sequence_in_place(grad, scale)
    return scale


def _sequence_squared_norm(values: object) -> float:
    try:
        return math.fsum(
            _sequence_squared_norm(value) if isinstance(value, (list, tuple)) else float(value) ** 2
            for value in values  # type: ignore[union-attr]
        )
    except TypeError as error:
        raise TypeError("gradients must be tensors or nested mutable sequences") from error


def _scale_sequence_in_place(values: object, scale: float) -> None:
    for index, value in enumerate(values):  # type: ignore[union-attr]
        if isinstance(value, (list, tuple)):
            _scale_sequence_in_place(value, scale)
        else:
            values[index] = value * scale  # type: ignore[index]
