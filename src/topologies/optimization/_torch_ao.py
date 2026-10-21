"""Single access point for PyTorch's eager ``torch.ao.quantization`` API.

PyTorch has deprecated eager-mode quantization and will remove it. Routing every
use through this shim turns a missing API into an actionable
``OptimizationError`` instead of an ``AttributeError`` deep inside a workflow.
"""

from __future__ import annotations

import warnings
from typing import Any

from .contracts import OptimizationError

_MIGRATION_HINT = (
    "This PyTorch build no longer provides eager torch.ao.quantization. Use the "
    "TorchAO-based backends instead (pip install 'topologies[optimization]'; "
    "quantization presets 'int8-static' / 'int8-qat' run through "
    "topologies.optimization.torchao_pt2e)."
)


def eager_api() -> Any:
    """``torch.ao.quantization`` or an ``OptimizationError`` with a migration hint."""
    import torch

    api = getattr(getattr(torch, "ao", None), "quantization", None)
    if api is None or not callable(getattr(api, "quantize_dynamic", None)):
        raise OptimizationError(
            _MIGRATION_HINT,
            code="missing_backend",
            backend="pytorch-dynamic-legacy",
            stage="prepare",
        )
    return api


def quantize_dynamic(model: Any, **kwargs: Any) -> Any:
    """``torch.ao.quantization.quantize_dynamic`` through the shim.

    PyTorch's own deprecation warning is silenced here: the legacy backend is
    an explicit opt-in, and its removal surfaces as an ``OptimizationError``
    with a migration hint (see ``eager_api``).
    """
    api = eager_api()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"torch\.ao\.quantization is deprecated", category=DeprecationWarning
        )
        return api.quantize_dynamic(model, **kwargs)
