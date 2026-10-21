"""Auxiliary losses reported by submodules during a forward.

A mixer or expert layer may set ``module.aux_loss`` to a scalar tensor or a
dict of named scalar tensors. :func:`collect_aux_losses` sums those terms and
clears the attributes so the next forward starts clean. Training code adds
the result to the optimized loss; validation should collect and discard it
so reported cross-entropy stays pure.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


def normalize_aux_loss(raw: Any) -> dict[str, torch.Tensor]:
    """Turn a tensor or a name→tensor mapping into a dict. Other values are ignored.

    A bare tensor is recorded as ``"loss"``.
    """
    if raw is None:
        return {}
    if isinstance(raw, torch.Tensor):
        return {"loss": raw}
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items() if isinstance(value, torch.Tensor)}
    return {}


def collect_aux_losses(module: nn.Module | None) -> dict[str, torch.Tensor]:
    """Sum ``aux_loss`` attributes written by ``module`` and its submodules, then clear them.

    The same key from several modules is summed. Clearing is a plain attribute
    write to ``None``; a submodule that cannot be cleared keeps its value.
    """
    totals: dict[str, torch.Tensor] = {}
    if module is None:
        return totals
    for mod in module.modules():
        raw = getattr(mod, "aux_loss", None)
        if raw is None:
            continue
        for key, term in normalize_aux_loss(raw).items():
            totals[key] = term if key not in totals else totals[key] + term
        attrs = getattr(mod, "__dict__", None)
        if isinstance(attrs, dict):
            attrs["aux_loss"] = None
    return totals
