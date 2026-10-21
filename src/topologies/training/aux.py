"""Add auxiliary losses to a training objective without touching validation metrics."""

from __future__ import annotations

from typing import Any

import torch

from topologies.nn.aux import collect_aux_losses, normalize_aux_loss


def _as_scalar(term: torch.Tensor) -> torch.Tensor:
    if term.ndim == 0 or term.numel() == 1:
        return term.reshape(())
    return term.sum()


def apply_aux_losses(
    loss: torch.Tensor, module: Any, returned: Any = None
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return ``loss + aux`` and a log dict ``aux/<name>``.

    ``returned`` is the model's ``aux_loss`` output (a tensor or a dict).
    Submodule attributes are collected and cleared. A tensor object that
    appears in both places is counted once.
    """
    terms: dict[str, torch.Tensor] = {}
    seen: set[int] = set()

    def _take(raw: Any) -> None:
        for key, term in normalize_aux_loss(raw).items():
            if id(term) in seen:
                continue
            seen.add(id(term))
            scalar = _as_scalar(term)
            terms[key] = scalar if key not in terms else terms[key] + scalar

    _take(returned)
    _take(collect_aux_losses(module))
    total = loss
    metrics: dict[str, float] = {}
    for key, term in terms.items():
        total = total + term
        metrics[f"aux/{key}"] = float(term.detach())
    return total, metrics


def fold_leftover_aux(
    loss: torch.Tensor, metrics: dict[str, Any], module: Any
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Add auxiliary terms a recipe did not already consume.

    Recipes that call :func:`apply_aux_losses` leave nothing to fold, so this
    is a no-op for them. Validation must not call it.
    """
    _total, extra = apply_aux_losses(loss, module, None)
    if not extra:
        return loss, metrics
    out = dict(metrics)
    out.update(extra)
    out["loss"] = float(_total.detach())
    return _total, out
