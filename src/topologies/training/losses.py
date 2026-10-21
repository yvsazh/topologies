"""Loss registry and the opt-in recipe loss override.

Recipes ``causal-lm``, ``masked-lm`` and ``vision-classify`` accept
``recipe_config={"loss": <name or callable>, "loss_kwargs": {...}}``.
``loss=None`` (default) keeps each recipe's exact built-in computation.

Call contracts:

- token recipes (``causal-lm``, ``masked-lm``):
  ``fn(logits_2d [N, V], targets_1d [N], *, ignore_index=None, **loss_kwargs)``;
  positions equal to ``ignore_index`` (padding, unmasked MLM tokens) must
  not contribute. Validation still reports cross-entropy as ``loss`` (so
  ``ppl`` and bits-per-byte stay meaningful) plus ``objective``.
- ``vision-classify``: ``fn(logits [B, C], labels [B], **loss_kwargs)``;
  mixup/CutMix mix the loss over both label sets. Validation reports the
  override as ``loss`` (like label smoothing today).

    from topologies.training import register_loss
    register_loss("poly1", lambda logits, y, *, ignore_index=None, eps=1.0: ...)
    Trainer(model, train, recipe="vision-classify",
            recipe_config={"loss": "focal", "loss_kwargs": {"gamma": 2.0}})
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import Field, field_serializer, field_validator

from topologies.training.recipe_base import RecipeConfig

_LOSSES: dict[str, Callable[..., torch.Tensor]] = {}


def register_loss(name: str, fn: Callable[..., torch.Tensor], *, overwrite: bool = True) -> None:
    """Make ``recipe_config={"loss": name}`` use ``fn``."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("loss name must be non-empty.")
    if not callable(fn):
        raise TypeError(f"loss {name!r} must be callable.")
    if key in _LOSSES and not overwrite:
        raise ValueError(f"loss {name!r} is already registered.")
    _LOSSES[key] = fn


def list_losses() -> list[str]:
    return sorted(_LOSSES)


def get_loss(spec: str | Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Registered name or callable -> callable."""
    if callable(spec) and not isinstance(spec, str):
        return spec
    key = str(spec).strip().lower()
    if key not in _LOSSES:
        sug = difflib.get_close_matches(key, list_losses(), n=1)
        hint = f" Did you mean {sug[0]!r}?" if sug else ""
        raise ValueError(f"Unknown loss {spec!r}.{hint} Available: {', '.join(list_losses())}.")
    return _LOSSES[key]


def recipes_with_loss_override() -> list[str]:
    """Recipe names whose config schema has a ``loss`` field."""
    from topologies.training.recipe_base import _RECIPE_REGISTRY

    return sorted(n for n, cls in _RECIPE_REGISTRY.items()
                  if "loss" in getattr(cls, "ConfigSchema", RecipeConfig).model_fields)


# ---------------------------------------------------------------- built-ins


def _keep(targets: torch.Tensor, ignore_index: int | None) -> torch.Tensor | None:
    if ignore_index is None or targets.is_floating_point():
        return None
    keep = targets != ignore_index
    return None if bool(keep.all()) else keep


def _dense_targets(pred: torch.Tensor, targets: torch.Tensor,
                   ignore_index: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Regression-style losses on class targets: drop ignored rows, one-hot encode."""
    keep = _keep(targets, ignore_index)
    if keep is not None:
        pred, targets = pred[keep], targets[keep]
    if not targets.is_floating_point() and pred.dim() == targets.dim() + 1:
        targets = F.one_hot(targets.long(), pred.shape[-1])
    return pred, targets.to(pred.dtype)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, *,
                  ignore_index: int | None = None, label_smoothing: float = 0.0,
                  weight: Any = None) -> torch.Tensor:
    w = torch.as_tensor(weight, dtype=logits.dtype, device=logits.device) if weight is not None else None
    return get_loss("cross-entropy")(logits, targets.long(), weight=w,
                                    ignore_index=-100 if ignore_index is None else ignore_index,
                                    label_smoothing=label_smoothing)


def focal(logits: torch.Tensor, targets: torch.Tensor, *, ignore_index: int | None = None,
          gamma: float = 2.0, alpha: float | None = None) -> torch.Tensor:
    """Focal loss ``-(1 - p_t)^gamma log p_t`` (Lin et al., 2017); ``alpha`` scales it."""
    keep = _keep(targets, ignore_index)
    if keep is not None:
        logits, targets = logits[keep], targets[keep]
    if targets.numel() == 0:
        return logits.sum() * 0.0
    return get_loss("softmax-focal")(logits, targets.long(), gamma=gamma, alpha=alpha,
                                      ignore_index=-100 if keep is not None else
                                      (-100 if ignore_index is None else ignore_index))


def mse(pred: torch.Tensor, targets: torch.Tensor, *,
        ignore_index: int | None = None) -> torch.Tensor:
    pred, targets = _dense_targets(pred, targets, ignore_index)
    return get_loss("mean-squared-error")(pred, targets)


def l1(pred: torch.Tensor, targets: torch.Tensor, *,
       ignore_index: int | None = None) -> torch.Tensor:
    pred, targets = _dense_targets(pred, targets, ignore_index)
    return get_loss("mean-absolute-error")(pred, targets)


def huber(pred: torch.Tensor, targets: torch.Tensor, *, ignore_index: int | None = None,
          delta: float = 1.0) -> torch.Tensor:
    pred, targets = _dense_targets(pred, targets, ignore_index)
    return get_loss("huber")(pred, targets, delta=float(delta))


def bce_logits(logits: torch.Tensor, targets: torch.Tensor, *, ignore_index: int | None = None,
               pos_weight: Any = None) -> torch.Tensor:
    logits, targets = _dense_targets(logits, targets, ignore_index)
    pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device) if pos_weight is not None else None
    return get_loss("binary-cross-entropy-logits")(logits, targets, pos_weight=pw)


def adversarial_hinge_generator(fake_scores: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    return get_loss("adversarial-hinge-generator")(fake_scores, reduction=reduction)


# Canonical public IDs are kebab-case. Historical spellings are aliases to
# these exact callable objects (not separately implemented objectives).
CANONICAL_LOSS_ALIASES = {
    "cross-entropy": (None, ("cross_entropy",)),
    "softmax-focal": (None, ("focal",)),
    "mean-squared-error": (None, ("mse",)),
    "mean-absolute-error": (None, ("l1",)),
    "huber": (None, ()),
    "binary-cross-entropy-logits": (None, ("bce_logits",)),
    "bpr": (None, ()),
    "adversarial-hinge": (None, ()),
}
for _alias, _fn in (("cross_entropy", cross_entropy), ("focal", focal), ("mse", mse),
                     ("l1", l1), ("huber", huber), ("bce_logits", bce_logits)):
    register_loss(_alias, _fn)
register_loss("adversarial-hinge-generator", lambda scores, *, reduction="mean":
              (-scores if reduction == "none" else -scores.sum() if reduction == "sum" else -scores.mean()))


# ---------------------------------------------------------------- recipe config mixin


def _callable_name(fn: Any) -> str:
    mod, qual = getattr(fn, "__module__", None), getattr(fn, "__qualname__", None)
    if mod and qual and "<" not in qual:
        return f"{mod}:{qual}"
    return f"<callable {getattr(fn, '__name__', type(fn).__name__)}>"


class LossOverrideConfig(RecipeConfig):
    """Recipe config fields for the opt-in loss override (see module docstring)."""

    loss: str | Callable[..., Any] | None = Field(
        default=None, description="Registered loss name or callable; None = built-in loss.")
    loss_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("loss")
    @classmethod
    def _loss_known(cls, v: Any) -> Any:
        if isinstance(v, str):
            get_loss(v)  # raises ValueError with a suggestion
            return v.strip().lower()
        return v

    @field_serializer("loss")
    def _loss_json(self, v: Any) -> Any:
        return v if v is None or isinstance(v, str) else _callable_name(v)


def compute_loss(cfg: LossOverrideConfig, logits: torch.Tensor, targets: torch.Tensor,
                 **extra: Any) -> torch.Tensor:
    """Call the configured override with ``loss_kwargs`` (and recipe ``extra`` kwargs)."""
    fn = get_loss(cfg.loss)  # type: ignore[arg-type]
    return fn(logits, targets, **extra, **dict(cfg.loss_kwargs or {}))


__all__ = [
    "CANONICAL_LOSS_ALIASES",
    "LossOverrideConfig",
    "adversarial_hinge_generator",
    "bce_logits",
    "compute_loss",
    "cross_entropy",
    "focal",
    "get_loss",
    "huber",
    "l1",
    "list_losses",
    "mse",
    "recipes_with_loss_override",
    "register_loss",
]
