"""Recipe base classes, contexts, and registry.

Levels:
- loss: batch_to_loss()            (Trainer owns the loop)
- step: training_step(ctx)         (Trainer owns epoch loop)
- epoch: epoch_loop(ctx)           (Trainer owns fit lifecycle)
- fit: fit_loop(ctx)               (last-resort escape hatch)
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict

from topologies.training.errors import TrainingError

RecipeLevel = Literal["loss", "step", "epoch", "fit"]


class RecipeConfig(BaseModel):
    """Strict per-recipe hyperparameters. Unknown keys are forbidden."""

    model_config = ConfigDict(extra="forbid")


class RecipeComponents(BaseModel):
    """Typed executable dependencies for a recipe."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


# ---------------------------------------------------------------- contexts


@dataclass
class StepContext:
    model: nn.Module
    batch: dict | None
    batch_idx: int = 0
    epoch: int = 0
    global_step: int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    precision: str = "fp32"
    scaler: Any = None
    accumulation_steps: int = 1
    is_accumulating: bool = False
    rank: int = 0
    world_size: int = 1
    config: Any = None
    components: Any = None
    adapter: Any = None
    logger: Any = None
    metrics: dict = field(default_factory=dict)
    _backward_fn: Any = None
    _zero_grad_fn: Any = None
    _step_optimizer_fn: Any = None
    _step_scheduler_fn: Any = None
    _clip_fn: Any = None
    _log_fn: Any = None

    def backward(self, loss: torch.Tensor, optimizer_name: str = "default") -> None:
        if self._backward_fn is None:
            raise TrainingError("backward() unavailable in this context.", stage="recipe-loop")
        self._backward_fn(loss, optimizer_name)

    def zero_grad(self, name: str | None = None) -> None:
        if self._zero_grad_fn is None:
            raise TrainingError("zero_grad() unavailable in this context.", stage="recipe-loop")
        self._zero_grad_fn(name)

    def step_optimizer(self, name: str = "default") -> None:
        if self._step_optimizer_fn is None:
            raise TrainingError(
                "step_optimizer() unavailable in this context.", stage="recipe-loop"
            )
        self._step_optimizer_fn(name)

    def step_scheduler(self, name: str = "default") -> None:
        if self._step_scheduler_fn is None:
            return
        self._step_scheduler_fn(name)

    def clip_gradients(self, name: str = "default") -> None:
        if self._clip_fn is None:
            return
        self._clip_fn(name)

    def log(self, values: dict[str, float]) -> None:
        if self._log_fn is not None:
            self._log_fn(values)
        else:
            self.metrics.update(values)


@dataclass
class StepResult:
    loss: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationStepResult:
    """Legacy scalar metrics and the already-computed predictions/targets."""
    metrics: dict[str, float]
    record: Any


@dataclass
class EpochContext:
    model: nn.Module
    train_loader: Any = None
    val_loader: Any = None
    epoch: int = 0
    global_step: int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    precision: str = "fp32"
    rank: int = 0
    world_size: int = 1
    config: Any = None
    recipe_config: Any = None
    components: Any = None
    adapter: Any = None
    logger: Any = None
    trainer: Any = None  # Trainer services (device, logging, checkpoint, ...)


@dataclass
class EpochResult:
    loss: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    global_step: int = 0


@dataclass
class FitContext:
    model: nn.Module
    train_loader: Any = None
    val_loader: Any = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    precision: str = "fp32"
    rank: int = 0
    world_size: int = 1
    config: Any = None
    recipe_config: Any = None
    components: Any = None
    adapter: Any = None
    logger: Any = None
    trainer: Any = None


@dataclass
class FitResult:
    global_step: int = 0
    epochs: int = 0
    history: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------- recipe


def _rng_scoped(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run ``fn`` with the caller's global RNG state saved and restored.

    ``dummy_batch`` implementations seed for reproducibility; without this the
    seed would silently reset the user's global RNG.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from topologies._execution import execution_rng

        with execution_rng(preserve_rng=True):
            return fn(*args, **kwargs)

    wrapper._topologies_rng_scoped = True  # type: ignore[attr-defined]
    return wrapper


class Recipe:
    """Algorithm/task-specific training behavior. Subclass per algorithm."""

    name: str = "base"
    level: RecipeLevel = "loss"
    ConfigSchema: type[RecipeConfig] = RecipeConfig
    ComponentsSchema: type[RecipeComponents] | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        fn = cls.__dict__.get("dummy_batch")
        if callable(fn) and not getattr(fn, "_topologies_rng_scoped", False):
            cls.dummy_batch = _rng_scoped(fn)  # type: ignore[method-assign]

    # -- defaults ---------------------------------------------------------
    def default_config(self) -> dict:
        try:
            return self.ConfigSchema().model_dump()
        except Exception:  # noqa: BLE001
            return {}

    def default_training_policy(self) -> dict:
        """Recipe-specific optimizer/training defaults (lr, scheduler, ...)."""
        return {"optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.0, "scheduler": None}

    # -- batch contract ---------------------------------------------------
    def expected_batch_spec(self) -> dict[str, str]:
        return {}

    def optional_batch_keys(self) -> set[str]:
        """Spec keys a batch may omit (e.g. causal-lm ``labels``/``mask``)."""
        return set()

    def target_keys(self) -> set[str]:
        """Keys holding supervision targets (excluded from prediction inputs)."""
        return {"labels", "label", "targets", "target"}

    def required_batch_keys(self) -> list[str]:
        optional = self.optional_batch_keys()
        return [k for k in self.expected_batch_spec() if k not in optional]

    def input_keys(self) -> list[str]:
        """Required keys that are model inputs (first one = primary input)."""
        targets = self.target_keys()
        return [k for k in self.required_batch_keys() if k not in targets]

    def validate_batch(self, batch: dict, *, strict: bool = True) -> None:
        spec = self.expected_batch_spec()
        if not spec or not strict:
            return
        # Only enforce presence of required keys; shapes checked at loss time.
        required = self.required_batch_keys()
        missing = [k for k in required if k not in batch]
        if missing:
            actual = {k: _tensor_summary(v) for k, v in batch.items()}
            exp = "\n".join(f"    {k}: {v}" for k, v in spec.items())
            rec = "\n".join(f"    {k}: {v}" for k, v in actual.items())
            raise TrainingError(
                "Training failed during batch validation.",
                recipe=self.name,
                stage="batch",
                expected=exp,
                received=rec or "(empty batch)",
                hint=(
                    f"Missing keys {missing}. Rename batch keys or map them with "
                    f"input_map={{'{_extra_key(batch, spec)}': '{missing[0]}'}} "
                    "(batch key -> expected key)."
                ),
            )

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        raise NotImplementedError(f"Recipe {self.name!r} has no dummy_batch().")

    def optimizer_params(self, optimizer_name: str, model: Any, components: Any):
        """Return parameters owned by a named optimizer (multi-optimizer support)."""
        return model.parameters()

    # -- checks / evaluation / inference hooks ------------------------------
    def preflight(
        self,
        model: nn.Module,
        batch: dict,
        cfg: RecipeConfig,
        components: RecipeComponents | None,
        adapter: Any = None,
    ) -> None:
        """One-time sanity checks on the first training batch.

        Raise ``TrainingError`` with a fix (vocab/label/shape mismatches)
        instead of letting a raw tensor error surface mid-training.
        """

    def validation_step(
        self,
        model: nn.Module,
        batch: dict,
        cfg: RecipeConfig,
        components: RecipeComponents | None,
        adapter: Any = None,
    ) -> dict[str, float]:
        """Metrics for one evaluation batch (called under ``torch.no_grad``).

        Loss-level recipes reuse ``batch_to_loss``. A reserved ``_weight``
        entry sets the batch weight for averaging (default: batch size).
        Step/epoch/fit recipes override this to support validation.
        """
        if self.level != "loss":
            raise NotImplementedError(
                f"Recipe {self.name!r} (level={self.level}) defines no validation_step()."
            )
        loss, metrics = self.batch_to_loss(model, batch, cfg, components, adapter)
        out = {k: float(v) for k, v in metrics.items()}
        out.setdefault("loss", float(loss.detach()))
        return out

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        """Optional combined hook returning ``(scalar_metrics, EvaluationRecord)``.

        Legacy recipes deliberately have no implicit prediction forward: override this
        hook when extra prediction metrics are needed.
        """
        raise NotImplementedError

    def summarize_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Derive aggregate metrics after averaging (e.g. perplexity from loss)."""
        return metrics

    def predict_step(
        self,
        model: nn.Module,
        batch: dict,
        cfg: RecipeConfig,
        components: RecipeComponents | None,
        adapter: Any = None,
    ) -> Any:
        """Model outputs for one batch of inputs (targets are not required)."""
        from topologies.training.adapters import call_with_adapter

        targets = self.target_keys()
        inputs = {k: v for k, v in batch.items() if k not in targets}
        keys = self.input_keys()
        primary = next((k for k in keys if k in inputs), None)
        return call_with_adapter(adapter, model, inputs, None, primary=primary)

    # -- level 1 ----------------------------------------------------------
    def batch_to_loss(
        self,
        model: nn.Module,
        batch: dict,
        cfg: RecipeConfig,
        components: RecipeComponents | None,
        adapter: Any = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError(f"Recipe {self.name!r} does not implement batch_to_loss().")

    # -- level 2 ----------------------------------------------------------
    def training_step(self, ctx: StepContext) -> StepResult:
        raise NotImplementedError(f"Recipe {self.name!r} does not implement training_step().")

    # -- level 3 ----------------------------------------------------------
    def epoch_loop(self, ctx: EpochContext) -> EpochResult:
        raise NotImplementedError(f"Recipe {self.name!r} does not implement epoch_loop().")

    def fit_loop(self, ctx: FitContext) -> FitResult:
        raise NotImplementedError(f"Recipe {self.name!r} does not implement fit_loop().")


def _extra_key(batch: dict, spec: dict) -> str:
    extra = [k for k in batch if k not in spec]
    return extra[0] if len(extra) == 1 else "<your-key>"


def _tensor_summary(v: Any) -> str:
    try:
        import torch as _t

        if isinstance(v, _t.Tensor):
            return f"{v.dtype}[{tuple(v.shape)}]"
        return type(v).__name__
    except Exception:  # noqa: BLE001
        return type(v).__name__


# ---------------------------------------------------------------- registry

_RECIPE_REGISTRY: dict[str, Callable[[], Recipe]] = {}
_RECIPE_HELP: dict[str, str] = {}


def register_recipe(name: str, *, help: str = "",
                    replace: bool = False) -> Callable[[type[Recipe]], type[Recipe]]:
    """Register a recipe class under ``name``.

    A different class under an existing name raises unless ``replace=True``
    (re-registering the same class, e.g. after a module reload, is allowed).
    """
    def deco(cls: type[Recipe]) -> type[Recipe]:
        existing = _RECIPE_REGISTRY.get(name)
        same = existing is not None and (
            existing is cls or (existing.__module__, existing.__qualname__)
            == (cls.__module__, cls.__qualname__))
        if existing is not None and not same and not replace:
            raise ValueError(
                f"Recipe {name!r} is already registered by {existing.__module__}."
                f"{existing.__qualname__}; pass register_recipe({name!r}, replace=True) to "
                "override it.")
        _RECIPE_REGISTRY[name] = cls
        doc_first = (cls.__doc__ or "").strip().splitlines()
        _RECIPE_HELP[name] = help or (doc_first[0] if doc_first else "")
        cls.name = name
        return cls

    return deco


def get_recipe(name: str) -> Recipe:
    try:
        cls = _RECIPE_REGISTRY[name]
    except KeyError:
        from topologies._util.suggest import did_you_mean
        from topologies.training.errors import TrainingError as _TE

        raise _TE(
            f"Unknown recipe {name!r}.{did_you_mean(name, _RECIPE_REGISTRY, n=3)}",
            stage="resolve",
            available=sorted(_RECIPE_REGISTRY.keys()),
            hint="Use list_recipes() or `topologies recipes list` to see available recipes.",
        ) from None
    return cls()


def list_recipes() -> list[dict[str, str]]:
    return [
        {"name": k, "help": _RECIPE_HELP.get(k, ""), "level": _RECIPE_REGISTRY[k].level}
        for k in sorted(_RECIPE_REGISTRY.keys())
    ]


def recipe_schema(name: str) -> dict:
    r = get_recipe(name)
    out: dict[str, Any] = {
        "name": r.name,
        "level": r.level,
        "config_schema": r.ConfigSchema.model_json_schema(),
        "expected_batch": r.expected_batch_spec(),
        "optional_batch_keys": sorted(r.optional_batch_keys()),
        "default_config": r.default_config(),
        "default_training_policy": r.default_training_policy(),
    }
    if r.ComponentsSchema is not None:
        try:
            out["components_schema"] = r.ComponentsSchema.model_json_schema()
        except Exception:  # noqa: BLE001
            out["components_schema"] = {"title": r.ComponentsSchema.__name__}
    else:
        out["components_schema"] = None
    return out


def auto_recipe_for(*args: Any, **kwargs: Any) -> Any:
    """Back-compat alias for resolve.resolve_recipe (import here to avoid cycles)."""
    from topologies.training.resolve import resolve_recipe

    return resolve_recipe(*args, **kwargs)
