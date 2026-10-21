"""ModelAdapter / ModelSpec for arbitrary external nn.Modules."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from topologies.training.errors import TrainingError


class ModelSpec(BaseModel):
    """Optional external training metadata (no registry required)."""

    model_config = ConfigDict(extra="allow")

    recipe: str | None = None
    input_keys: list[str] = Field(default_factory=list)
    output_kind: str = "logits"
    defaults: dict[str, Any] = Field(default_factory=dict)


class ModelAdapter:
    """Adapt dict batches -> model.forward(...) -> adapted output.

    - ``input_map`` renames batch keys to forward parameter names.
    - ``forward_map`` fully overrides the call: ``forward_map(model, batch)``.
    - ``output_map`` post-processes the raw model output.
    Never modifies the wrapped model.
    """

    def __init__(
        self,
        input_map: dict[str, str] | None = None,
        forward_map: Callable[[nn.Module, dict], Any] | None = None,
        output_map: Callable[[Any], Any] | None = None,
    ) -> None:
        self.input_map = dict(input_map or {})
        self.forward_map = forward_map
        self.output_map = output_map

    def adapted_batch(self, batch: dict) -> dict:
        if not self.input_map:
            return dict(batch)
        out: dict = {}
        for k, v in batch.items():
            out[self.input_map.get(k, k)] = v
        # Detect missing renames that drop required keys is done at call time.
        return out

    def call(self, model: nn.Module, batch: dict, *, primary: str | None = None) -> Any:
        if self.forward_map is not None:
            try:
                out = self.forward_map(model, batch)
            except Exception as e:
                raise TrainingError(
                    "ModelAdapter.forward_map failed.",
                    stage="adapter",
                    received=f"batch keys={sorted(batch.keys())}",
                    hint=str(e),
                ) from e
            return self.output_map(out) if self.output_map else out
        adapted = self.adapted_batch(batch)
        if primary is not None and primary not in adapted:
            primary = self.input_map.get(primary, primary)
        args, kwargs = bind_inputs(model, adapted, primary=primary)
        try:
            out = model(*args, **kwargs)
        except TypeError as e:
            raise TrainingError(
                f"Argument binding failed for {type(model).__name__}.forward(): {e}",
                stage="adapter",
                received=f"batch keys={sorted(adapted.keys())}",
                expected=_signature_str(model),
                hint="Provide input_map={'<batch-key>': '<forward-arg>'} or forward_map=...",
            ) from e
        return self.output_map(out) if self.output_map else out


#: Batch keys that may bind to differently-named forward() parameters.
INPUT_ALIASES: dict[str, tuple[str, ...]] = {
    "mask": ("attention_mask",),
    "tokens": ("input_ids",),
    "pixels": ("images", "pixel_values"),
}


def _forward_of(model: nn.Module) -> Any:
    """The user model's forward, looking through DDP / compile wrappers."""
    from topologies.build.contract import unwrap

    return unwrap(model).forward


def _signature_str(model: nn.Module) -> str:
    try:
        return f"forward{inspect.signature(_forward_of(model))}"
    except (TypeError, ValueError):
        return "forward(...)"


def bind_inputs(model: nn.Module, inputs: dict, *, primary: str | None = None
                ) -> tuple[list, dict]:
    """Map batch inputs onto ``model.forward``'s parameters.

    1. Keys matching parameter names (or a known alias such as
       ``mask -> attention_mask``) bind by keyword.
    2. If exactly one required positional parameter is still unbound and the
       caller named a ``primary`` input (recipes do: ``pixels``, ``tokens``),
       it is passed positionally — so ``nn.Sequential``, ``forward(x)``,
       ``forward(images)`` all work. Without ``primary`` binding stays strict.
    3. Otherwise a ``TrainingError`` names the missing parameters.
    Unknown optional keys (e.g. a ``mask`` the model does not accept) are dropped.
    """
    try:
        params = list(inspect.signature(_forward_of(model)).parameters.values())
    except (TypeError, ValueError):
        return [], dict(inputs)
    if any(p.kind == p.VAR_KEYWORD for p in params):
        return [], dict(inputs)
    named = [p for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    names = {p.name for p in named}
    kwargs: dict = {}
    used: set[str] = set()
    for k, v in inputs.items():
        if k in names:
            kwargs[k] = v
            used.add(k)
    for k, v in inputs.items():
        if k in used:
            continue
        for alias in INPUT_ALIASES.get(k, ()):
            if alias in names and alias not in kwargs:
                kwargs[alias] = v
                used.add(k)
                break
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    required = [p for p in params
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                and p.default is p.empty and p.name not in kwargs]
    has_var_pos = any(p.kind == p.VAR_POSITIONAL for p in params)
    if not required and (kwargs or not inputs or not has_var_pos):
        return [], kwargs
    if primary is not None and primary in inputs and primary not in used:
        unbound_pos = [p for p in positional if p.name not in kwargs]
        if (len(required) == 1 and unbound_pos and required[0] is unbound_pos[0]) or (
            not required and has_var_pos
        ):
            return [inputs[primary]], kwargs
    missing = [p.name for p in required]
    raise TrainingError(
        "Argument binding failed for model.forward().",
        stage="adapter",
        expected=_signature_str(model),
        received=f"batch keys={sorted(inputs.keys())}",
        hint=(
            f"Missing {missing}. Provide input_map to rename batch keys, e.g. "
            f"input_map={{'{next(iter(inputs), '<batch-key>')}': '{missing[0] if missing else '<arg>'}'}}, "
            "or forward_map=lambda model, batch: model(...)."
        ),
    )


def call_with_adapter(
    adapter: ModelAdapter | None,
    model: nn.Module,
    batch: dict,
    keys: list[str] | None = None,
    *,
    primary: str | None = None,
) -> Any:
    """Call ``model`` honoring a Trainer-level adapter when it defines behavior.

    - If ``adapter`` carries ``forward_map``/``output_map``, it is authoritative
      and receives the (already input-mapped) batch.
    - Otherwise ``keys`` (default: all batch keys) are bound to ``forward``
      by name, alias, or — for single-input models — positionally via
      ``primary``, so plain input_map renaming (already applied by Trainer)
      keeps working.
    """
    if adapter is not None and (
        adapter.forward_map is not None or adapter.output_map is not None
    ):
        return adapter.call(model, batch, primary=primary)
    sub = {k: batch[k] for k in (keys or list(batch.keys())) if k in batch}
    return ModelAdapter().call(model, sub, primary=primary)


def model_spec_for(model: nn.Module, explicit: ModelSpec | dict | None = None) -> ModelSpec | None:
    if explicit is not None:
        if isinstance(explicit, ModelSpec):
            return explicit
        return ModelSpec(**dict(explicit))
    spec = getattr(model, "__topologies_spec__", None)
    if spec is None:
        return None
    if isinstance(spec, ModelSpec):
        return spec
    if isinstance(spec, dict):
        try:
            return ModelSpec(**spec)
        except Exception:  # noqa: BLE001
            return None
    return None
