"""``build()``: one construction entry point for every architecture.

    build(TransformerLM, "small", vocab_size=tok.vocab_size, max_seq=256)
    build(TransformerLM, vocab_size=512, target_params="2M")
    build(MyNet, target_params="15k", search={"width": range(8, 513, 8)})
    build("llama-decoder", vocab=128)          # registry id / class name
    build("my_pkg.models:MyNet", size="tiny")  # import path

Precedence of keyword arguments: preset (``size``) < explicit kwargs; with
``target_params`` the free knobs (those not given explicitly) are searched.
The built model records its constructor call, so ``save_pretrained`` can
rebuild it later.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from topologies.build.budget import BudgetResult, fit_budget, measure_params
from topologies.build.config import import_path, set_init_config
from topologies.build.params import format_params
from topologies.build.presets import get_budget_knobs, get_presets, resolve_preset


def resolve_architecture(arch: Any) -> Callable[..., nn.Module]:
    """Class/factory from a class, callable, import path, or catalog name."""
    if callable(arch) and not isinstance(arch, str):
        return arch
    if not isinstance(arch, str):
        raise TypeError(f"Cannot build from {type(arch).__name__}; pass a class or name.")
    if ":" in arch:
        return import_path(arch)
    import topologies.nn as tnn

    if arch in getattr(tnn, "__all__", ()):
        return getattr(tnn, arch)
    from topologies.models import get_model

    return get_model(arch)


def _signature_defaults(factory: Any) -> dict[str, Any]:
    import inspect

    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return {}
    return {n: p.default for n, p in params.items()
            if p.default is not p.empty and isinstance(p.default, int)
            and not isinstance(p.default, bool)}


def build(
    arch: Any,
    size: str | None = None,
    *,
    target_params: float | str | None = None,
    search: dict[str, Any] | None = None,
    tolerance: float | None = 0.15,
    param_count: str | Callable[..., int] = "total",
    **kwargs: Any,
) -> nn.Module:
    """Construct ``arch`` from a size preset and/or a parameter budget.

    ``search`` maps knob names to candidate integers; it defaults to the
    knobs the architecture declares via ``register_budget_knobs``. Explicit
    ``kwargs`` are never searched over. ``model.build_report`` holds the
    budget result when ``target_params`` is used. ``param_count`` chooses
    what ``target_params`` counts: ``"total"`` (default), ``"trainable"``,
    ``"non_embedding"`` or a callable ``fn(module) -> int``.
    """
    factory = resolve_architecture(arch)
    final: dict[str, Any] = resolve_preset(factory, size) if size else {}
    final.update(kwargs)
    report: BudgetResult | None = None
    if target_params is not None:
        knobs = dict(search) if search is not None else get_budget_knobs(factory)
        knobs = {k: v for k, v in knobs.items() if k not in kwargs}
        fixed = {k: v for k, v in final.items() if k not in knobs}
        anchor = {**_signature_defaults(factory), **final}
        report = fit_budget(factory, target_params, knobs, fixed=fixed, tolerance=tolerance,
                            anchor={k: v for k, v in anchor.items() if k in knobs},
                            count=param_count)
        final = {**fixed, **report.kwargs}
    model = factory(**final)
    set_init_config(model, factory, final)
    if report is not None:
        object.__setattr__(model, "build_report", report)
    return model


def size_table(arch: Any, **required: Any) -> list[dict[str, Any]]:
    """Presets of ``arch`` with parameter counts for the given required kwargs.

    >>> size_table(TransformerLM, vocab_size=2048)
    [{'size': 'tiny', 'params': 853_000, 'pretty': '853.0K', 'kwargs': {...}}, ...]
    """
    factory = resolve_architecture(arch)
    rows = []
    for name, kw in get_presets(factory).items():
        full = {**kw, **required}
        try:
            n = measure_params(factory, full)
        except TypeError:
            n = None  # a required constructor argument was not supplied
        rows.append({"size": name, "params": n, "pretty": format_params(n), "kwargs": kw})
    return rows
