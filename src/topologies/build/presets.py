"""Named size presets (``tiny`` / ``small`` / ``base`` ...) for any architecture.

Presets live in a registry keyed by the class (or factory), so they can be
declared on your own architectures with a decorator *or* attached to
classes you do not own:

    @presets(tiny=dict(dim=64, layers=2), small=dict(dim=256, layers=6))
    class MyNet(nn.Module): ...

    register_presets(SomeoneElsesNet, tiny=dict(width=16))

Subclasses inherit their parents' presets (and may override them).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_PRESETS: dict[Any, dict[str, dict[str, Any]]] = {}
_KNOBS: dict[Any, dict[str, Any]] = {}


def register_presets(target: Any, **named: dict[str, Any]) -> Any:
    """Attach named keyword presets to ``target`` (class or factory callable)."""
    for name, kw in named.items():
        if not isinstance(kw, dict):
            raise TypeError(f"Preset {name!r} must be a dict of constructor kwargs.")
    _PRESETS.setdefault(target, {}).update({k: dict(v) for k, v in named.items()})
    return target


def presets(**named: dict[str, Any]) -> Callable[[Any], Any]:
    """Decorator form of ``register_presets``."""
    def deco(target: Any) -> Any:
        return register_presets(target, **named)

    return deco


def register_budget_knobs(target: Any, **knobs: Any) -> Any:
    """Declare default integer search spaces for ``build(..., target_params=...)``.

    ``register_budget_knobs(MyNet, width=range(16, 1025, 16), depth=range(1, 9))``
    """
    _KNOBS.setdefault(target, {}).update({k: list(v) for k, v in knobs.items()})
    return target


def budget_knobs(**knobs: Any) -> Callable[[Any], Any]:
    """Decorator form of ``register_budget_knobs``."""
    def deco(target: Any) -> Any:
        return register_budget_knobs(target, **knobs)

    return deco


def _lineage(target: Any) -> list[Any]:
    if isinstance(target, type):
        return list(reversed(target.__mro__))
    return [target]


def _catalog_defaults(target: Any, key: str) -> dict[str, Any]:
    """Presets / knobs declared for catalog classes in ``topologies.build.catalog``."""
    from topologies.build.catalog import catalog_spec

    return catalog_spec(target).get(key, {})


def get_presets(target: Any) -> dict[str, dict[str, Any]]:
    """All presets visible for ``target`` (inherited ones first, overrides win)."""
    out = {k: dict(v) for k, v in _catalog_defaults(target, "presets").items()}
    for t in _lineage(target):
        out.update({k: dict(v) for k, v in _PRESETS.get(t, {}).items()})
    return out


def get_budget_knobs(target: Any) -> dict[str, list]:
    out = {k: list(v) for k, v in _catalog_defaults(target, "budget_knobs").items()}
    for t in _lineage(target):
        out.update({k: list(v) for k, v in _KNOBS.get(t, {}).items()})
    return out


def resolve_preset(target: Any, size: str) -> dict[str, Any]:
    available = get_presets(target)
    if size in available:
        return dict(available[size])
    name = getattr(target, "__name__", repr(target))
    if not available:
        raise ValueError(
            f"{name} declares no size presets. Pass explicit kwargs, or register some with "
            f"topologies.build.register_presets({name}, tiny={{...}})."
        )
    import difflib

    sug = difflib.get_close_matches(size, list(available), n=1)
    hint = f" Did you mean {sug[0]!r}?" if sug else ""
    raise ValueError(f"Unknown size {size!r} for {name}.{hint} Available: {list(available)}.")
