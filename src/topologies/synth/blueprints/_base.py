"""Blueprint registry and shared helpers.

A blueprint is a function ``fn(name, **options) -> Spec``. ``@blueprint(example=...)``
registers it, records the call (so ``spec.to_yaml()`` writes the compact
``blueprint: {name: ..., ...}`` form and ``spec.with_(...)`` stays a blueprint plus
overrides) and stores the example options ``topologies synth init`` scaffolds.
Everything a blueprint builds is made of module-level functions and recorded
parts, so the expanded spec is YAML-serializable too.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from topologies.synth.errors import SynthError, unknown

BLUEPRINTS: dict[str, Callable[..., Any]] = {}
#: Spec keyword arguments every blueprint passes through unchanged.
SPEC_KEYS = ("root", "seed", "workers", "on_error", "version", "meta", "chunk_size",
             "compress", "max_raw", "max_time", "monitor", "reference", "oversample",
             "sampler", "where", "require")


def blueprint(*, example: Mapping[str, Any], aliases: Sequence[str] = ()) -> Callable[..., Any]:
    """Register a blueprint function (see module docstring)."""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        def make(kind: str) -> Callable[..., Any]:
            @functools.wraps(fn)
            def wrapper(name: str, **options: Any) -> Any:
                bad = [k for k in options if k not in _params(fn)]
                if bad:
                    raise unknown(f"{kind} option", bad[0], _params(fn), stage="blueprint",
                                  extra=f"See docs/synth/blueprints/{fn.__name__}.md.")
                spec = fn(name, **options)
                spec._from_blueprint = {"name": kind, **options}
                spec._bp_overrides = {}
                return spec

            wrapper.example = dict(example)  # type: ignore[attr-defined]
            wrapper.blueprint = kind  # type: ignore[attr-defined]
            return wrapper

        main = make(fn.__name__)
        BLUEPRINTS[fn.__name__] = main
        for a in aliases:
            BLUEPRINTS[a] = make(a)
        return main

    return deco


@functools.cache
def _params_cached(fn: Callable[..., Any]) -> tuple[str, ...]:
    import inspect

    params = inspect.signature(fn).parameters.values()
    names = [p.name for p in params if p.name != "name" and p.kind != p.VAR_KEYWORD]
    if any(p.kind == p.VAR_KEYWORD for p in params):
        names += [k for k in SPEC_KEYS if k not in names]
    return tuple(names)


def _params(fn: Callable[..., Any]) -> list[str]:
    return list(_params_cached(fn))


def spec_kwargs(kw: Mapping[str, Any]) -> dict[str, Any]:
    """The pass-through ``Spec`` arguments among a blueprint's ``**spec``."""
    bad = [k for k in kw if k not in SPEC_KEYS]
    if bad:
        raise unknown("Spec argument", bad[0], SPEC_KEYS, stage="blueprint")
    return dict(kw)


def merge_vary(base: Mapping[str, Any], vary: Mapping[str, Any] | None) -> dict[str, Any]:
    """Default attributes updated by ``vary`` (``None`` removes an attribute)."""
    out = dict(base)
    for k, v in (vary or {}).items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = v
    return out


def label_map(labels: Any, what: str = "labels") -> dict[str, str]:
    """``["a", "b"]`` or ``{"a": "description"}`` -> ``{name: description}``."""
    if isinstance(labels, Mapping):
        out = {str(k): str(v or k) for k, v in labels.items()}
    elif isinstance(labels, (list, tuple)):
        out = {str(k): str(k) for k in labels}
    else:
        out = {}
    if len(out) < 2 and what == "labels":
        raise SynthError(f"Give at least two {what}: ['a', 'b'] or {{'a': 'what a means'}}.",
                         stage="blueprint")
    if not out:
        raise SynthError(f"{what} is empty.", stage="blueprint")
    return out


def default_splits(stratify: str | None = None, *, group: str | None = None) -> Any:
    from topologies.synth.ops.split import SplitSpec

    return SplitSpec({"train": 0.9, "val": 0.1}, stratify=stratify, group=group)


# -- small module-level functions blueprints reference (YAML-serializable) ----------------
def describe_values(attrs: Mapping[str, Any], *, field: str, desc: Mapping[str, str]) -> str:
    v = attrs.get(field)
    if isinstance(v, (list, tuple)):
        return "; ".join(f"{x}: {desc.get(x, x)}" for x in v)
    return desc.get(v, str(v))


def sample_subset(rng: Any, *, values: Sequence[str], low: int, high: int) -> list[str]:
    k = rng.randint(low, min(high, len(values)))
    picked = rng.sample(list(values), k)
    return sorted(picked, key=list(values).index)


__all__ = ["BLUEPRINTS", "SPEC_KEYS", "blueprint", "default_splits", "describe_values",
           "label_map", "merge_vary", "sample_subset", "spec_kwargs"]
