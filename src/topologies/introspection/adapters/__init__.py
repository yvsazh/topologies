"""Adapters: optional, explicit descriptions of a model's semantics.

An adapter maps semantic roles (``residual``, ``attention``, ``unembed`` ...) onto
existing modules and declares axes. Universal techniques never need one;
semantic techniques refuse to guess without one.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from topologies.introspection.semantics import Axes

ROLES = ("embed", "position", "block", "residual", "residual_pre", "attention", "attention_out",
         "mlp", "mlp_hidden", "norm", "final_norm", "unembed", "head", "encoder", "decoder",
         "bottleneck", "latent", "denoiser", "router", "expert", "patch", "stage")


@dataclass(frozen=True)
class AttentionSite:
    """Where attention weights come from for one attention module."""

    path: str
    method: str = "sdpa"           # "sdpa" | "mha" | "softmax" | "native"
    heads: int | None = None
    note: str | None = None


@dataclass(frozen=True)
class ProcessSpec:
    """How a loop's steps are identified (occurrences of ``step_module``)."""

    step_from: str = "occurrence"
    step_module: str | None = None
    sampler_callback: bool = False


class Adapter:
    """Base adapter: no semantics. Subclass and override what the model can declare."""

    name: str = "adapter"
    priority: int = 0

    def matches(self, model: nn.Module) -> float:
        return 0.0

    def roles(self, index: Any) -> dict[str, list[str]]:
        return {}

    def axes(self, index: Any) -> dict[str, Axes]:
        return {}

    def decoders(self, model: nn.Module) -> dict[str, Callable]:
        return {}

    def attention_sites(self, index: Any) -> list[AttentionSite]:
        return []

    def input_render(self, subject: Any) -> Callable | None:
        return None

    def process(self) -> ProcessSpec | None:
        return None

    def notes(self) -> list[str]:
        return []

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "type": f"{type(self).__module__}.{type(self).__qualname__}"}


class AdapterRegistry:
    """Adapters keyed by name; ``resolve`` picks the highest ``matches`` score."""

    def __init__(self) -> None:
        self._adapters: dict[str, Adapter] = {}
        self._factories: dict[str, Callable[[], Adapter]] = {}

    def register(self, adapter: Adapter | Callable[[], Adapter], name: str | None = None) -> None:
        if isinstance(adapter, Adapter):
            self._adapters[name or adapter.name] = adapter
        elif callable(adapter):
            self._factories[name or getattr(adapter, "name", adapter.__name__)] = adapter
        else:
            raise TypeError("register an Adapter instance or a factory returning one")

    def names(self) -> list[str]:
        return sorted({*self._adapters, *self._factories})

    def get(self, name: str) -> Adapter:
        if name in self._adapters:
            return self._adapters[name]
        if name in self._factories:
            adapter = self._factories[name]()
            self._adapters[name] = adapter
            return adapter
        raise KeyError(f"unknown adapter {name!r}; registered: {self.names()}")

    def candidates(self, model: nn.Module) -> list[tuple[float, Adapter]]:
        scored = []
        for name in self.names():
            adapter = self.get(name)
            try:
                score = float(adapter.matches(model))
            except Exception:  # noqa: BLE001 - a broken third-party matcher never blocks resolution
                score = 0.0
            if score > 0:
                scored.append((score + adapter.priority * 1e-3, adapter))
        return sorted(scored, key=lambda item: -item[0])

    def resolve(self, model: nn.Module, requested: Any = "auto") -> tuple[Adapter, dict[str, Any]]:
        """User-provided > registered best match > ``GenericAdapter``; returns provenance too."""
        from topologies.introspection.adapters.generic import GenericAdapter

        if isinstance(requested, Adapter):
            return requested, {"adapter": requested.name, "source": "user"}
        if requested is None:
            return GenericAdapter(), {"adapter": "generic", "source": "requested-none"}
        if isinstance(requested, str) and requested != "auto":
            adapter = self.get(requested)
            return adapter, {"adapter": adapter.name, "source": "named"}
        candidates = self.candidates(model)
        if not candidates:
            return GenericAdapter(), {"adapter": "generic", "source": "fallback", "alternatives": []}
        score, best = candidates[0]
        return best, {"adapter": best.name, "source": "auto", "score": score,
                      "alternatives": [(a.name, s) for s, a in candidates[1:]]}


DEFAULT_ADAPTERS = AdapterRegistry()
_BUILTINS_LOADED = False


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    try:
        from topologies.introspection.adapters import builtins  # noqa: F401
    except ImportError:  # pragma: no cover - builtins arrive with phase 4
        pass


def register_adapter(adapter: Adapter | Callable[[], Adapter], name: str | None = None) -> None:
    """Register globally; prefer passing ``adapter=`` to ``introspect`` in plugins."""
    DEFAULT_ADAPTERS.register(adapter, name)


def resolve_adapter(model: nn.Module, requested: Any = "auto") -> tuple[Adapter, dict[str, Any]]:
    _load_builtins()
    return DEFAULT_ADAPTERS.resolve(model, requested)


__all__ = ["DEFAULT_ADAPTERS", "ROLES", "Adapter", "AdapterRegistry", "AttentionSite",
           "ProcessSpec", "register_adapter", "resolve_adapter"]
