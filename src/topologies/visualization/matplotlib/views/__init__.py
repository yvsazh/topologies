"""Matplotlib view registry: ``(domain, name) -> view(fig, data, *, theme, options)``.

Domains: ``"training"`` (history and live panels), ``"experiments"``, ``"analysis"`` and
``"result:<kind>"`` for any ``VisualizationResult.kind``. Built-in views register
themselves the same way, so a user registration with the same key replaces them::

    @register_view("training", "grad-health")
    def grad_health(fig, data, *, theme, options):
        ax = fig.subplots()
        ...
        return fig
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

_VIEWS: dict[str, dict[str, Callable[..., Any]]] = {}
_BUILTINS = ("training", "experiments", "analysis", "engineering", "profile", "hardware",
             "generic")
_loaded = False

DEFAULT_VIEW = {"training": "history", "experiments": "overview", "analysis": "auto"}


def _ensure_builtins() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    for name in _BUILTINS:
        importlib.import_module(f"{__name__}.{name}")


def register_view(domain: str, name: str | None = None) -> Callable[[Callable], Callable]:
    """Decorator registering ``fn(fig, data, *, theme, options)`` under ``(domain, name)``.

    ``name`` defaults to ``"default"``; for ``"result:<kind>"`` domains the view is
    picked automatically for results of that kind.
    """
    def decorator(fn: Callable) -> Callable:
        if not callable(fn):
            raise TypeError("a view must be callable")
        _VIEWS.setdefault(str(domain), {})[str(name or "default")] = fn
        return fn

    return decorator


def builtin_view(domain: str, name: str | None = None) -> Callable[[Callable], Callable]:
    """Register a built-in view without replacing a user view registered earlier."""
    def decorator(fn: Callable) -> Callable:
        _VIEWS.setdefault(str(domain), {}).setdefault(str(name or "default"), fn)
        return fn

    return decorator


def get_view(domain: str, name: str | None = None) -> Callable[..., Any] | None:
    _ensure_builtins()
    views = _VIEWS.get(str(domain), {})
    if name is None:
        name = DEFAULT_VIEW.get(domain, "default")
        if name not in views and views:
            return next(iter(views.values()))
    return views.get(str(name))


def view_names(domain: str | None = None) -> list[str]:
    """Registered view names (``"domain/name"`` when ``domain`` is None)."""
    _ensure_builtins()
    if domain is not None:
        return list(_VIEWS.get(domain, {}))
    return [f"{d}/{n}" for d, names in _VIEWS.items() for n in names]


def has_domain(domain: str) -> bool:
    _ensure_builtins()
    return bool(_VIEWS.get(domain))


__all__ = ["builtin_view", "get_view", "has_domain", "register_view", "view_names"]
