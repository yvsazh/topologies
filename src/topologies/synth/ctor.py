"""Constructor recording: how YAML specs serialize arbitrary spec parts.

Classes that derive from ``Recorded`` remember the arguments they were built
with (``obj._ctor = (cls, args, kwargs)``), so ``Spec.to_yaml()`` can write them
back as ``{call: pkg.mod:Class, ...kwargs}`` nodes. Builder-style objects (the
SCM) also log their method calls (``@logged``), written as ``then: [...]``.

    class Shapes(Generator): ...         # Generator derives from Recorded
    Shapes(size=32)._ctor                # (Shapes, (), {"size": 32})
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any


def _wrap_init(init: Callable[..., None]) -> Callable[..., None]:
    @functools.wraps(init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        d = getattr(self, "__dict__", None)
        if d is not None and "_ctor" not in d:
            d["_ctor"] = (type(self), args, dict(kwargs))
        init(self, *args, **kwargs)

    __init__._records_ctor = True  # type: ignore[attr-defined]
    return __init__


class Recorded:
    """Mixin: instances remember their constructor arguments (see module docstring)."""

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        init = cls.__dict__.get("__init__")
        if init is not None and not getattr(init, "_records_ctor", False):
            cls.__init__ = _wrap_init(init)  # type: ignore[method-assign]


def logged(method: Callable[..., Any]) -> Callable[..., Any]:
    """Builder method decorator: append ``(name, args, kwargs)`` to ``self._calls``."""

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls = self.__dict__.setdefault("_calls", [])
        depth = self.__dict__.get("_call_depth", 0)
        if depth == 0:
            calls.append((method.__name__, args, dict(kwargs)))
        self.__dict__["_call_depth"] = depth + 1
        try:
            return method(self, *args, **kwargs)
        finally:
            self.__dict__["_call_depth"] = depth

    return wrapper


def factory(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Factory-function decorator: the returned object serializes as a call of ``fn``
    (``synth.personas(200)`` instead of 200 expanded values)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        obj = fn(*args, **kwargs)
        d = getattr(obj, "__dict__", None)
        if d is not None:
            d["_ctor"] = (wrapper, args, dict(kwargs))
            d.pop("_ctor_stale", None)
        return obj

    return wrapper


def set_recorded(obj: Any, key: str, value: Any) -> None:
    """``obj.key = value`` was applied to a copy: keep ``_ctor`` in sync (or mark it
    stale when ``key`` is not a constructor argument)."""
    import inspect

    ctor = getattr(obj, "__dict__", {}).get("_ctor")
    if ctor is None:
        return
    cls, args, kwargs = ctor
    try:
        sig = inspect.signature(cls.__init__ if isinstance(cls, type) else cls)
        plist = list(sig.parameters.values())[1 if isinstance(cls, type) else 0:]
    except (TypeError, ValueError):
        plist = []
    params = [p.name for p in plist]
    kwargs = dict(kwargs)
    if key in kwargs:
        kwargs[key] = value
    elif key in params:
        i = params.index(key)
        positional = all(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                         for p in plist[:i + 1])
        if i < len(args) and positional:
            args = (*args[:i], value, *args[i + 1:])
        elif i >= len(args):
            kwargs[key] = value
        else:
            obj.__dict__["_ctor_stale"] = key
            return
    else:
        obj.__dict__["_ctor_stale"] = key
        return
    obj.__dict__["_ctor"] = (cls, args, kwargs)


__all__ = ["Recorded", "factory", "logged", "set_recorded"]
