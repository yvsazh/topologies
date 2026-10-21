"""Constructor-argument capture so any model can be rebuilt from a saved config.

``Configurable`` records the arguments its ``__init__`` was called with
(opt-in for toolbox stacks and user classes; catalog classes are plain
``nn.Module`` and get their record from ``build()`` instead).
``save_pretrained`` stores them, ``load_pretrained`` re-imports the class and
calls it again — no per-model serialization code.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import json
from collections.abc import Iterable, Iterator
from typing import Any

import torch.nn as nn

INIT_ATTR = "_topologies_init"


def class_path(obj: Any) -> str:
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}:{cls.__qualname__}"


def import_path(path: str) -> Any:
    """Inverse of ``class_path``: ``"pkg.mod:Class"`` -> object."""
    import importlib

    mod, qual = _split_path(path)
    obj = importlib.import_module(mod)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


class UntrustedTargetError(ValueError):
    """A saved config names an import target outside the trusted modules."""


#: Modules whose objects a saved config may name without extra opt-in.
TRUSTED_PREFIXES = ("topologies", "torch")

_TRUST: contextvars.ContextVar[tuple[tuple[str, ...], bool]] = contextvars.ContextVar(
    "topologies_import_trust", default=((), False))


@contextlib.contextmanager
def trusted_imports(trusted_modules: Iterable[str] = (), *,
                    trust_remote_code: bool = False) -> Iterator[None]:
    """Widen what ``decode_value`` / ``safe_import_path`` may import in this block.

    By default only objects from ``topologies.*`` and ``torch.*`` may be
    named by a saved config. ``trusted_modules`` adds module prefixes (e.g.
    ``["my_pkg"]``); ``trust_remote_code=True`` lifts the restriction
    entirely (only for files you created yourself).
    """
    extra, remote = _TRUST.get()
    token = _TRUST.set((extra + tuple(trusted_modules), remote or trust_remote_code))
    try:
        yield
    finally:
        _TRUST.reset(token)


def _split_path(path: str) -> tuple[str, str]:
    if ":" in path:
        mod, _, qual = path.partition(":")
    else:
        mod, _, qual = path.rpartition(".")
    return mod, qual


def _module_trusted(module: str) -> bool:
    extra, remote = _TRUST.get()
    if remote:
        return True
    return any(module == p or module.startswith(p + ".")
               for p in (*TRUSTED_PREFIXES, *extra))


def safe_import_path(path: str) -> Any:
    """``import_path`` restricted to trusted modules (see ``trusted_imports``).

    The module is checked *before* it is imported, so an untrusted config
    cannot run module-level code either.
    """
    if not isinstance(path, str) or not path:
        raise UntrustedTargetError(f"untrusted constructor target {path!r}")
    mod, _ = _split_path(path)
    if not _module_trusted(mod):
        raise UntrustedTargetError(
            f"untrusted constructor target {path!r}: saved configs may only name objects "
            f"from {', '.join(TRUSTED_PREFIXES)}. If you trust this file, pass "
            f"trusted_modules=[{mod.split('.')[0]!r}] (or trust_remote_code=True).")
    return import_path(path)


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


_UNENCODABLE = object()


def _importable(obj: Any) -> str | None:
    mod, qual = getattr(obj, "__module__", None), getattr(obj, "__qualname__", None)
    if not mod or not qual or "<" in qual or mod == "__main__":
        return None
    try:
        return f"{mod}:{qual}" if import_path(f"{mod}:{qual}") is obj else None
    except Exception:  # noqa: BLE001
        return None


def encode_value(value: Any) -> Any:
    """JSON view of a constructor argument; importable callables become
    ``{"__import__": "mod:qual"}`` and objects with ``to_config()`` (e.g.
    ``topologies.nn.slot``) ``{"__object__": "mod:Class", "config": {...}}``.
    Returns a private sentinel when the value cannot be recorded."""
    if _jsonable(value):
        return value
    if isinstance(value, (list, tuple)):
        out = [encode_value(v) for v in value]
        return _UNENCODABLE if any(v is _UNENCODABLE for v in out) else out
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        enc = {k: encode_value(v) for k, v in value.items()}
        return _UNENCODABLE if any(v is _UNENCODABLE for v in enc.values()) else enc
    to_config = getattr(value, "to_config", None)
    if callable(to_config) and not isinstance(value, type):
        path = _importable(type(value))
        try:
            cfg = to_config()
        except Exception:  # noqa: BLE001
            cfg = None
        if path and cfg is not None and _jsonable(cfg):
            return {"__object__": path, "config": cfg}
        return _UNENCODABLE
    if callable(value):
        path = _importable(value)
        return {"__import__": path} if path else _UNENCODABLE
    return _UNENCODABLE


def decode_value(value: Any) -> Any:
    """Inverse of ``encode_value``.

    Only objects from trusted modules are imported (see ``trusted_imports``),
    and ``__object__`` targets must be classes.
    """
    if isinstance(value, dict):
        if set(value) == {"__import__"}:
            return safe_import_path(value["__import__"])
        if set(value) == {"__object__", "config"}:
            cls = safe_import_path(value["__object__"])
            if not isinstance(cls, type):
                raise UntrustedTargetError(
                    f"untrusted constructor target {value['__object__']!r}: not a class")
            from_config = getattr(cls, "from_config", None)
            return from_config(value["config"]) if callable(from_config) else \
                cls(**value["config"])
        return {k: decode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_value(v) for v in value]
    return value


def make_init_record(target: Any, kwargs: dict) -> dict:
    """``{"target": "mod:Class", "kwargs": {...}, "complete": bool}``."""
    encoded = {k: encode_value(v) for k, v in kwargs.items()}
    clean = {k: v for k, v in encoded.items() if v is not _UNENCODABLE}
    if isinstance(target, type):
        tpath = class_path(target)
    else:
        tpath = f"{getattr(target, '__module__', '?')}:{getattr(target, '__qualname__', '?')}"
    return {"target": tpath, "kwargs": clean, "complete": len(clean) == len(kwargs)}


def _wrap_init(cls: type) -> None:
    init = cls.__dict__.get("__init__")
    if init is None or getattr(init, "_topologies_wrapped", False):
        return
    try:
        sig = inspect.signature(init)
    except (TypeError, ValueError):
        return

    @functools.wraps(init)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        outermost = not getattr(self, "_topologies_in_init", False)
        if outermost:
            object.__setattr__(self, "_topologies_in_init", True)
        try:
            init(self, *args, **kwargs)
        finally:
            if outermost:
                object.__setattr__(self, "_topologies_in_init", False)
        if outermost:
            try:
                bound = sig.bind(self, *args, **kwargs)
                bound.apply_defaults()
                params = dict(bound.arguments)
                params.pop(next(iter(sig.parameters)), None)  # self
                for name, p in sig.parameters.items():
                    if p.kind is p.VAR_KEYWORD and name in params:
                        params.update(params.pop(name) or {})
                    elif p.kind is p.VAR_POSITIONAL and name in params and params.pop(name):
                        return  # positional varargs cannot be replayed by name
                object.__setattr__(self, INIT_ATTR, make_init_record(type(self), params))
            except Exception:  # noqa: BLE001, S110 - capture is best-effort
                pass

    __init__._topologies_wrapped = True  # type: ignore[attr-defined]
    cls.__init__ = __init__  # type: ignore[method-assign]


class Configurable:
    """Mixin: subclasses automatically record their constructor arguments."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _wrap_init(cls)


def init_config(model: nn.Module) -> dict | None:
    """The recorded ``{"target", "kwargs", "complete"}`` for ``model`` (or None)."""
    rec = getattr(model, INIT_ATTR, None)
    if rec is None:
        from topologies.build.contract import unwrap

        inner = unwrap(model)
        if inner is not model:
            return init_config(inner)
    return dict(rec) if isinstance(rec, dict) else None


def set_init_config(model: nn.Module, target: Any, kwargs: dict) -> None:
    object.__setattr__(model, INIT_ATTR, make_init_record(target, kwargs))
