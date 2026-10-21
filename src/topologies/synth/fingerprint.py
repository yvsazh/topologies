"""Content fingerprints for spec parts (functions, templates, ops, values).

Functions hash their *source* (plus defaults and closure values), so editing
a generator or a lambda creates a new dataset version, while renaming the
file that defines it does not. Objects with a ``fingerprint()`` method use
it; everything else goes through ``experiments.spec.canonical``.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import textwrap
from typing import Any

from topologies.experiments.cache import stable_hash


def _sha(text: str | bytes, n: int = 16) -> str:
    data = text.encode("utf-8", "surrogatepass") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()[:n]


def callable_fingerprint(fn: Any, _depth: int = 0) -> str | None:
    """Stable hash of a callable's behaviour (source, defaults, closure, partial args)."""
    if fn is None:
        return None
    if _depth > 4:
        return "<deep>"
    fp = getattr(fn, "fingerprint", None)
    if callable(fp) and not isinstance(fn, type) and not inspect.isfunction(fn):
        try:
            return str(fp())
        except Exception:  # noqa: BLE001, S110 - fall through to the generic rules
            pass
    if isinstance(fn, functools.partial):
        return stable_hash({"partial": callable_fingerprint(fn.func, _depth + 1),
                            "args": fp_canonical(list(fn.args), _depth + 1),
                            "kw": fp_canonical(dict(fn.keywords), _depth + 1)})
    target = inspect.unwrap(fn) if callable(fn) else fn
    if isinstance(target, type):
        from topologies.experiments.spec import import_name

        name = import_name(target) or target.__qualname__
        try:
            src = inspect.getsource(target)
        except (OSError, TypeError):
            src = ""
        return stable_hash({"class": name, "src": _sha(src) if src else None})
    if not (inspect.isfunction(target) or inspect.ismethod(target)):
        # A callable object: type + state.
        state = getattr(target, "__dict__", None)
        return stable_hash({"type": f"{type(target).__module__}:{type(target).__qualname__}",
                            "state": fp_canonical(state, _depth + 1) if state else None})
    if inspect.ismethod(target):
        return stable_hash({"method": callable_fingerprint(target.__func__, _depth + 1),
                            "self": fp_canonical(getattr(target.__self__, "__dict__", None),
                                                 _depth + 1)})
    code = getattr(target, "__code__", None)
    try:
        src = textwrap.dedent(inspect.getsource(target))
    except (OSError, TypeError):
        src = ""
    parts: dict[str, Any] = {"name": getattr(target, "__qualname__", "?")}
    if src:
        parts["src"] = _sha(src)
    elif code is not None:
        parts["code"] = _sha(code.co_code + repr(code.co_consts).encode())
    if getattr(target, "__defaults__", None):
        parts["defaults"] = fp_canonical(list(target.__defaults__), _depth + 1)
    if getattr(target, "__kwdefaults__", None):
        parts["kwdefaults"] = fp_canonical(dict(target.__kwdefaults__), _depth + 1)
    cells = getattr(target, "__closure__", None) or ()
    if cells and code is not None:
        closure = {}
        for name, cell in zip(code.co_freevars, cells):
            try:
                closure[name] = fp_canonical(cell.cell_contents, _depth + 1)
            except ValueError:  # empty cell
                closure[name] = None
        parts["closure"] = closure
    return stable_hash(parts)


def fp_canonical(value: Any, _depth: int = 0) -> Any:
    """JSON-able, behaviour-identifying view of ``value`` (callables by source)."""
    from topologies.experiments.spec import canonical

    if value is None or isinstance(value, (bool, int, float, str)):
        return canonical(value, warn=False)
    if isinstance(value, dict):
        return {str(k): fp_canonical(v, _depth + 1) for k, v in sorted(
            value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [fp_canonical(v, _depth + 1) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((fp_canonical(v, _depth + 1) for v in value), key=repr)
    to_config = getattr(value, "to_config", None)
    if callable(to_config) and not isinstance(value, type):
        try:
            return {"__type__": type(value).__name__, **fp_canonical(to_config(), _depth + 1)}
        except Exception:  # noqa: BLE001, S110 - user to_config may fail; falls back to other forms
            pass
    fp = getattr(value, "fingerprint", None)
    if callable(fp) and not isinstance(value, type) and not inspect.isfunction(value):
        try:
            return {"__fp__": str(fp())}
        except Exception:  # noqa: BLE001, S110 - user fingerprint() may fail; falls back to other forms
            pass
    if hasattr(value, "model_json_schema") and isinstance(value, type):
        return {"pydantic": canonical(value, warn=False),
                "schema": stable_hash(value.model_json_schema())}
    if callable(value):
        return {"fn": callable_fingerprint(value, _depth + 1)}
    return canonical(value, warn=False)


def fingerprint(value: Any) -> str:
    """``stable_hash`` of ``fp_canonical(value)``."""
    return stable_hash(fp_canonical(value))


__all__ = ["callable_fingerprint", "fingerprint", "fp_canonical"]
