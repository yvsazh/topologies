"""Canonical JSON views, trial hashing, and spec files (YAML / Python) for the CLI.

Canonical form (used for trial IDs and display, §7.3 of the design):

- classes and importable functions -> ``"module:qualname"``;
- other callables (lambdas, closures) -> ``"<callable name@file:line>"`` (with a
  warning: the trial ID then depends on source location);
- tokenizer instances -> ``{"tokenizer": kind, "sha256": ...}``;
- floats keep their repr round-trip, dicts get sorted keys, tuples become lists;
- anything else -> ``repr`` (memory addresses stripped, truncated) with a warning.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import inspect
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any

from topologies.experiments.errors import ExperimentError


class ExperimentWarning(UserWarning):
    """Non-fatal experiment issues (unstable trial IDs, ignored options, ...)."""


_WARNED: set[str] = set()


def warn_once(msg: str) -> None:
    if msg in _WARNED:
        return
    _WARNED.add(msg)
    warnings.warn(msg, ExperimentWarning, stacklevel=3)


def import_name(obj: Any) -> str | None:
    """``"module:qualname"`` when importing that path gives back ``obj``."""
    mod, qual = getattr(obj, "__module__", None), getattr(obj, "__qualname__", None)
    if not mod or not qual or "<" in qual:
        return None
    try:
        from topologies.build.config import import_path

        target = import_path(f"{mod}:{qual}")
    except Exception:  # noqa: BLE001
        return None
    return f"{mod}:{qual}" if target is obj else None


def _source_location(obj: Any) -> str:
    try:
        fn = inspect.unwrap(obj)
        code = getattr(fn, "__code__", None)
        if code is not None:
            return f"{Path(code.co_filename).name}:{code.co_firstlineno}"
        return f"{Path(inspect.getsourcefile(fn) or '?').name}"
    except Exception:  # noqa: BLE001
        return "?"


def _tokenizer_like(v: Any) -> bool:
    return (not isinstance(v, type) and hasattr(v, "encode") and hasattr(v, "vocab_size")
            and hasattr(v, "decode"))


_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")


def canonical(value: Any, *, warn: bool = True) -> Any:
    """JSON-able, stable view of ``value`` (see module docstring)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): canonical(v, warn=warn) for k, v in sorted(value.items(),
                                                                     key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, range)):
        return [canonical(v, warn=warn) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((canonical(v, warn=warn) for v in value), key=repr)
    if _tokenizer_like(value):
        to_dict = getattr(value, "to_dict", None)
        try:
            blob = json.dumps(to_dict(), sort_keys=True) if callable(to_dict) else repr(value)
        except Exception:  # noqa: BLE001
            blob = repr(value)
        return {"tokenizer": str(getattr(value, "kind", type(value).__name__)),
                "sha256": hashlib.sha256(blob.encode()).hexdigest()[:16]}
    try:
        import torch

        if isinstance(value, torch.Tensor):
            t = value.detach().cpu()
            return {"tensor": list(t.shape), "dtype": str(t.dtype),
                    "sha256": hashlib.sha256(repr(t.flatten()[:4096].tolist()).encode())
                    .hexdigest()[:16]}
    except Exception:  # noqa: BLE001, S110 - fingerprinting is best-effort; falls back to repr
        pass
    if isinstance(value, functools.partial):
        return {"partial": canonical(value.func, warn=warn),
                "args": canonical(list(value.args), warn=warn),
                "keywords": canonical(dict(value.keywords), warn=warn)}
    from topologies.experiments.space import Distribution, Fixed, Variant

    if isinstance(value, Fixed):
        return canonical(value.value, warn=warn)
    if isinstance(value, Variant):
        return {"variant": value.label, "value": canonical(value.value, warn=warn),
                **{k: canonical(v, warn=warn) for k, v in sorted(value.overrides.items())}}
    if isinstance(value, Distribution):
        return {"distribution": repr(value)}
    to_config = getattr(value, "to_config", None)
    if callable(to_config) and not isinstance(value, type):
        try:
            return {"__type__": f"{type(value).__module__}:{type(value).__qualname__}",
                    **canonical(to_config(), warn=warn)}
        except Exception:  # noqa: BLE001, S110 - fall back to the generic rules below
            pass
    if isinstance(value, type) or callable(value):
        name = import_name(value)
        if name is not None:
            return name
        label = getattr(value, "__name__", type(value).__name__)
        loc = _source_location(value)
        if warn:
            warn_once(f"{label!s} is not importable (lambda/closure); its trial ID depends on "
                      f"its source location ({loc}). Use a module-level function for stable "
                      "IDs and for workers > 1.")
        return f"<callable {label}@{loc}>"
    if hasattr(value, "value") and type(value).__module__ == "enum":
        return canonical(value.value, warn=warn)
    text = _ADDR.sub("", repr(value))
    if len(text) > 120:
        text = text[:117] + "..."
    if warn:
        warn_once(f"Using repr() for {type(value).__name__} in the trial ID: {text!r}. "
                  "Prefer plain values, classes or import paths.")
    return text


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def trial_hash(task_fingerprint: str | None, config: dict, seed: int) -> str:
    """``trial_id``: first 12 hex chars of sha256 over task, canonical config and seed."""
    blob = canonical_json({"task": task_fingerprint, "config": config, "seed": int(seed)})
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- spec files


def _load_python_spec(path: Path, attr: str | None) -> Any:
    from topologies.experiments.core import Experiment

    mod_name = f"_topologies_spec_{hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ExperimentError(f"Cannot import {path}.", stage="spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    if str(path.parent.resolve()) not in sys.path:
        sys.path.insert(0, str(path.parent.resolve()))
    spec.loader.exec_module(module)
    if attr:
        obj = getattr(module, attr, None)
        if obj is None:
            raise ExperimentError(f"{path} has no attribute {attr!r}.", stage="spec")
        if callable(obj) and not isinstance(obj, Experiment):
            obj = obj()
        if not isinstance(obj, Experiment):
            raise ExperimentError(f"{path}:{attr} is not an Experiment.", stage="spec")
        obj._spec_path = f"{path}:{attr}"
        return obj
    found = [(k, v) for k, v in vars(module).items() if isinstance(v, Experiment)]
    if not found:
        raise ExperimentError(f"{path} defines no Experiment.", stage="spec",
                              hint="Assign one at module level: exp = Experiment(...).")
    if len(found) > 1:
        raise ExperimentError(
            f"{path} defines {len(found)} experiments: {', '.join(k for k, _ in found)}.",
            stage="spec", hint=f"Pick one: {path}:{found[0][0]}")
    found[0][1]._spec_path = str(path)
    return found[0][1]


def load_spec(target: str | Path) -> Any:
    """``spec.py`` / ``spec.py:attr`` / ``spec.yaml`` -> ``Experiment``."""
    text = str(target)
    attr = None
    if ":" in text and not Path(text).exists():
        text, _, attr = text.rpartition(":")
    path = Path(text)
    if not path.is_file():
        raise ExperimentError(f"No such spec file: {path}", stage="spec")
    if path.suffix == ".py":
        return _load_python_spec(path, attr)
    if path.suffix in (".yaml", ".yml", ".json"):
        from topologies.experiments.yaml_spec import experiment_from_yaml

        exp = experiment_from_yaml(path)
        exp._spec_path = str(path)
        return exp
    raise ExperimentError(f"Unsupported spec file {path} (use .py, .yaml or .json).",
                          stage="spec")


__all__ = ["ExperimentWarning", "canonical", "canonical_json", "import_name", "load_spec",
           "trial_hash", "warn_once"]
