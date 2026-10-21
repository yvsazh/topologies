"""Load executable training components materialized from pinned artifacts.

Catalog records are intentionally descriptive, but executable implementations
are Python files downloaded and checksum-verified by ``materialize_artifacts``.
Each record declares ``implementation.path`` relative to ``topologies`` and a
``symbol``. Loss symbols are callables; optimizer symbols are factories using
the normal ``register_optimizer`` contract.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

_REGISTERED_ARTIFACTS: dict[tuple[str, str], tuple[str, Any]] = {}


def _module_for(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load training implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_training_components(package_root: str | Path) -> dict[str, list[str]]:
    """Register all executable catalog records present in a materialized tree.

    Missing catalogs and definitions without an implementation are ignored to
    retain compatibility with older metadata-only catalogs. Declared but
    missing or invalid implementations fail loudly rather than silently using
    a similarly named ``torch`` class.
    """
    root = Path(package_root).resolve()
    loaded: dict[str, list[str]] = {"loss": [], "optimizer": []}
    for kind, plural in (("loss", "losses"), ("optimizer", "optimizers")):
        catalog_path = root / plural / "catalog.json"
        if not catalog_path.is_file():
            continue
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if not isinstance(catalog, dict) or not isinstance(catalog.get("entries"), list):
            raise TypeError(f"invalid {kind} catalog schema: expected entries array")
        seen: set[str] = set()
        pending: list[tuple[str, str, str, Any]] = []
        for entry in catalog["entries"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"].strip():
                raise ValueError(f"invalid {kind} catalog entry: expected non-empty id")
            if entry["id"] in seen:
                raise ValueError(f"duplicate {kind} catalog id {entry['id']!r}")
            seen.add(entry["id"])
            declaration = entry.get("implementation")
            if not isinstance(declaration, dict):
                continue
            rel = Path(str(declaration.get("path", "")))
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe implementation path for {entry.get('id')!r}")
            source = (root / rel).resolve()
            if root not in source.parents or not source.is_file():
                raise FileNotFoundError(f"missing implementation for {entry.get('id')!r}: {rel}")
            expected = declaration.get("sha256")
            if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected):
                raise ValueError(f"missing implementation sha256 for {entry.get('id')!r}")
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            if actual != expected.lower():
                raise ValueError(f"implementation digest mismatch for {entry.get('id')!r}")
            from topologies.validation.source_style import comment_and_docstring_violations

            source_text = source.read_text(encoding="utf-8")
            violations = comment_and_docstring_violations(source_text, str(rel))
            if violations:
                raise ValueError(f"implementation source violates comment/docstring policy for "
                                 f"{entry.get('id')!r}: {violations}")
            symbol = declaration.get("symbol")
            artifact_key = (kind, entry["id"].lower())
            prior = _REGISTERED_ARTIFACTS.get(artifact_key)
            if prior and prior[0] == actual:
                pending.append((entry["id"], actual, "cached", prior[1]))
                continue
            module = _module_for(source, f"topologies_artifact_{kind}_{entry['id'].replace('-', '_')}_{actual[:12]}")
            component: Any = getattr(module, symbol, None)
            if not callable(component):
                raise TypeError(f"implementation symbol {symbol!r} is not callable")
            pending.append((entry["id"], actual, "new", component))
        # All source metadata and symbols validate before any registry changes.
        for entry_id, digest, state, component in pending:
            artifact_key = (kind, entry_id.lower())
            if state == "new":
                prior = _REGISTERED_ARTIFACTS.get(artifact_key)
                if kind == "loss":
                    from topologies.training.losses import _LOSSES, register_loss
                    current, register = _LOSSES.get(entry_id.lower()), register_loss
                else:
                    from topologies.training.registries import _OPTIMIZERS, register_optimizer
                    current, register = _OPTIMIZERS.get(entry_id.lower()), register_optimizer
                if prior is None or current is prior[1]:
                    register(entry_id, component)
                    _REGISTERED_ARTIFACTS[artifact_key] = (digest, component)
            loaded[kind].append(entry_id)
    return loaded


__all__ = ["load_training_components"]
