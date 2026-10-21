"""Tiny pytree helpers for tuple/list/dict/dataclass/namedtuple model inputs and outputs."""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import torch


def tree_map(fn: Callable[[torch.Tensor], Any], value: Any) -> Any:
    """Apply ``fn`` to every tensor leaf, rebuilding containers where possible."""
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(tree_map(fn, v) for v in value))
    if isinstance(value, (list, tuple)):
        return type(value)(tree_map(fn, v) for v in value)
    if isinstance(value, Mapping):
        items = {k: tree_map(fn, v) for k, v in value.items()}
        try:
            return type(value)(items)
        except Exception:  # noqa: BLE001 - exotic mappings fall back to dict
            return items
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return dataclasses.replace(value, **{f.name: tree_map(fn, getattr(value, f.name))
                                                 for f in dataclasses.fields(value) if f.init})
        except Exception:  # noqa: BLE001
            return value
    return value


def leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(path, tensor)`` in pytree order; paths look like ``0``, ``logits``, ``0.hidden``."""
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, tuple) and hasattr(value, "_fields"):
        for name, v in zip(value._fields, value):
            yield from leaves(v, f"{prefix}.{name}" if prefix else name)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from leaves(v, f"{prefix}.{i}" if prefix else str(i))
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            yield from leaves(getattr(value, f.name), f"{prefix}.{f.name}" if prefix else f.name)


def first_tensor(value: Any) -> tuple[str, torch.Tensor] | None:
    return next(leaves(value), None)


def get_path(value: Any, path: str) -> Any:
    """Follow a dotted pytree path (ints index sequences, names index mappings/attributes)."""
    for part in [p for p in path.split(".") if p]:
        if isinstance(value, Mapping):
            value = value[part] if part in value else value[int(part)]
        elif isinstance(value, (list, tuple)) and part.lstrip("-").isdigit():
            value = value[int(part)]
        else:
            value = getattr(value, part)
    return value


def set_path(value: Any, path: str, new: Any) -> Any:
    """Return ``value`` with the leaf at ``path`` replaced (containers are rebuilt, not mutated)."""
    parts = [p for p in path.split(".") if p]
    if not parts:
        return new
    head, rest = parts[0], ".".join(parts[1:])
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        index = value._fields.index(head) if head in value._fields else int(head)
        items = list(value)
        items[index] = set_path(items[index], rest, new)
        return type(value)(*items)
    if isinstance(value, (list, tuple)):
        items = list(value)
        items[int(head)] = set_path(items[int(head)], rest, new)
        return type(value)(items)
    if isinstance(value, Mapping):
        items = dict(value)
        key = head if head in items else int(head)
        items[key] = set_path(items[key], rest, new)
        try:
            return type(value)(items)
        except Exception:  # noqa: BLE001
            return items
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, **{head: set_path(getattr(value, head), rest, new)})
    raise TypeError(f"cannot replace {path!r} inside {type(value).__name__}")


def primary_path(value: Any) -> str | None:
    found = first_tensor(value)
    return None if found is None else found[0]


def describe(value: Any) -> Any:
    """Shape/dtype skeleton of a pytree (for call-tree records)."""
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        return [describe(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): describe(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: describe(getattr(value, f.name)) for f in dataclasses.fields(value)}
    return type(value).__name__
