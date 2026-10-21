"""Parameter counting and human-friendly parameter budgets ("15k", "2.5M")."""

from __future__ import annotations

import re
from collections.abc import Callable

import torch.nn as nn

_SUFFIX = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "g": 1_000_000_000}
_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([kKmMbBgG]?)\s*$")


def parse_param_count(value: float | str) -> int:
    """``"15k"`` -> 15000, ``"2.5M"`` -> 2500000, ``120_000`` -> 120000."""
    if isinstance(value, bool):
        raise TypeError(f"Invalid parameter count {value!r}.")
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError(f"Parameter count must be positive; got {value!r}.")
        return int(value)
    m = _RE.match(str(value).replace("_", ""))
    if not m:
        raise ValueError(
            f"Invalid parameter count {value!r}; use e.g. 15000, '15k', '2.5M', '1B'."
        )
    n = int(float(m.group(1)) * _SUFFIX[m.group(2).lower()])
    if n <= 0:
        raise ValueError(f"Parameter count must be positive; got {value!r}.")
    return n


def count_params(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Number of (unique) parameters, safe on ``meta`` tensors and shared weights."""
    seen: set[int] = set()
    total = 0
    for p in module.parameters():
        if id(p) in seen or (trainable_only and not p.requires_grad):
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def embedding_params(module: nn.Module) -> int:
    """Parameters owned by ``nn.Embedding`` modules (tied output heads included once)."""
    seen: set[int] = set()
    total = 0
    for m in module.modules():
        if isinstance(m, nn.Embedding):
            for p in m.parameters(recurse=False):
                if id(p) not in seen:
                    seen.add(id(p))
                    total += p.numel()
    return total


PARAM_COUNTS = ("total", "trainable", "non_embedding")


def count_params_as(module: nn.Module, count: str | Callable[[nn.Module], int] = "total") -> int:
    """Parameter count under a counting rule.

    ``"total"`` (unique parameters), ``"trainable"`` (``requires_grad``),
    ``"non_embedding"`` (excludes ``nn.Embedding`` weights and output heads tied
    to them: a tied head *is* the embedding parameter), or a callable
    ``fn(module) -> int``.
    """
    if callable(count) and not isinstance(count, str):
        return int(count(module))
    if count == "total":
        return count_params(module)
    if count == "trainable":
        return count_params(module, trainable_only=True)
    if count == "non_embedding":
        return count_params(module) - embedding_params(module)
    raise ValueError(f"Unknown parameter count {count!r}; use one of {PARAM_COUNTS} or a "
                     "callable fn(module) -> int.")


def format_params(n: int | None) -> str:
    """15000 -> '15.0K', 2_500_000 -> '2.50M'."""
    if n is None:
        return "?"
    n = int(n)
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)
