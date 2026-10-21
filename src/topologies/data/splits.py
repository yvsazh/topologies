"""Subset / split helpers (reproducible, optionally stratified)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import Subset

_LABEL_FIELDS = ("labels", "label", "targets", "target", "y")


def _resolve_count(n: float, total: int, what: str) -> int:
    if isinstance(n, float) and n < 1.0:
        if n <= 0.0:
            raise ValueError(f"{what} fraction must be in (0, 1); got {n}.")
        return max(1, round(total * n))
    if isinstance(n, float) and not n.is_integer():
        raise ValueError(f"{what} must be an integer count or a fraction in (0, 1); got {n}.")
    k = int(n)
    if k < 1:
        raise ValueError(f"{what} must be >= 1; got {n}.")
    return min(k, total)


def dataset_labels(dataset: Any) -> list[int]:
    """Best-effort integer labels of every sample (for stratification).

    Uses ``targets``/``labels`` attributes when present (torchvision datasets,
    our wrappers forward them), maps through ``Subset`` indices, and only
    falls back to reading every sample.
    """
    if isinstance(dataset, Subset):
        parent = dataset_labels(dataset.dataset)
        return [parent[i] for i in dataset.indices]
    for attr in ("targets", "labels"):
        try:
            vals = getattr(dataset, attr)
        except Exception:  # noqa: BLE001, S112 - dataset label properties may raise; try the next one
            continue
        if vals is None or callable(vals):
            continue
        if isinstance(vals, torch.Tensor):
            return [int(v) for v in vals.tolist()]
        try:
            if len(vals) == len(dataset):
                return [int(v) for v in vals]
        except (TypeError, ValueError):  # not a sized sequence of integer labels
            continue
    out: list[int] = []
    for i in range(len(dataset)):
        s = dataset[i]
        if isinstance(s, dict):
            key = next((k for k in _LABEL_FIELDS if k in s), None)
            if key is None:
                raise ValueError(
                    f"Cannot find a label field in sample keys {sorted(s)}; pass labels=[...]."
                )
            out.append(int(s[key]))
        elif isinstance(s, (tuple, list)) and len(s) >= 2:
            out.append(int(s[1]))
        else:
            raise ValueError("Cannot infer labels for stratification; pass labels=[...].")
    return out


def _stratified_indices(labels: list[int], k: int, g: torch.Generator) -> list[int]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[int(y)].append(i)
    total = len(labels)
    picked: list[int] = []
    remainders: list[tuple[float, int]] = []
    for y, idx in sorted(by_class.items()):
        perm = torch.randperm(len(idx), generator=g).tolist()
        exact = k * len(idx) / total
        take = int(exact)
        picked.extend(idx[j] for j in perm[:take])
        remainders.append((exact - take, y))
        by_class[y] = [idx[j] for j in perm[take:]]
    # Distribute the rounding remainder to the classes with the largest fraction.
    for _, y in sorted(remainders, reverse=True):
        if len(picked) >= k:
            break
        if by_class[y]:
            picked.append(by_class[y].pop(0))
    order = torch.randperm(len(picked), generator=g).tolist()
    return [picked[i] for i in order]


def _pick(dataset: Any, idx: list[int]) -> Any:
    """Lists/tuples (e.g. text documents) stay lists; datasets become Subsets."""
    if isinstance(dataset, (list, tuple)):
        return [dataset[i] for i in idx]
    return Subset(dataset, idx)


def take_subset(
    dataset: Any,
    n: float,
    *,
    seed: int | None = 0,
    stratify: bool = False,
    labels: list[int] | None = None,
) -> Any:
    """Return ``n`` samples (or a fraction ``0 < n < 1``) of ``dataset``.

    ``seed=None`` takes the first ``n`` samples in order; otherwise samples are
    drawn reproducibly at random. ``stratify=True`` keeps class proportions.
    """
    total = len(dataset)
    k = _resolve_count(n, total, "n")
    if seed is None and not stratify:
        return _pick(dataset, list(range(k)))
    g = torch.Generator().manual_seed(0 if seed is None else int(seed))
    if stratify:
        idx = _stratified_indices(labels if labels is not None else dataset_labels(dataset), k, g)
    else:
        idx = torch.randperm(total, generator=g)[:k].tolist()
    return _pick(dataset, idx)


def train_val_split(
    dataset: Any,
    val: float = 0.1,
    *,
    seed: int | None = 0,
    stratify: bool = False,
    labels: list[int] | None = None,
) -> tuple[Any, Any]:
    """Split into ``(train, val)``; ``val`` is a count (``200``) or fraction (``0.1``).

    Works on datasets (returns ``Subset``s) and on plain lists such as text
    documents (returns lists). Random but reproducible (``seed``);
    ``seed=None`` keeps order and uses the last ``val`` items for validation.
    ``stratify=True`` keeps class proportions.
    """
    total = len(dataset)
    if total < 2:
        raise ValueError("Need at least 2 samples to split.")
    k = _resolve_count(val, total, "val")
    if k >= total:
        raise ValueError(f"val={val} leaves no training samples (dataset has {total}).")
    if seed is None and not stratify:
        head, tail = list(range(total - k)), list(range(total - k, total))
        return _pick(dataset, head), _pick(dataset, tail)
    g = torch.Generator().manual_seed(0 if seed is None else int(seed))
    if stratify:
        val_idx = _stratified_indices(
            labels if labels is not None else dataset_labels(dataset), k, g
        )
        chosen = set(val_idx)
        train_idx = [i for i in torch.randperm(total, generator=g).tolist() if i not in chosen]
    else:
        perm = torch.randperm(total, generator=g).tolist()
        val_idx, train_idx = perm[:k], perm[k:]
    return _pick(dataset, train_idx), _pick(dataset, val_idx)
