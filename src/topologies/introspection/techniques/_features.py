"""Shared helpers: pooled per-example features (and labels) streamed from data."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch

from topologies.introspection.errors import SemanticsRequired
from topologies.introspection.inputs import Batch, is_collection, normalize_inputs
from topologies.introspection.reducers import pool_features
from topologies.introspection.semantics import SemanticSource

POOLS = ("mean", "last", "cls", "first", "flatten")


def pooled(value: torch.Tensor, axes: Any, pool: str, site: str) -> tuple[torch.Tensor, str]:
    """``[B, F]`` features and a description of how they were pooled."""
    if value.ndim == 2:
        return value.float(), "none (already [batch, feature])"
    if pool == "flatten":
        return value.reshape(value.shape[0], -1).float(), "flatten"
    known = axes is not None and axes.source != SemanticSource.UNKNOWN and \
        axes.feature_axis is not None and axes.batch_axis is not None
    if not known:
        if pool == "mean" and value.ndim > 2:
            # Unknown layout: average over every non-batch axis except the last.
            raise SemanticsRequired(
                f"pooling {site!r} needs batch and feature axes; declare them (e.g. "
                f"insp.declare({site!r}, axes='b t d')) or pass pool='flatten'")
        raise SemanticsRequired(f"pool={pool!r} at {site!r} needs declared axes")
    return pool_features(value.float(), axes, pool, site=site), f"{pool} over non-feature axes"


def batches(subject: Any, data: Any, max_batches: int | None = None) -> Iterator[Batch]:
    source = data if is_collection(data) else [data]
    for i, batch in enumerate(normalize_inputs(subject, source, from_data=True)):
        if max_batches is not None and i >= max_batches:
            return
        yield batch


def batch_labels(batch: Batch, labels: Any, offset: int, size: int) -> torch.Tensor | None:
    if labels is None:
        target = batch.meta.get("target")
        if target is None:
            targets = batch.meta.get("targets") or {}
            target = next(iter(targets.values()), None)
        return None if target is None else torch.as_tensor(target).reshape(size, -1).squeeze(-1)
    if isinstance(labels, torch.Tensor):
        return labels[offset:offset + size]
    if isinstance(labels, (list, tuple)):
        return torch.as_tensor(labels[offset:offset + size])
    if isinstance(labels, str):
        if batch.batch is not None and labels in batch.batch:
            return torch.as_tensor(batch.batch[labels])
        targets = batch.meta.get("targets") or {}
        if labels in targets:
            return torch.as_tensor(targets[labels])
        raise KeyError(f"labels field {labels!r} not in the batch")
    if callable(labels):
        if batch.call is not None and batch.call.args:
            return torch.as_tensor(labels(batch.call.args[0]))
        return torch.as_tensor(labels(batch.batch))
    raise TypeError("labels must be a tensor, a sequence, a batch field name or a callable")


def collect_features(subject: Any, data: Any, sites: list[str], *, pool: str = "mean",
                     labels: Any = None, max_batches: int | None = None,
                     max_examples: int | None = None) -> tuple[dict[str, torch.Tensor],
                                                               torch.Tensor | None, list[str]]:
    """Pooled ``[N, F]`` features per site (+ labels) streamed batch by batch."""
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}")
    features: dict[str, list[torch.Tensor]] = {s: [] for s in sites}
    label_chunks: list[torch.Tensor] = []
    notes: list[str] = []
    offset = 0
    selector = ",".join(s or "<root>" for s in sites)
    for batch in batches(subject, data, max_batches):
        trace = subject.inspector.trace(batch, sites=selector, keep="tensors", call_tree=False,
                                        max_bytes=None)
        size = None
        for site in sites:
            records = [r for r in trace.records.values() if r.key.path == site and r.key.op is None
                       and r.port == "output"]
            if not records:
                raise SemanticsRequired(f"site {site!r} did not run for these inputs")
            record = max(records, key=lambda r: r.occurrence)
            feats, how = pooled(record.value, record.axes, pool, site)
            features[site].append(feats.cpu())
            size = feats.shape[0]
            note = f"{site}: pooling = {how}"
            if note not in notes:
                notes.append(note)
        if labels is not None or batch.meta.get("target") is not None or batch.meta.get("targets"):
            got = batch_labels(batch, labels, offset, size or 0)
            if got is not None:
                label_chunks.append(got.cpu())
        offset += size or 0
        if max_examples is not None and offset >= max_examples:
            break
    out = {s: torch.cat(v) for s, v in features.items() if v}
    if max_examples is not None:
        out = {s: v[:max_examples] for s, v in out.items()}
    ys = torch.cat(label_chunks) if label_chunks else None
    if ys is not None and max_examples is not None:
        ys = ys[:max_examples]
    return out, ys, notes
