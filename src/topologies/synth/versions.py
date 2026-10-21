"""Derived versions: edits, drops, relabels, merges and mixes create new immutable versions.

Each derived version records ``parent`` (the version it came from), the
operation, a note and a diff summary in its manifest; the parent is never
modified. ``diff(a, b)`` compares any two versions.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from topologies.experiments.cache import stable_hash
from topologies.experiments.store import now_iso, write_json_atomic
from topologies.synth.errors import SynthError
from topologies.synth.records import as_records, strip_meta
from topologies.synth.store import ShardWriter, SynthStore, dumps, iter_shards


def _content_hash(rows: dict[str, list[dict]]) -> str:
    h = hashlib.sha256()
    for s in sorted(rows):
        h.update(s.encode())
        for r in rows[s]:
            h.update(dumps(r).encode())
    return h.hexdigest()[:16]


def write_derived(store: SynthStore, rows: dict[str, list[dict]], *, parent: Any,
                  op: str, note: str, extra: Mapping[str, Any] | None = None,
                  lineage: Mapping[str, dict] | None = None,
                  rejected: list[dict] | None = None, label: str | None = None) -> Any:
    """Write ``rows`` (per split) as a new version derived from ``parent``."""
    from topologies.synth.dataset import SynthDataset

    parent_id = parent.version_id if parent is not None else None
    vid = stable_hash({"parent": parent_id, "op": op, "content": _content_hash(rows)}, n=12)
    if store.has_version(vid):
        store.set_latest(vid)
        return SynthDataset(store, vid)
    base = dict(parent.manifest) if parent is not None else {}
    with store.new_version(vid) as tmp:
        files = {}
        for s, recs in rows.items():
            w = ShardWriter(tmp / "records" / s)
            for r in recs:
                w.add(r)
            files[s] = w.close()
        lw = ShardWriter(tmp / "lineage")
        for recs in rows.values():
            for r in recs:
                lin = (lineage or {}).get(r.get("_id")) or {"_id": r.get("_id")}
                lw.add(lin)
        rw = ShardWriter(tmp / "rejected")
        for r in rejected or []:
            rw.add(r)
        manifest = {**base, "version_id": vid, "created_at": now_iso(), "parent": parent_id,
                    "derived": {"op": op, "note": note, **dict(extra or {})},
                    "label": label, "files": files, "lineage_files": lw.close(),
                    "rejected_files": rw.close(),
                    "splits": {s: len(v) for s, v in rows.items()},
                    "total": sum(len(v) for v in rows.values())}
        if parent is not None:
            manifest["diff"] = _diff_counts(parent, rows)
        write_json_atomic(tmp / "manifest.json", manifest)
    store.set_latest(vid)
    store.log_event("version", version_id=vid, parent=parent_id, op=op, note=note)
    return SynthDataset(store, vid)


def _diff_counts(parent: Any, rows: dict[str, list[dict]]) -> dict[str, int]:
    old = {r["_id"]: strip_meta(r) for r in parent.view()}
    new = {r.get("_id"): strip_meta(r) for v in rows.values() for r in v}
    added = sum(1 for k in new if k not in old)
    removed = sum(1 for k in old if k not in new)
    changed = sum(1 for k, v in new.items() if k in old and old[k] != v)
    return {"added": added, "removed": removed, "changed": changed}


def _lineage(ds: Any) -> dict[str, dict]:
    return {d["_id"]: d for d in iter_shards(ds.dir / "lineage")}


def derive(ds: Any, fn: Callable[[dict], dict | None], *, op: str, note: str = "") -> Any:
    """New version with ``fn`` applied to every record (``None`` drops it)."""
    rows: dict[str, list[dict]] = {}
    dropped = []
    for s in ds.splits:
        out = []
        for r in ds.view(s):
            new = fn(dict(r))
            if new is None:
                dropped.append({**r, "_op": op, "_verdict": {"ok": False, "reason": note}})
                continue
            new = dict(new)
            new.setdefault("_id", r["_id"])
            new.setdefault("_attrs", r.get("_attrs", {}))
            out.append(new)
        rows[s] = out
    rejected = list(iter_shards(ds.dir / "rejected")) + dropped
    return write_derived(ds.store, rows, parent=ds, op=op, note=note, lineage=_lineage(ds),
                         rejected=rejected)


def relabel(ds: Any, op: Any, *, field: str | None = None) -> Any:
    """Apply a relabel op (``synth.ops.label.*`` / map op / ``fn(record) -> label``)."""
    from topologies.synth.ops.base import MapOp
    from topologies.synth.pipeline import run_ops_on_records

    f = field or (ds.schema.label_field() if ds.schema is not None else None) or "label"
    if isinstance(op, MapOp) or hasattr(op, "kind"):
        records = {s: list(ds.view(s)) for s in ds.splits}
        new = {s: run_ops_on_records(recs, [op], name=f"relabel-{ds.name}") for s, recs in
               records.items()}
        changed = {r["_id"]: r for v in new.values() for r in v}
        return derive(ds, lambda r: changed.get(r["_id"], r), op="relabel",
                      note=getattr(op, "name", "relabel"))
    if callable(op):
        def apply(r: dict) -> dict:
            out = dict(r)
            out[f] = op(r)
            return out

        return derive(ds, apply, op="relabel", note=getattr(op, "__name__", "fn"))
    raise SynthError(f"Cannot relabel with {op!r}.", stage="versions",
                     hint="Pass a label op (synth.ops.label.program(...)) or fn(record) -> label.")


def merge(ds: Any, other: Any, *, dedup: bool = True, name: str | None = None) -> Any:
    """Union of two versions/datasets (``other`` may be records); exact dedup by content."""
    rows: dict[str, list[dict]] = {s: list(ds.view(s)) for s in ds.splits}
    if hasattr(other, "splits") and hasattr(other, "view"):
        extra = {s: list(other.view(s)) for s in other.splits}
    else:
        extra = {"train": [{"_id": _rid(r), **r} for r in as_records(other)]}
    seen = {dumps(strip_meta(r)) for v in rows.values() for r in v} if dedup else set()
    added = 0
    for s, recs in extra.items():
        for r in recs:
            key = dumps(strip_meta(r))
            if dedup and key in seen:
                continue
            seen.add(key)
            rows.setdefault(s, []).append(r)
            added += 1
    store = ds.store if name is None else SynthStore(ds.store.root, name)
    lin = _lineage(ds)
    if hasattr(other, "dir"):
        lin.update(_lineage(other))
    return write_derived(store, rows, parent=ds if name is None else None, op="merge",
                         note=f"+{added} records", lineage=lin)


def _rid(r: Mapping[str, Any]) -> str:
    return r.get("_id") or hashlib.blake2b(dumps(strip_meta(r)).encode(),
                                           digest_size=8).hexdigest()


def mix(parts: Mapping[Any, float], *, size: int, name: str, seed: int = 0,
        root: str | Path | None = None, split: str | None = "train") -> Any:
    """Weighted mixture of datasets / record lists into a new dataset ``name``."""
    if not parts:
        raise SynthError("mix() needs at least one part.", stage="versions")
    total_w = sum(float(w) for w in parts.values())
    if total_w <= 0:
        raise SynthError("mix() weights sum to 0.", stage="versions")
    rng = random.Random(seed)
    out: list[dict] = []
    sources = []
    for src, w in parts.items():
        if hasattr(src, "view"):
            recs = list(src.view(split if split in getattr(src, "splits", {}) else None))
            label = getattr(src, "name", "dataset")
        else:
            recs = [{"_id": _rid(r), **r} for r in as_records(src)]
            label = "records"
        k = round(size * float(w) / total_w)
        if not recs:
            continue
        picked = [rng.choice(recs) for _ in range(k)] if k > len(recs) else rng.sample(recs, k)
        for r in picked:
            r = dict(r)
            r.setdefault("_attrs", {})
            r["_attrs"] = {**r["_attrs"], "_source": label}
            out.append(r)
        sources.append({"source": label, "weight": float(w) / total_w, "n": len(picked)})
    rng.shuffle(out)
    store = SynthStore(root, name)
    return write_derived(store, {"train": out}, parent=None, op="mix", note="mix",
                         extra={"sources": sources})


def diff(a: Any, b: Any) -> dict[str, Any]:
    """Added / removed / changed records and distribution shifts between two versions."""
    ra = {r["_id"]: r for r in a.view()}
    rb = {r["_id"]: r for r in b.view()}
    added = [k for k in rb if k not in ra]
    removed = [k for k in ra if k not in rb]
    changed = [k for k in rb if k in ra and strip_meta(ra[k]) != strip_meta(rb[k])]
    out: dict[str, Any] = {"a": a.version_id, "b": b.version_id, "added": len(added),
                           "removed": len(removed), "changed": len(changed),
                           "examples": {"added": added[:5], "removed": removed[:5],
                                        "changed": changed[:5]}}
    lf = (b.schema.label_field() if b.schema is not None else None)
    if lf:
        ca = Counter(str(r.get(lf)) for r in ra.values())
        cb = Counter(str(r.get(lf)) for r in rb.values())
        out["labels"] = {k: {"a": ca.get(k, 0), "b": cb.get(k, 0)}
                         for k in sorted(set(ca) | set(cb))}
    out["splits"] = {"a": a.splits, "b": b.splits}
    return out


__all__ = ["derive", "diff", "merge", "mix", "relabel", "write_derived"]
