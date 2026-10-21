"""``SynthDataset``: a versioned, lineage-tracked synthetic dataset on disk.

    ds = synth.load("support-intents")            # latest version
    len(ds); ds.splits; ds.schema; ds.manifest; ds.gen_seconds
    ds.view("train")                              # lazy record view (list-like)
    ds.sample(10, by="label"); ds.head(); ds.show(5)
    ds.filter(label="refund", where=lambda r: len(r["text"]) > 20)
    ds.explain(record_id)                         # attributes, latents, verdicts
    ds.rejected(op="dedup.near").sample(10)
    train = ds.for_recipe("text-classify", split="train", tokenizer=tok, max_len=128)
    ds.export("out/", format="jsonl")

Edits (``relabel``, ``drop``, ``edit``, ``merge``) create new immutable
versions with ``parent`` set; the original version never changes.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from topologies.synth.errors import SynthError, unknown
from topologies.synth.records import strip_meta
from topologies.synth.store import SynthStore, iter_shards


class RecordView:
    """List-like view over records (lazy: shards are read on first indexing)."""

    def __init__(self, source: Callable[[], Iterable[dict]] | Sequence[dict], *,
                 length: int | None = None, name: str = "records", dataset: Any = None) -> None:
        if callable(source):
            self._source: Callable[[], Iterable[dict]] | None = source
            self._items: list[dict] | None = None
        else:
            self._source = None
            self._items = list(source)
        self._length = length
        self.name = name
        self.dataset = dataset

    def _load(self) -> list[dict]:
        if self._items is None:
            assert self._source is not None
            self._items = list(self._source())
        return self._items

    def __len__(self) -> int:
        if self._items is None and self._length is not None:
            return self._length
        return len(self._load())

    def __iter__(self) -> Iterator[dict]:
        if self._items is None and self._source is not None:
            return iter(self._source())
        return iter(self._load())

    def __getitem__(self, i: Any) -> Any:
        items = self._load()
        if isinstance(i, slice):
            return RecordView(items[i], name=self.name, dataset=self.dataset)
        return items[i]

    def __repr__(self) -> str:
        return f"RecordView({self.name}, {len(self)} records)"

    def decoded(self) -> list[dict]:
        """Records with live values (blobs decoded; needs the dataset)."""
        if self.dataset is None:
            from topologies.synth.valuetypes import from_jsonable

            return [{k: from_jsonable(v) for k, v in r.items()} for r in self._load()]
        return [self.dataset.decode(r) for r in self._load()]

    def to_records(self, *, meta: bool = True) -> list[dict]:
        return [dict(r) if meta else strip_meta(r) for r in self]

    # -- selection -------------------------------------------------------------
    def head(self, n: int = 5) -> RecordView:
        out = []
        for r in self:
            if len(out) >= n:
                break
            out.append(r)
        return RecordView(out, name=f"{self.name}[:{n}]", dataset=self.dataset)

    def sample(self, n: int = 10, *, by: str | None = None, seed: int = 0) -> RecordView:
        """``n`` random records; ``by="label"`` takes up to ``n`` per value."""
        items = self._load()
        rng = random.Random(seed)
        if by is None:
            pick = rng.sample(items, min(n, len(items)))
        else:
            groups: dict[str, list[dict]] = defaultdict(list)
            for r in items:
                groups[_val(r, by)].append(r)
            pick = []
            for k in sorted(groups):
                g = groups[k]
                pick += rng.sample(g, min(n, len(g)))
        return RecordView(pick, name=f"sample({self.name})", dataset=self.dataset)

    def filter(self, where: Callable[[dict], bool] | None = None, **equals: Any) -> RecordView:
        """Records matching ``field=value`` (fields or attributes) and ``where(record)``."""
        out = []
        for r in self:
            if any(_val(r, k, raw=True) != v for k, v in equals.items()):
                continue
            if where is not None and not where(r):
                continue
            out.append(r)
        desc = ", ".join(f"{k}={v!r}" for k, v in equals.items())
        return RecordView(out, name=f"{self.name}.filter({desc})", dataset=self.dataset)

    def values(self, key: str) -> list[Any]:
        return [_val(r, key, raw=True) for r in self]

    def counts(self, key: str) -> Counter:
        return Counter(_val(r, key) for r in self)

    def show(self, n: int = 5, *, fields: Sequence[str] | None = None) -> None:
        from topologies.synth.report import format_records

        print(format_records(list(self.head(n)), fields=fields))

    # -- torch ------------------------------------------------------------------------
    def for_recipe(self, recipe: str, **kw: Any) -> Any:
        """Torch dataset in ``recipe``'s batch format (see ``synth.torch_views``)."""
        from topologies.synth.torch_views import build_dataset

        schema = self.dataset.schema if self.dataset is not None else None
        if self.dataset is not None:
            kw.setdefault("blob_loader", self.dataset.load_blob)
            kw.setdefault("decode", self.dataset.decode)
        return build_dataset(list(self), recipe, schema=schema, **kw)

    def to_torch(self, **kw: Any) -> Any:
        from topologies.synth.torch_views import build_dataset

        schema = self.dataset.schema if self.dataset is not None else None
        if self.dataset is not None:
            kw.setdefault("decode", self.dataset.decode)
        return build_dataset(list(self), None, schema=schema, **kw)


def _val(r: Mapping[str, Any], key: str, *, raw: bool = False) -> Any:
    if key in r:
        v = r[key]
    else:
        v = (r.get("_attrs") or {}).get(key)
    if raw:
        return v
    return v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)


class SynthDataset:
    """One immutable version of a synthetic dataset (see module docstring)."""

    def __init__(self, store: SynthStore, version_id: str) -> None:
        self.store = store
        self.version_id = version_id
        self.dir = store.version_dir(version_id)
        self.manifest: dict[str, Any] = store.read_manifest(version_id)
        self._lineage: dict[str, dict] | None = None

    # -- basic views ---------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.store.name

    @property
    def splits(self) -> dict[str, int]:
        return dict(self.manifest.get("splits") or {})

    @property
    def schema(self) -> Any:
        from topologies.synth.schema import Schema

        cfg = self.manifest.get("schema")
        return Schema.from_config(cfg) if cfg else None

    @property
    def gen_seconds(self) -> float:
        """Wall-clock seconds spent generating this version (all runs that built it)."""
        return float((self.manifest.get("time") or {}).get("seconds") or 0.0)

    @property
    def complete(self) -> bool:
        return bool(self.manifest.get("complete"))

    def __len__(self) -> int:
        return int(self.manifest.get("total", 0))

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in self.splits.items())
        flag = "" if self.complete else ", partial"
        return f"SynthDataset({self.name!r}, version={self.version_id}, {parts}{flag})"

    def __iter__(self) -> Iterator[dict]:
        return iter(self.view())

    def view(self, split: str | None = None) -> RecordView:
        """Records of one split (``None`` = all splits, in split order)."""
        if split is None:
            names = list(self.splits)

            def all_records() -> Iterator[dict]:
                for s in names:
                    yield from iter_shards(self.dir / "records" / s)

            return RecordView(all_records, length=len(self), name=self.name, dataset=self)
        if split not in self.splits:
            raise unknown("split", split, self.splits, stage="dataset")
        d = self.dir / "records" / split
        return RecordView(lambda: iter_shards(d), length=self.splits[split],
                          name=f"{self.name}/{split}", dataset=self)

    def to_records(self, split: str | None = None, *, meta: bool = True) -> list[dict]:
        return self.view(split).to_records(meta=meta)

    def head(self, n: int = 5, split: str | None = None) -> RecordView:
        return self.view(split).head(n)

    def sample(self, n: int = 10, *, by: str | None = None, split: str | None = None,
               seed: int = 0) -> RecordView:
        return self.view(split).sample(n, by=by, seed=seed)

    def filter(self, where: Callable[[dict], bool] | None = None, *, split: str | None = None,
               **equals: Any) -> RecordView:
        return self.view(split).filter(where, **equals)

    def show(self, n: int = 5, *, split: str | None = None, by: str | None = None) -> None:
        from topologies.synth.report import format_records

        rows = self.sample(max(1, n // max(1, len(self.view(split).counts(by)))) if by else n,
                           by=by, split=split) if by else self.head(n, split)
        print(format_records(list(rows), title=f"{self.name} ({self.version_id})"))

    def load_blob(self, ref: dict) -> Any:
        """Live value of a blob reference (``{"$blob": sha, "type": "png", "kind": ...}``):
        a tensor, an ``Audio`` value, a ``Graph`` value, ..."""
        from topologies.synth.valuetypes import decode_value

        return decode_value(ref, self._read_blob)

    def _read_blob(self, ref: Mapping[str, Any]) -> bytes:
        return self.store.blob_path(dict(ref), self.version_id).read_bytes()

    def decode(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """``record`` with stored values (blob refs, inline tensors) turned into live values."""
        from topologies.synth.valuetypes import decode_value

        return {k: (v if str(k).startswith("_") else decode_value(v, self._read_blob))
                for k, v in record.items()}

    # -- lineage / rejected ---------------------------------------------------------------
    def _lineage_index(self) -> dict[str, dict]:
        if self._lineage is None:
            self._lineage = {d["_id"]: d for d in iter_shards(self.dir / "lineage")}
        return self._lineage

    def lineage(self, record_id: str) -> dict[str, Any]:
        lin = self._lineage_index().get(record_id)
        if lin is None:
            for r in self.rejected():
                if r.get("_id") == record_id:
                    return {"_id": record_id, "rejected": True, "op": r.get("_op"),
                            "verdict": r.get("_verdict"), "index": r.get("_index")}
            raise SynthError(f"Record {record_id!r} is not in version {self.version_id}.",
                             stage="dataset", hint="Ids come from ds.head()/ds.sample() "
                                                   "('_id').")
        return lin

    def record(self, record_id: str) -> dict[str, Any]:
        for r in self:
            if r.get("_id") == record_id:
                return r
        for r in self.rejected():
            if r.get("_id") == record_id:
                return r
        raise SynthError(f"Record {record_id!r} not found.", stage="dataset")

    def explain(self, record_id: str) -> dict[str, Any]:
        """Everything known about one record: fields, attributes, generator, latents,
        verdicts, evidence and parents."""
        rec = self.record(record_id)
        lin = self.lineage(record_id)
        return {"record": strip_meta(rec), "attributes": rec.get("_attrs", {}),
                "lineage": lin}

    def rejected(self, op: str | None = None) -> RecordView:
        """Rejected records (with ``_op`` and ``_verdict``), optionally of one op."""
        d = self.dir / "rejected"

        def it() -> Iterator[dict]:
            for r in iter_shards(d):
                if op is None or r.get("_op") == op or str(r.get("_op", "")).startswith(
                        f"{op}#") or str(r.get("_op", "")).split(".")[0] == op:
                    yield r

        return RecordView(it, name=f"rejected({op or 'all'})", dataset=self)

    # -- coverage / stats -------------------------------------------------------------------
    def attributes(self) -> Any:
        """Coverage table: per attribute value, generated / accepted / acceptance rate."""
        from topologies.synth.report import coverage_table

        return coverage_table(self)

    def funnel(self) -> Any:
        from topologies.synth.report import funnel_table

        return funnel_table(self)

    def labels(self, field: str | None = None) -> Counter:
        f = field or (self.schema.label_field() if self.schema is not None else None) or "label"
        return self.view().counts(f)

    # -- torch ---------------------------------------------------------------------------
    def for_recipe(self, recipe: str, *, split: str | None = "train", **kw: Any) -> Any:
        """Torch dataset for ``recipe`` built from one split (see ``synth.torch_views``)."""
        if split is not None and split not in self.splits:
            split = next(iter(self.splits)) if len(self.splits) == 1 else split
        return self.view(split).for_recipe(recipe, **kw)

    def to_torch(self, split: str | None = "train", **kw: Any) -> Any:
        return self.view(split).to_torch(**kw)

    def as_task(self, **kw: Any) -> Any:
        """An ``experiments`` task training on this dataset (see ``synth.tasks.SynthTask``)."""
        from topologies.synth.tasks import SynthTask

        return SynthTask(self, **kw)

    # -- export ----------------------------------------------------------------------------
    def export(self, out: str | Path, *, format: str = "jsonl",
               meta: bool = False, **kw: Any) -> list[Path]:
        """Write the version to ``out/`` (``jsonl`` | ``csv`` | ``txt`` | ``parquet`` |
        ``hf`` | ``imagefolder``); returns the files written."""
        from topologies.synth.export import export

        return export(self, Path(out), format=format, meta=meta, **kw)

    # -- quality / evaluation (later phases build on these) ---------------------------------------
    def quality(self, reference: Any = None, **kw: Any) -> Any:
        from topologies.synth.quality import quality_report

        return quality_report(self, reference=reference, **kw)

    def evaluate(self, real_eval: Any, **kw: Any) -> Any:
        from topologies.synth.quality.utility import evaluate

        return evaluate(self, real_eval, **kw)

    def report(self, *, html: str | Path | None = None, reference: Any = None,
               open_browser: bool = False) -> Any:
        from topologies.synth.report import report

        return report(self, html=html, reference=reference, open_browser=open_browser)

    def label_issues(self, **kw: Any) -> Any:
        from topologies.synth.ops.label import label_issues

        return label_issues(self, **kw)

    # -- versions ---------------------------------------------------------------------------
    def versions(self) -> list[dict]:
        return self.store.versions()

    def diff(self, other: SynthDataset | str) -> dict[str, Any]:
        from topologies.synth.versions import diff

        if isinstance(other, str):
            other = SynthDataset(self.store, self.store.resolve_version(other))
        return diff(self, other)

    def drop(self, ids: Iterable[str], *, reason: str = "dropped") -> SynthDataset:
        from topologies.synth.versions import derive

        drop = set(ids)
        return derive(self, lambda r: None if r["_id"] in drop else r, op="drop",
                      note=f"{reason}: {len(drop)} ids")

    def edit(self, fn: Callable[[dict], dict | None], *, note: str = "edit") -> SynthDataset:
        """New version with ``fn(record)`` applied (return ``None`` to drop a record)."""
        from topologies.synth.versions import derive

        return derive(self, fn, op="edit", note=note)

    def relabel(self, op: Any, *, field: str | None = None) -> SynthDataset:
        """New version with labels rewritten by a ``map`` op / function."""
        from topologies.synth.versions import relabel

        return relabel(self, op, field=field)

    def merge(self, other: Any, *, dedup: bool = True, name: str | None = None) -> SynthDataset:
        from topologies.synth.versions import merge

        return merge(self, other, dedup=dedup, name=name)

    def push_hf(self, repo: str, **kw: Any) -> Any:
        """Upload to the Hugging Face Hub (needs ``datasets``; never automatic)."""
        from topologies.synth.export import push_hf

        return push_hf(self, repo, **kw)


def load(name: str, version: str = "latest", *, root: str | Path | None = None) -> SynthDataset:
    """Open a dataset version (``"latest"``, a version id or prefix, or a label)."""
    if isinstance(name, SynthDataset):
        return name
    store = SynthStore(root, name)
    if not store.exists():
        from topologies.synth.store import list_datasets

        raise unknown("dataset", name, list_datasets(root), stage="dataset",
                      extra=f"No dataset directory under {store.root}.")
    return SynthDataset(store, store.resolve_version(version))


def mix(parts: Mapping[Any, float], *, size: int, name: str, seed: int = 0,
        root: str | Path | None = None, split: str | None = "train") -> SynthDataset:
    """New dataset mixing sources by weight: ``synth.mix({ds_a: 0.7, real: 0.3}, size=...)``."""
    from topologies.synth.versions import mix as _mix

    return _mix(parts, size=size, name=name, seed=seed, root=root, split=split)


__all__ = ["RecordView", "SynthDataset", "load", "mix"]
