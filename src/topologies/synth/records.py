"""Plain record lists: load real data, split it, normalize ``Records`` arguments.

    real = synth.load_records("data/support_real.jsonl")      # jsonl / json / csv / txt
    seeds, real_eval = synth.split_records(real, 0.3, stratify="label")

``Records`` arguments (``seeds=``, ``reference=``, ``real_eval``) accept a list
of dicts, a path, a ``SynthDataset`` / view, or a list of strings (each becomes
``{"text": s}``).
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from topologies.synth.errors import SynthError

Records = Any  # list[dict] | str | Path | SynthDataset | RecordView | list[str]


def load_records(path: str | Path, *, text_field: str = "text",
                 limit: int | None = None) -> list[dict[str, Any]]:
    """Read records from ``.jsonl`` / ``.json`` (list or ``{"records": [...]}``) / ``.csv`` /
    ``.tsv`` / ``.txt`` (one ``{text_field: line}`` per line) / ``.parquet`` (extra)."""
    p = Path(path)
    if p.is_dir():
        from topologies.synth.store import iter_shards

        out = list(iter_shards(p))
        return out[:limit] if limit else out
    if not p.is_file():
        raise SynthError(f"No such file: {p}", stage="records")
    suffixes = "".join(p.suffixes[-2:])
    if suffixes.endswith((".jsonl", ".jsonl.gz")):
        from topologies.synth.store import iter_jsonl

        out = []
        for r in iter_jsonl(p):
            out.append(r if isinstance(r, dict) else {text_field: r})
            if limit and len(out) >= limit:
                break
        return out
    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("records", data.get("data", []))
        if not isinstance(data, list):
            raise SynthError(f"{p} holds no list of records.", stage="records")
        out = [r if isinstance(r, dict) else {text_field: r} for r in data]
        return out[:limit] if limit else out
    if p.suffix in (".csv", ".tsv"):
        with p.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t" if p.suffix == ".tsv" else ",")
            out = [dict(r) for r in reader]
        return out[:limit] if limit else out
    if p.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise SynthError("Reading parquet needs pyarrow.", stage="records",
                             hint="pip install 'topologies[synth-parquet]'") from e
        out = pq.read_table(p).to_pylist()
        return out[:limit] if limit else out
    lines = [line.rstrip("\n") for line in p.read_text(encoding="utf-8").splitlines()]
    out = [{text_field: line} for line in lines if line.strip()]
    return out[:limit] if limit else out


def as_records(obj: Records, *, what: str = "records") -> list[dict[str, Any]]:
    """Normalize a ``Records`` argument (see module docstring) to a list of dicts."""
    if obj is None:
        return []
    if isinstance(obj, (str, Path)):
        return load_records(obj)
    to_records = getattr(obj, "to_records", None)
    if callable(to_records):
        return list(to_records())
    if isinstance(obj, Mapping):
        raise SynthError(f"{what} must be a list of records, not a dict.", stage="records",
                         hint="Wrap a single record in a list: [record].")
    if isinstance(obj, Iterable):
        out = []
        for r in obj:
            if isinstance(r, Mapping):
                out.append(dict(r))
            elif isinstance(r, str):
                out.append({"text": r})
            else:
                raise SynthError(f"{what} items must be dicts or strings; got "
                                 f"{type(r).__name__}.", stage="records")
        return out
    raise SynthError(f"Cannot read {what} from {type(obj).__name__}.", stage="records",
                     hint="Pass a list of dicts, a .jsonl/.csv path, or a SynthDataset.")


def split_records(records: Records, test: float = 0.2, *, stratify: str | None = None,
                  group: str | None = None, seed: int = 0
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(train, test)`` with the ``data.splits`` semantics: ``test`` is a fraction
    (``0 < test < 1``) or a count; ``stratify="label"`` keeps label proportions;
    ``group="parent"`` keeps every record of a group on one side."""
    rows = as_records(records)
    n = len(rows)
    if n < 2:
        raise SynthError("Need at least 2 records to split.", stage="records")
    k = round(n * test) if isinstance(test, float) and test < 1 else int(test)
    k = min(max(k, 1), n - 1)
    rng = random.Random(seed)
    if group is not None:
        groups: dict[Any, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            groups[_key(r.get(group, i))].append(i)
        keys = sorted(groups, key=repr)
        rng.shuffle(keys)
        test_idx: list[int] = []
        for g in keys:
            if len(test_idx) >= k:
                break
            test_idx += groups[g]
        chosen = set(test_idx)
    elif stratify is not None:
        by: dict[Any, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            by[_key(r.get(stratify))].append(i)
        chosen = set()
        exact = {c: k * len(idx) / n for c, idx in by.items()}
        take = {c: int(v) for c, v in exact.items()}
        rest = k - sum(take.values())
        for c in sorted(by, key=lambda c: exact[c] - take[c], reverse=True)[:rest]:
            take[c] += 1
        for c in sorted(by, key=repr):
            idx = list(by[c])
            rng.shuffle(idx)
            chosen.update(idx[: take[c]])
    else:
        idx = list(range(n))
        rng.shuffle(idx)
        chosen = set(idx[:k])
    train = [r for i, r in enumerate(rows) if i not in chosen]
    test_rows = [r for i, r in enumerate(rows) if i in chosen]
    return train, test_rows


def _key(v: Any) -> Any:
    try:
        hash(v)
        return v
    except TypeError:
        return json.dumps(v, sort_keys=True, default=str)


def strip_meta(record: Mapping[str, Any]) -> dict[str, Any]:
    """Record without ``_``-prefixed metadata keys."""
    return {k: v for k, v in record.items() if not str(k).startswith("_")}


def fingerprint_records(records: Sequence[Mapping[str, Any]] | None) -> str | None:
    if not records:
        return None
    from topologies.experiments.cache import stable_hash

    return stable_hash([strip_meta(r) for r in records])


__all__ = ["Records", "as_records", "fingerprint_records", "load_records", "split_records",
           "strip_meta"]
