"""Contamination checks: synthetic records that overlap eval / reference data.

    synth.ops.contamination.check(real_eval, ngram=13, action="drop")   # pipeline op
    overlap(synthetic_records, eval_records)                            # report of matches

A record is contaminated when it shares at least ``min_shared`` word
``ngram``-grams with a reference record, or (for short texts, where 13-grams
rarely exist) when it is a near duplicate (MinHash Jaccard >= ``near``).
``action="flag"`` keeps the record and records the evidence instead.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from typing import Any

from topologies.synth.errors import unknown
from topologies.synth.ops.base import Aggregate
from topologies.synth.textutil import LSHIndex, MinHasher, normalize, text_of
from topologies.synth.verdict import Verdict


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def record_text(record: dict, key: Any = None) -> str:
    """Text a record is compared by.

    With ``key`` (or for records without numbers) this is :func:`text_of`. Records
    with numeric fields — tabular data — are compared by all their scalar fields
    (``name=value``, numbers to 6 significant digits) and match only exactly:
    comparing only their text would make every row with the same categories a
    duplicate, and character-level similarity says nothing about numbers.
    """
    if key is not None or not any(_is_number(v) for k, v in record.items()
                                  if not str(k).startswith("_")):
        return text_of(record, key)
    words = []
    for k, v in sorted(record.items()):
        if str(k).startswith("_"):
            continue
        if _is_number(v):
            words.append(f"{k}={float(v):.6g}")
        elif isinstance(v, str) or v is None or isinstance(v, bool):
            words.append(f"{k}={v}")
    return _TABULAR + " ".join(words)


_TABULAR = "\x00tabular "


def _grams(text: str, n: int) -> set[int]:
    w = normalize(text).split()
    if len(w) < n:
        return set()
    return {zlib.crc32(" ".join(w[i:i + n]).encode("utf-8", "surrogatepass"))
            for i in range(len(w) - n + 1)}


class ContaminationIndex:
    """n-gram set + MinHash LSH over reference records."""

    def __init__(self, records: Sequence[dict], *, ngram: int = 13, near: float | None = 0.8,
                 key: Any = None) -> None:
        self.ngram, self.near = int(ngram), near
        self.gram_owner: dict[int, int] = {}
        texts = [record_text(r, key) for r in records]
        self.exact: dict[str, int] = {}
        for i, t in enumerate(texts):
            if t.startswith(_TABULAR):
                self.exact.setdefault(t, i)
        texts = [t for t in texts if not t.startswith(_TABULAR)]
        for i, t in enumerate(texts):
            for g in _grams(t, self.ngram):
                self.gram_owner.setdefault(g, i)
        self.hasher = MinHasher(128)
        self.lsh: LSHIndex | None = None
        if near is not None and texts:
            self.lsh = LSHIndex(128, near)
            sigs = self.hasher.signatures(texts)
            keys = self.lsh.band_keys(sigs)
            for i in range(len(texts)):
                self.lsh.add(sigs[i], keys[i], flush=False)
            self.lsh.maybe_flush()

    def match(self, text: str, min_shared: int = 1) -> dict[str, Any] | None:
        if text.startswith(_TABULAR):
            owner = self.exact.get(text)
            return None if owner is None else {"kind": "exact", "reference": owner}
        grams = _grams(text, self.ngram)
        shared = [self.gram_owner[g] for g in grams if g in self.gram_owner]
        if len(shared) >= min_shared:
            owner = max(set(shared), key=shared.count)
            return {"kind": "ngram", "reference": owner, "shared": len(shared),
                    "of": len(grams)}
        if self.lsh is not None:
            sig = self.hasher.signatures([text])
            cands = self.lsh.candidates(self.lsh.band_keys(sig))[0]
            slot, sim = self.lsh.best_match(sig[0], cands)
            if slot is not None and sim >= float(self.near or 1.0):
                return {"kind": "near", "reference": slot, "similarity": round(sim, 4)}
        return None


class Check(Aggregate):
    """Streaming contamination guard (static index; see module docstring)."""

    def __init__(self, reference: Any = None, *, ngram: int = 13, near: float | None = 0.8,
                 min_shared: int = 1, action: str = "drop", key: Any = None,
                 ref_key: Any = None, name: str = "contamination") -> None:
        if action not in ("drop", "flag"):
            raise unknown("contamination action", action, ["drop", "flag"],
                          stage="contamination")
        self.reference = reference
        self.ngram, self.near, self.min_shared = int(ngram), near, int(min_shared)
        self.action, self.key, self.ref_key, self.name = action, key, ref_key, name
        self.evidence = None
        self._index: ContaminationIndex | None = None
        self._records: list[dict] | None = None

    def setup(self, runtime: Any) -> None:
        from topologies.synth.records import as_records

        ref = self.reference
        if ref is None and runtime is not None:
            ref = getattr(runtime.spec, "reference", None)
        self._records = as_records(ref, what="contamination reference") if ref is not None \
            else []

    def start(self) -> ContaminationIndex:
        if self._index is None:
            if self._records is None:
                self.setup(None)
            self._index = ContaminationIndex(self._records or [], ngram=self.ngram,
                                             near=self.near, key=self.ref_key or self.key)
        return self._index

    def offer(self, state: ContaminationIndex, record: dict, meta: dict) -> Verdict:
        m = state.match(record_text(record, self.key), self.min_shared)
        if m is None:
            return Verdict(True)
        if self.action == "flag":
            return Verdict(True, score=1.0, reason="contaminated (flagged)",
                           evidence={"contamination": m})
        return Verdict(False, score=1.0, reason=f"overlaps reference record #{m['reference']} "
                       f"({m['kind']})", evidence={"contamination": m})

    def observe(self, state: Any, record: dict, meta: dict) -> None:
        return None

    def config(self) -> dict[str, Any]:
        from topologies.synth.fingerprint import fp_canonical
        from topologies.synth.records import as_records, fingerprint_records

        ref = None
        if self.reference is not None:
            try:
                ref = fingerprint_records(as_records(self.reference))
            except Exception:  # noqa: BLE001
                ref = fp_canonical(self.reference)
        return {"op": self.name, "reference": ref, "ngram": self.ngram, "near": self.near,
                "min_shared": self.min_shared, "action": self.action,
                "key": fp_canonical(self.key), "ref_key": fp_canonical(self.ref_key)}


def check(reference: Any = None, **kw: Any) -> Check:
    """Pipeline op (``reference=None`` uses ``Spec(reference=...)``)."""
    return Check(reference, **kw)


def overlap(records: Sequence[dict], reference: Sequence[dict], *, ngram: int = 13,
            near: float | None = 0.8, key: Any = None, ref_key: Any = None,
            min_shared: int = 1) -> list[dict[str, Any]]:
    """Records of ``records`` that overlap ``reference`` (with what they match)."""
    idx = ContaminationIndex(reference, ngram=ngram, near=near, key=ref_key or key)
    out = []
    for i, r in enumerate(records):
        m = idx.match(record_text(r, key), min_shared)
        if m is not None:
            out.append({"index": i, "id": r.get("_id"), **m})
    return out


__all__ = ["Check", "ContaminationIndex", "check", "overlap", "record_text"]


from topologies.synth.ops._callable import make_callable

make_callable(__name__, "check")
