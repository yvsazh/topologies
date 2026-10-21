"""Deduplication: exact, normalized, near (MinHash-LSH / SimHash / embeddings), against a set.

    synth.ops.dedup.exact("text")                  # identical field values
    synth.ops.dedup.normalized("text")             # NFKC + casefold + punctuation + spaces
    synth.ops.dedup.near(threshold=0.85)           # MinHash LSH (128 perms, banded)
    synth.ops.dedup.near(0.9, method="simhash")    # 64-bit SimHash, Hamming radius
    synth.ops.dedup.near(0.92, method="embedding") # cosine on embeddings (synth-embed or
                                                   # hashed TF-IDF fallback)
    synth.ops.dedup.prefix(5)                      # first 5 words ("I ordered ..." openings)
    synth.ops.dedup.against(reference, threshold=0.8)  # copy / contamination guard

``key`` picks the text: a field name, a list of fields, ``fn(record)`` or
``None`` (all string fields). Records are seen in raw-index order; the
first occurrence wins, later ones are rejected with the id of the record
they duplicate (``verdict.evidence["duplicate_of"]``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import torch

from topologies.synth.errors import SynthError, unknown
from topologies.synth.fingerprint import callable_fingerprint
from topologies.synth.ops.base import Aggregate
from topologies.synth.textutil import (
    LSHIndex,
    MinHasher,
    hamming,
    normalize,
    simhash64,
    text_of,
    to_signed64,
)
from topologies.synth.verdict import Verdict

_OK = Verdict(True)


def _key_cfg(key: Any) -> Any:
    return callable_fingerprint(key) if callable(key) else key


class Exact(Aggregate):
    """Reject records whose key text (optionally normalized) was seen before."""

    def __init__(self, key: Any = None, *, normalize_text: bool = False,
                 name: str | None = None) -> None:
        self.key = key
        self.normalize_text = bool(normalize_text)
        self.name = name or ("dedup.normalized" if normalize_text else "dedup.exact")
        self.evidence = None

    def _digest(self, record: dict) -> str:
        t = text_of(record, self.key)
        if self.normalize_text:
            t = normalize(t)
        return hashlib.blake2b(t.encode("utf-8", "surrogatepass"), digest_size=12).hexdigest()

    def start(self) -> dict[str, str]:
        return {}

    def offer(self, state: dict[str, str], record: dict, meta: dict) -> Verdict:
        d = self._digest(record)
        first = state.get(d)
        if first is not None:
            return Verdict(False, score=1.0, reason="exact duplicate",
                           evidence={"duplicate_of": first})
        state[d] = meta.get("id", "")
        return _OK

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "key": _key_cfg(self.key), "normalize": self.normalize_text}


class Prefix(Aggregate):
    """At most ``max_per_prefix`` records per normalized opening of ``words`` words."""

    def __init__(self, words: int = 5, *, key: Any = None, max_per_prefix: int = 1,
                 name: str | None = None) -> None:
        self.words, self.key, self.max_per_prefix = int(words), key, int(max_per_prefix)
        self.name = name or "dedup.prefix"

    def start(self) -> dict[str, int]:
        return {}

    def offer(self, state: dict[str, int], record: dict, meta: dict) -> Verdict:
        p = " ".join(normalize(text_of(record, self.key)).split()[: self.words])
        n = state.get(p, 0)
        if n >= self.max_per_prefix:
            return Verdict(False, reason=f"opening {p!r} used {n} times")
        state[p] = n + 1
        return _OK

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "words": self.words, "key": _key_cfg(self.key),
                "max": self.max_per_prefix}


class _NearState:
    def __init__(self, op: Near) -> None:
        self.op = op
        self.ids: list[str] = []
        if op.method == "minhash":
            self.index = LSHIndex(op.num_perm, op.threshold)
        elif op.method == "simhash":
            self.hashes: list[int] = []
            self.tensor: torch.Tensor | None = None
        else:
            self.vecs: list[torch.Tensor] = []
            self.mat: torch.Tensor | None = None


class Near(Aggregate):
    """Near-duplicate removal (see module docstring)."""

    def __init__(self, threshold: float = 0.85, *, method: str = "minhash", key: Any = None,
                 num_perm: int = 128, shingle: int = 5, unit: str = "char",
                  embed: Any = None, value_type: Any = None, name: str | None = None) -> None:
        if method not in ("minhash", "simhash", "embedding", "features"):
            raise unknown("dedup method", method, ["minhash", "simhash", "embedding", "features"],
                          stage="dedup")
        if not 0 < threshold <= 1:
            raise SynthError(f"dedup threshold must be in (0, 1]; got {threshold}.",
                             stage="dedup")
        self.threshold, self.method, self.key = float(threshold), method, key
        self.num_perm, self.shingle, self.unit = int(num_perm), int(shingle), unit
        self.embed = embed
        self.value_type = value_type
        self.name = name or "dedup.near"
        self._hasher: MinHasher | None = None
        self._embedder: Any = None

    @property
    def hasher(self) -> MinHasher:
        if self._hasher is None:
            self._hasher = MinHasher(self.num_perm, k=self.shingle, unit=self.unit)
        return self._hasher

    def embedder(self) -> Any:
        if self._embedder is None:
            from topologies.synth.quality.embed import get_embedder

            self._embedder = get_embedder(self.embed)
        return self._embedder

    def start(self) -> _NearState:
        return _NearState(self)

    def offer(self, state: _NearState, record: dict, meta: dict) -> Verdict:
        return self.offer_batch(state, [record], [meta])[0]

    def observe(self, state: _NearState, record: dict, meta: dict) -> None:
        self.observe_batch(state, [record], [meta])

    def observe_batch(self, state: _NearState, records: Sequence[dict],
                      metas: Sequence[dict]) -> None:
        self._process(state, records, metas, check=False)

    def offer_batch(self, state: _NearState, records: Sequence[dict],
                    metas: Sequence[dict]) -> list[Verdict]:
        return self._process(state, records, metas, check=True)

    def _process(self, state: _NearState, records: Sequence[dict], metas: Sequence[dict], *,
                 check: bool) -> list[Verdict]:
        if not records:
            return []
        if self.method == "features":
            if self.value_type is None:
                raise SynthError("features dedup requires value_type", stage="dedup")
            vals = [self.value_type.unwrap(r.get(self.key)) if r.get(self.key) is not None
                    else None for r in records]
            present = [v for v in vals if v is not None]
            if present:
                encoded = torch.as_tensor(self.value_type.features(present), dtype=torch.float32)
                if encoded.ndim == 1:
                    encoded = encoded[:, None]
                if encoded.shape[1] == 0:
                    raise SynthError("features dedup value_type.features returned zero columns; "
                                     "provide at least one feature", stage="dedup")
                vecs = torch.zeros((len(vals), encoded.shape[1] + 1), dtype=torch.float32)
                row = 0
                for i, value in enumerate(vals):
                    if value is None:
                        vecs[i, -1] = 1.0
                    else:
                        vecs[i, :-1] = torch.nan_to_num(encoded[row])
                        row += 1
            else:
                vecs = torch.ones((len(vals), 1), dtype=torch.float32)
            vecs = torch.nn.functional.normalize(vecs, dim=-1)
            return self._vectors(state, vecs, metas, check)
        texts = [text_of(r, self.key) for r in records]
        if self.method == "minhash":
            return self._minhash(state, texts, metas, check)
        if self.method == "simhash":
            return self._simhash(state, texts, metas, check)
        return self._embedding(state, texts, metas, check)

    def _minhash(self, state: _NearState, texts: list[str], metas: Sequence[dict],
                 check: bool) -> list[Verdict]:
        idx = state.index
        sigs = self.hasher.signatures(texts)
        keys = idx.band_keys(sigs)
        out: list[Verdict] = []
        cands = idx.candidates(keys) if check else [set() for _ in texts]
        for n, meta in enumerate(metas):
            if check:
                # Earlier chunks (runs + recent) were looked up for the whole batch; records
                # added from this batch are still in ``_recent`` (no flush mid-batch).
                extra = {s for kk in keys[n].tolist()
                         if (s := idx._recent.get(kk)) is not None} if n else set()
                slot, sim = idx.best_match(sigs[n], cands[n] | extra)
                if slot is not None and sim >= self.threshold:
                    out.append(Verdict(False, score=round(sim, 4), reason=f"near duplicate "
                                       f"(est. Jaccard {sim:.2f})",
                                       evidence={"duplicate_of": state.ids[slot]}))
                    continue
            idx.add(sigs[n], keys[n], flush=False)
            state.ids.append(meta.get("id", ""))
            out.append(_OK)
        idx.maybe_flush()
        return out

    def _simhash(self, state: _NearState, texts: list[str], metas: Sequence[dict],
                 check: bool) -> list[Verdict]:
        radius = round((1 - self.threshold) * 64)
        out = []
        for t, meta in zip(texts, metas):
            h = to_signed64(simhash64(t))
            if check and state.hashes:
                if state.tensor is None or state.tensor.numel() != len(state.hashes):
                    state.tensor = torch.tensor(state.hashes, dtype=torch.int64)
                d = hamming(state.tensor, torch.tensor(h, dtype=torch.int64))
                j = int(d.argmin())
                if int(d[j]) <= radius:
                    out.append(Verdict(False, score=1 - int(d[j]) / 64,
                                       reason=f"near duplicate (simhash distance {int(d[j])})",
                                       evidence={"duplicate_of": state.ids[j]}))
                    continue
            state.hashes.append(h)
            state.ids.append(meta.get("id", ""))
            out.append(_OK)
        return out

    def _embedding(self, state: _NearState, texts: list[str], metas: Sequence[dict],
                   check: bool) -> list[Verdict]:
        vecs = self.embedder().encode(texts)
        vecs = torch.nn.functional.normalize(torch.as_tensor(vecs, dtype=torch.float32), dim=-1)
        out = []
        for n, meta in enumerate(metas):
            v = vecs[n]
            if check and state.vecs:
                if state.mat is None or state.mat.shape[0] != len(state.vecs):
                    state.mat = torch.stack(state.vecs)
                sims = state.mat @ v
                j = int(sims.argmax())
                if float(sims[j]) >= self.threshold:
                    out.append(Verdict(False, score=round(float(sims[j]), 4),
                                       reason=f"near duplicate (cosine {float(sims[j]):.2f})",
                                       evidence={"duplicate_of": state.ids[j]}))
                    continue
            state.vecs.append(v)
            state.ids.append(meta.get("id", ""))
            out.append(_OK)
        return out

    def _vectors(self, state: _NearState, vecs: torch.Tensor, metas: Sequence[dict],
                 check: bool) -> list[Verdict]:
        out = []
        for vec, meta in zip(vecs, metas):
            if check and state.vecs:
                previous = torch.stack(state.vecs)
                sims = previous @ vec
                if float(vec.norm()) == 0.0:
                    sims = torch.where(previous.norm(dim=1) == 0, 1.0, sims)
                j = int(sims.argmax())
                if float(sims[j]) >= self.threshold:
                    out.append(Verdict(False, score=round(float(sims[j]), 4),
                                       reason=f"near duplicate (feature cosine {float(sims[j]):.2f})",
                                       evidence={"duplicate_of": state.ids[j]}))
                    continue
            state.vecs.append(vec)
            state.ids.append(meta.get("id", ""))
            out.append(_OK)
        return out

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "threshold": self.threshold, "method": self.method,
                "key": _key_cfg(self.key), "num_perm": self.num_perm, "shingle": self.shingle,
                "unit": self.unit, "embed": str(self.embed) if self.embed else None,
                "value_type": self.value_type.to_config() if self.value_type is not None else None}


class Against(Aggregate):
    """Reject records close to any record of a fixed set (seed copies, eval leakage)."""

    def __init__(self, records: Any, threshold: float = 0.8, *, key: Any = None,
                 ref_key: Any = None, method: str = "minhash", name: str | None = None) -> None:
        from topologies.synth.records import as_records

        self.records = as_records(records, what="dedup.against records")
        self.threshold, self.key, self.ref_key = float(threshold), key, ref_key
        self.method = method
        self.name = name or "dedup.against"
        self._near = Near(threshold, method=method, key=key)
        self._ref_state: _NearState | None = None

    def start(self) -> Any:
        if self._ref_state is None:
            st = self._near.start()
            ref_near = Near(self.threshold, method=self.method,
                            key=self.ref_key if self.ref_key is not None else self.key)
            ref_near._hasher = self._near.hasher
            texts_meta = [{"id": f"ref:{i}"} for i in range(len(self.records))]
            ref_near.observe_batch(st, self.records, texts_meta)
            self._ref_state = st
        return self._ref_state

    def offer(self, state: Any, record: dict, meta: dict) -> Verdict:
        return self.offer_batch(state, [record], [meta])[0]

    def offer_batch(self, state: Any, records: Sequence[dict],
                    metas: Sequence[dict]) -> list[Verdict]:
        # Check only: accepted records are not added to the reference index.
        out = []
        for r, m in zip(records, metas):
            texts = [text_of(r, self.key)]
            if self.method == "minhash":
                sig = self._near.hasher.signatures(texts)
                keys = state.index.band_keys(sig)
                cands = state.index.candidates(keys)[0]
                slot, sim = state.index.best_match(sig[0], cands)
                if slot is not None and sim >= self.threshold:
                    out.append(Verdict(False, score=round(sim, 4),
                                       reason=f"too close to a reference record ({sim:.2f})",
                                       evidence={"reference": state.ids[slot]}))
                    continue
            else:
                tmp = self._near._process(state, [r], [m], check=True)[0]
                if not tmp.ok:
                    # Undo the insertion side effect is unnecessary: rejected rows are not added.
                    out.append(Verdict(False, score=tmp.score, reason="too close to a "
                                       "reference record", evidence=tmp.evidence))
                    continue
                self._pop_last(state)
            out.append(_OK)
        return out

    def _pop_last(self, state: Any) -> None:
        # _process appended the accepted record; the reference set must stay fixed.
        state.ids.pop()
        if self.method == "simhash":
            state.hashes.pop()
            state.tensor = None
        else:
            state.vecs.pop()
            state.mat = None

    def observe(self, state: Any, record: dict, meta: dict) -> None:
        return None

    def config(self) -> dict[str, Any]:
        from topologies.synth.records import fingerprint_records

        return {"op": self.name, "records": fingerprint_records(self.records),
                "threshold": self.threshold, "key": _key_cfg(self.key),
                "ref_key": _key_cfg(self.ref_key), "method": self.method}


def exact(key: Any = None, *, name: str | None = None) -> Exact:
    return Exact(key, name=name)


def normalized(key: Any = None, *, name: str | None = None) -> Exact:
    return Exact(key, normalize_text=True, name=name)


def near(threshold: float = 0.85, *, method: str = "minhash", key: Any = None,
         **kw: Any) -> Near:
    return Near(threshold, method=method, key=key, **kw)


def prefix(words: int = 5, *, key: Any = None, max_per_prefix: int = 1) -> Prefix:
    return Prefix(words, key=key, max_per_prefix=max_per_prefix)


def against(records: Any, threshold: float = 0.8, *, key: Any = None, ref_key: Any = None,
            method: str = "minhash", name: str | None = None) -> Against:
    return Against(records, threshold, key=key, ref_key=ref_key, method=method, name=name)


__all__ = ["Against", "Exact", "Near", "Prefix", "against", "exact", "near", "normalized",
           "prefix"]


from topologies.synth.ops._callable import make_callable

make_callable(__name__, "near")
