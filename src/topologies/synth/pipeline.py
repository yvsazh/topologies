"""Streaming executor: chunked, resumable, ordered-deterministic generation.

Raw record indices are processed in chunks (``chunk_size``, default 256):

1. *generate*: per index, the sampler's attributes + derived attributes, then the
   generator (each record gets its own seeded RNG);
2. *per-record ops* (map / filter / flat_map) run inside a chunk;
3. *streaming aggregates* (dedup, contamination) are applied in raw-index order,
   chunk after chunk, with incremental state (a chunk waits for its turn);
4. the processed chunk (accepted + rejected + lineage) is written atomically,
   followed by its ``.done`` marker.

Every order-dependent step is applied in chunk order, so the output never
depends on timing. The accepted stream then passes the *final* aggregates
(balance) as a streaming quota selection until ``size`` records are selected;
splits are assigned by hash rank. A finished chunk is never recomputed
(resume), and stopping early (``limit``, ``max_time``, Ctrl+C) still writes a
valid partial version.
"""

from __future__ import annotations

import hashlib
import math
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from topologies.synth.errors import RunStopped, SynthError
from topologies.synth.evaluate import Evaluator, Sample
from topologies.synth.generators.base import GenContext, Raw, collect, default_assemble
from topologies.synth.ops.base import Aggregate, Op, Reject
from topologies.synth.rng import SynthRandom, derive_seed
from topologies.synth.valuetypes import decode_value, encode_value, normalize_record
from topologies.synth.verdict import Verdict, as_verdict

CELL_MAX_VALUES = 64


def record_id(gen_fp: str, index: int, sub: Sequence[int]) -> str:
    """Stable record id: ``hash(generation fingerprint, raw index, sub-index path)``."""
    key = f"{gen_fp}:{index}:{'.'.join(str(s) for s in sub)}"
    return hashlib.blake2b(key.encode(), digest_size=8).hexdigest()


@dataclass
class Item:
    """A record in flight (fields + metadata); ``status``: ``a`` accepted, ``r`` rejected."""

    rec: dict[str, Any]
    id: str
    i: int
    j: tuple[int, ...]
    attrs: dict[str, Any]
    lin: dict[str, Any] = field(default_factory=dict)
    status: str = "a"
    op: str | None = None
    pos: int = -1
    verdict: dict[str, Any] | None = None

    @property
    def key(self) -> tuple:
        return (self.i, self.j)

    def view(self) -> dict[str, Any]:
        """What ops see: fields + ``_id`` + ``_attrs``."""
        return {**self.rec, "_id": self.id, "_attrs": self.attrs}

    def reject(self, op: str, pos: int, verdict: Verdict) -> None:
        self.status, self.op, self.pos = "r", op, pos
        self.verdict = verdict.as_dict()

    def to_line(self) -> dict[str, Any]:
        d: dict[str, Any] = {"s": self.status, "id": self.id, "i": self.i, "j": list(self.j),
                             "rec": self.rec, "attrs": self.attrs, "lin": self.lin}
        if self.status == "r":
            d.update({"op": self.op, "pos": self.pos, "v": self.verdict})
        return d

    @classmethod
    def from_line(cls, d: dict[str, Any]) -> Item:
        return cls(rec=d["rec"], id=d["id"], i=int(d["i"]), j=tuple(d.get("j") or ()),
                   attrs=d.get("attrs") or {}, lin=d.get("lin") or {}, status=d.get("s", "a"),
                   op=d.get("op"), pos=int(d.get("pos", -1)), verdict=d.get("v"))

    def meta(self) -> dict[str, Any]:
        return {"id": self.id, "group": self.lin.get("group") or self.id, "i": self.i,
                "attrs": self.attrs}


class SchemaOp(Op):
    """Implicit first op: coerce + validate against the spec schema."""

    kind = "filter"
    name = "schema"

    def __init__(self, schema: Any) -> None:
        self.schema = schema

    def fingerprint(self) -> str:
        from topologies.experiments.cache import stable_hash

        return stable_hash({"op": "schema", "schema": self.schema.to_config()})


# ---------------------------------------------------------------- selection


class Selection:
    """Streaming selection of the final dataset from the accepted stream (balance quotas)."""

    def __init__(self, final_ops: list[Op], target: int) -> None:
        self.target = int(target)
        self.selectors = [op.selector(target) for op in final_ops]  # type: ignore[attr-defined]
        self.names = [op.name for op in final_ops]
        self.chosen: list[tuple[int, str]] = []  # (chunk, id)
        self.dropped: dict[str, str] = {}  # id -> op name
        self.surplus = 0
        self.dups: list[tuple[str, int]] = []  # (record id, copy number): upsampling

    def done(self) -> bool:
        return len(self.chosen) + len(self.dups) >= self.target

    def offer(self, chunk: int, item: Item) -> None:
        if self.done():
            self.surplus += 1
            return
        rec, meta = item.view(), {**item.meta(), "chunk": chunk}
        for sel, name in zip(self.selectors, self.names):
            if not sel.offer(rec, meta):
                self.dropped[item.id] = name
                return
        self.chosen.append((chunk, item.id))

    def relax(self) -> None:
        """Raw budget exhausted: fill remaining slots with quota-dropped records."""
        for sel in self.selectors:
            for chunk, rid in sel.relax():
                if self.done():
                    return
                if rid in self.dropped:
                    del self.dropped[rid]
                    self.chosen.append((chunk, rid))
            for rid, k in getattr(sel, "dups", []):
                if self.done():
                    return
                self.dups.append((rid, k))


# ---------------------------------------------------------------- the executor


class Generation:
    """Raw generation of records (attributes + node evaluation + lineage); shared by the
    executor and its worker processes (it pickles when the generator does)."""

    def __init__(self, generator: Any, attributes: Any, sampler: Any, seed: int, schema: Any,
                 on_error: str, gen_fp: str, env: Any = None) -> None:
        self.gen = generator
        self.attributes, self.sampler, self.seed = attributes, sampler, int(seed)
        self.schema, self.on_error, self.gen_fp = schema, on_error, gen_fp
        self.gen_label = generator.node_kind()
        self.ev = Evaluator(generator, seed=self.seed, env=env)

    @classmethod
    def from_spec(cls, spec: Any, env: Any = None) -> Generation:
        return cls(spec.generator, spec.attributes, spec.sampler, spec.seed, spec.schema,
                   spec.on_error, spec.gen_fingerprint(), env)

    def __getstate__(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "ev"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.ev = Evaluator(self.gen, seed=self.seed)

    def attrs(self, i: int) -> dict[str, Any]:
        base = self.sampler.assign(i)
        rng = SynthRandom(derive_seed(self.seed, i, "attrs"))
        return self.attributes.complete(base, rng, i)

    def items_from(self, s: Sample) -> list[Item]:
        """Records (with lineage) from one record's evaluation."""
        i, attrs = s.index, s.attrs
        if s.error is not None:
            e = s.error
            if isinstance(e, (RunStopped, KeyboardInterrupt)):
                raise e
            if isinstance(e, Reject):
                it = Item(rec={}, id=record_id(self.gen_fp, i, ("e",)), i=i, j=(-1,),
                          attrs=attrs, lin={"generator": self.gen_label, "latent": s.latents})
                it.reject("generator", -1, e.verdict)
                return [it]
            if isinstance(e, SynthError) or self.on_error == "raise":
                raise e
            it = Item(rec={}, id=record_id(self.gen_fp, i, ("e",)), i=i, j=(-1,), attrs=attrs,
                      lin={"generator": self.gen_label,
                           "error": f"{type(e).__name__}: {e}"[:500]})
            it.reject("generator", -1, Verdict(False, reason=f"{type(e).__name__}: {e}"[:300]))
            return [it]
        items = []
        gen_name = s.logs.pop("generator", None) or self.gen_label
        for j, out in enumerate(collect(s.value)):
            raw = out if isinstance(out, Raw) else Raw(out)
            rec = default_assemble(attrs, raw.value, self.schema)
            rec = normalize_record({k: v for k, v in rec.items() if not str(k).startswith("_")},
                                   self.schema)
            rid = record_id(self.gen_fp, i, (j,))
            evidence = raw.evidence or getattr(self.gen, "evidence", None)
            lin: dict[str, Any] = {"generator": gen_name, "seed": self.seed,
                                   "raw_index": i, "fanout_index": j,
                                   "evidence": [evidence] if evidence else [],
                                   "verdicts": {}}
            if s.latents:
                lin["latent"] = dict(s.latents)
            if raw.parents:
                lin["parents"] = raw.parents
            if raw.group:
                lin["group"] = raw.group
            if raw.lineage:
                lin.update(raw.lineage)
                # Generators that emit several related outputs refer to siblings by sub-index.
                if "parent_sub" in lin:
                    lin["parents"] = [record_id(self.gen_fp, i, (int(lin.pop("parent_sub")),))]
                if "group_sub" in lin:
                    lin["group"] = record_id(self.gen_fp, i, (int(lin.pop("group_sub")),))
            if s.logs:
                lin["logs"] = dict(s.logs)
            if s.stats:
                lin["stats"] = dict(s.stats)
            items.append(Item(rec=rec, id=rid, i=i, j=(j,), attrs=attrs, lin=lin))
        return items

    def generate_many(self, indices: Sequence[int], *, batch: bool = True) -> list[Item]:
        """Raw records of ``indices`` (batched nodes render once per node)."""
        attrs = [self.attrs(i) for i in indices]
        samples = self.ev.evaluate_many(list(indices), attrs, batch=batch)
        return [x for s in samples for x in self.items_from(s)]


# Worker processes (``Spec(workers=N)``): each holds one unpickled ``Generation``.
_WORKER: dict[str, Any] = {}


def _worker_init(payload: bytes) -> None:
    import pickle

    from topologies.synth.core import iter_nodes

    gen: Generation = pickle.loads(payload)
    for n in iter_nodes(gen.gen):
        n.setup(None)
    _WORKER["gen"] = gen


def _worker_chunk(indices: list[int]) -> tuple[list[Item], dict[str, float], dict[str, int],
                                               float]:
    gen: Generation = _WORKER["gen"]
    t0 = time.perf_counter()
    items = gen.generate_many(indices)
    t, n = gen.ev.take_timings()
    return items, t, n, time.perf_counter() - t0


@dataclass
class ChunkResult:
    index: int
    items: list[Item]
    summary: dict[str, Any]
    cached: bool = False


class Executor:
    """Runs one spec (see module docstring)."""

    def __init__(self, spec: Any, runtime: Any, *, progress: Any = None) -> None:
        self.spec = spec
        self.rt = runtime
        self.store = runtime.store
        self.gen = spec.generator
        self.gen_fp = spec.gen_fingerprint()
        self.stream_fp = spec.stream_fingerprint()
        self.schema = spec.schema
        ops: list[Op] = []
        if self.schema is not None:
            ops.append(SchemaOp(self.schema))
        ops += spec.stream_ops()
        self.ops = ops
        # Segments: per-record ops separated by streaming aggregates.
        self.segments: list[tuple[list[tuple[int, Op]], tuple[int, Aggregate] | None]] = []
        cur: list[tuple[int, Op]] = []
        for pos, op in enumerate(ops):
            if op.kind == "aggregate":
                self.segments.append((cur, (pos, op)))  # type: ignore[arg-type]
                cur = []
            else:
                cur.append((pos, op))
        self.segments.append((cur, None))
        self.aggs = [agg for _, agg in self.segments if agg is not None]
        self.agg_states = [agg[1].start() for agg in self.aggs]
        self.final_ops = spec.final_ops()
        self.progress = progress
        self.genr = Generation.from_spec(spec, runtime)
        self.gen_label = self.genr.gen_label
        self.ev = self.genr.ev
        self.cs = int(spec.chunk_size)
        self.pool: Any = None
        self._ahead: dict[int, Any] = {}
        self._timing: dict[str, float] = defaultdict(float)

    def chunk_id(self, c: int) -> str:
        return f"{self.stream_fp[:16]}-{c:06d}"

    # -- generation ---------------------------------------------------------------
    def _attrs(self, i: int) -> dict[str, Any]:
        return self.genr.attrs(i)

    def encode(self, rec: dict[str, Any]) -> dict[str, Any]:
        """Storage form of a record: inline JSON values and type-codec blobs."""
        put = self.store.put_blob if self.store is not None else None
        fields = self.schema.fields if self.schema is not None else {}
        return {k: encode_value(v, fields.get(k), put) for k, v in rec.items()}

    def decode(self, rec: dict[str, Any]) -> dict[str, Any]:
        load = (lambda ref: self.store.blob_path(ref).read_bytes()) if self.store is not None \
            else None
        return {k: decode_value(v, load) for k, v in rec.items()}

    def generate_many(self, indices: Sequence[int], *, batch: bool = True) -> list[Item]:
        """Raw records of ``indices`` (live values; batched nodes render once per node)."""
        return self.genr.generate_many(indices, batch=batch)

    def generate(self, i: int) -> list[Item]:
        return self.generate_many([i], batch=False)

    # -- worker processes ------------------------------------------------------------
    def start_workers(self, n: int) -> bool:
        """A process pool for generation; ``False`` (in-process) when the generator does
        not pickle (lambdas, local functions)."""
        import multiprocessing as mp
        import pickle
        from concurrent.futures import ProcessPoolExecutor

        try:
            payload = pickle.dumps(self.genr)
        except Exception as e:  # noqa: BLE001
            self.rt.event("workers", status="in-process",
                          reason=f"generator does not pickle: {type(e).__name__}: {e}"[:300])
            return False
        self.pool = ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"),
                                        initializer=_worker_init, initargs=(payload,))
        self._n_workers = n
        return True

    def stop_workers(self) -> None:
        if self.pool is not None:
            for f in self._ahead.values():
                f.cancel()
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
            self._ahead.clear()

    def _prefetch(self, c: int, last: int) -> None:
        """Submit generation of chunks ``c..`` (not done yet) to the pool."""
        k = c
        while len(self._ahead) < 2 * self._n_workers and k < last:
            cached = self.store is not None and self.store.chunk_done(self.chunk_id(k))
            if k not in self._ahead and not cached:
                self._ahead[k] = self.pool.submit(
                    _worker_chunk, list(range(k * self.cs, (k + 1) * self.cs)))
            k += 1

    def _generated(self, c: int) -> list[Item]:
        fut = self._ahead.pop(c, None)
        if fut is None:
            return self.generate_many(range(c * self.cs, (c + 1) * self.cs))
        try:
            items, t, n, _secs = fut.result()
        except (SynthError, Reject, KeyboardInterrupt):
            raise
        except Exception as e:
            from concurrent.futures.process import BrokenProcessPool

            if not isinstance(e, BrokenProcessPool):
                raise
            self.rt.event("workers", status="in-process", reason=f"worker pool broke: {e}"[:300])
            self.stop_workers()
            return self.generate_many(range(c * self.cs, (c + 1) * self.cs))
        for p, v in t.items():
            self.ev.timings[p] += v
        for p, v in n.items():
            self.ev.calls[p] += v
        return items

    # -- per-record ops -------------------------------------------------------------
    def _op_ctx(self, it: Item, op: Op) -> GenContext:
        stage = f"op:{op.name}:{'.'.join(str(s) for s in it.j)}"
        return GenContext(runtime=self.rt, index=it.i, attrs=it.attrs, seed=self.spec.seed,
                          stage=stage)

    def _after(self, it: Item, op: Op, pos: int, kind: str, out: Any,
               ctx: GenContext) -> list[Item]:
        if ctx.logs:
            it.lin.setdefault("logs", {}).update(ctx.logs)
        if kind == "map":
            if out is None:
                return [it]
            if not isinstance(out, dict):
                raise SynthError(f"map op {op.name} returned {type(out).__name__}.",
                                 stage="pipeline")
            it.rec = {k: v for k, v in out.items() if not str(k).startswith("_")}
            return [it]
        if kind == "filter":
            v = as_verdict(out)
            if v.score is not None or v.reason or v.evidence:
                it.lin.setdefault("verdicts", {})[op.name] = v.as_dict()
            if not v.ok:
                it.reject(op.name, pos, v)
            elif op.evidence:
                ev = it.lin.setdefault("evidence", [])
                if op.evidence not in ev:
                    ev.append(op.evidence)
            return [it]
        # flat_map
        children = []
        for k, child in enumerate(out or []):
            if isinstance(child, Raw):
                crec, extra = child.value, child.lineage
            else:
                crec, extra = child, {}
            if not isinstance(crec, dict):
                raise SynthError(f"flat_map op {op.name} produced {type(crec).__name__}.",
                                 stage="pipeline")
            cid = hashlib.blake2b(f"{it.id}:{op.name}:{k}".encode(), digest_size=8).hexdigest()
            lin = {**{kk: vv for kk, vv in it.lin.items() if kk != "logs"},
                   "parents": [it.id], "group": it.lin.get("group") or it.id, "via": op.name,
                   **extra}
            lin["evidence"] = list(it.lin.get("evidence", [])) if getattr(op, "inherit", True) \
                else []
            children.append(Item(rec={kk: vv for kk, vv in crec.items()
                                      if not str(kk).startswith("_")},
                                 id=cid, i=it.i, j=(*it.j, k), attrs=dict(it.attrs), lin=lin))
        return children

    def _schema_check(self, it: Item, pos: int) -> None:
        rec = self.schema.coerce(it.rec)
        it.rec = rec
        errs = self.schema.validate(rec)
        if errs:
            it.reject("schema", pos, Verdict(False, reason="; ".join(errs[:3])))

    def _apply(self, it: Item, pos: int, op: Op) -> list[Item]:
        if isinstance(op, SchemaOp):
            self._schema_check(it, pos)
            return [it]
        ctx = self._op_ctx(it, op)
        try:
            if op.kind == "map":
                out = op.apply(it.view(), ctx)  # type: ignore[attr-defined]
            elif op.kind == "filter":
                out = op.check(it.view(), ctx)  # type: ignore[attr-defined]
            else:
                out = op.expand(it.view(), ctx)  # type: ignore[attr-defined]
            return self._after(it, op, pos, op.kind, out, ctx)
        except Reject as r:
            it.reject(op.name, pos, r.verdict)
            return [it]
        except RunStopped:
            raise
        except Exception as e:
            if self.spec.on_error == "raise":
                raise
            it.reject(op.name, pos, Verdict(False, reason=f"error: {type(e).__name__}: {e}"[:300]))
            return [it]

    def _segment(self, items: list[Item], seg: list[tuple[int, Op]]) -> list[Item]:
        out: list[Item] = []
        for it in items:
            cur = [it]
            for pos, op in seg:
                nxt: list[Item] = []
                for x in cur:
                    if x.status != "a":
                        nxt.append(x)
                        continue
                    t0 = time.perf_counter()
                    nxt += self._apply(x, pos, op)
                    self._timing[op.name] += time.perf_counter() - t0
                cur = nxt
            out += cur
        return out

    def _aggregate(self, k: int, items: list[Item]) -> None:
        pos, agg = self.aggs[k]
        state = self.agg_states[k]
        t0 = time.perf_counter()
        acc = sorted((it for it in items if it.status == "a"), key=lambda x: x.key)
        batch = getattr(agg, "offer_batch", None)
        if callable(batch):
            verdicts = batch(state, [it.view() for it in acc], [it.meta() for it in acc])
        else:
            verdicts = [agg.offer(state, it.view(), it.meta()) for it in acc]
        for it, v in zip(acc, verdicts):
            v = as_verdict(v)
            if not v.ok:
                it.reject(agg.name, pos, v)
            elif v.score is not None or v.reason:
                it.lin.setdefault("verdicts", {})[agg.name] = v.as_dict()
        self._timing[agg.name] += time.perf_counter() - t0

    def _observe(self, k: int, items: list[Item]) -> None:
        pos, agg = self.aggs[k]
        state = self.agg_states[k]
        seen = sorted((it for it in items if it.status == "a" or
                       (it.status == "r" and it.pos > pos)), key=lambda x: x.key)
        batch = getattr(agg, "observe_batch", None)
        if callable(batch):
            batch(state, [it.view() for it in seen], [it.meta() for it in seen])
        else:
            for it in seen:
                agg.observe(state, it.view(), it.meta())

    def process(self, items: list[Item]) -> list[Item]:
        """Per-record ops and streaming aggregates over generated ``items``."""
        for k, (seg, agg) in enumerate(self.segments):
            items = self._segment(items, seg)
            if agg is not None:
                self._aggregate(k, items)
        items.sort(key=lambda x: x.key)
        return items

    # -- chunks -------------------------------------------------------------------------
    def run_chunk(self, c: int) -> ChunkResult:
        cid = self.chunk_id(c)
        if self.store is not None and self.store.chunk_done(cid):
            items = [Item.from_line(d) for d in self.store.read_chunk(cid)]
            if self.aggs:  # aggregates replay live values (blobs decoded)
                live = [Item(rec=self.decode(it.rec), id=it.id, i=it.i, j=it.j, attrs=it.attrs,
                             lin=it.lin, status=it.status, op=it.op, pos=it.pos)
                        for it in items]
                for k in range(len(self.aggs)):
                    self._observe(k, live)
            summary = self.store.chunk_summary(cid) or {}
            return ChunkResult(c, items, summary, cached=True)
        t0 = time.perf_counter()
        items = self._generated(c)
        n_gen = len(items)
        self._timing["generator"] += time.perf_counter() - t0
        items = self.process(items)
        for it in items:
            it.rec = self.encode(it.rec)
        summary = self._summary(c, items, n_gen, time.perf_counter() - t0)
        node_s, node_n = self.ev.take_timings()
        summary["nodes"] = {p or "<root>": [round(t, 6), node_n.get(p, 0)]
                            for p, t in sorted(node_s.items())}
        if self.store is not None:
            self.store.write_chunk(cid, (it.to_line() for it in items), summary)
        return ChunkResult(c, items, summary)

    def _summary(self, c: int, items: list[Item], n_gen: int, seconds: float) -> dict[str, Any]:
        funnel: dict[str, dict[str, int]] = {}
        live = [it for it in items if not (it.status == "r" and it.op == "generator")]
        funnel["generator"] = {"calls": self.cs, "out": len(live),
                               "errors": len(items) - len(live)}
        for pos, op in enumerate(self.ops):
            reached = [it for it in live if it.status == "a" or it.pos >= pos]
            out = sum(1 for it in reached if not (it.status == "r" and it.pos == pos))
            funnel[op.name] = {"in": len(reached), "out": out}
        cells: dict[str, dict[str, list[int]]] = {}
        for name in self.spec.attributes.names:
            a = self.spec.attributes[name]
            if a.kind != "categorical" or len(a.values) > CELL_MAX_VALUES:  # type: ignore
                continue
            d: dict[str, list[int]] = {}
            for it in items:
                if it.status == "r" and it.op == "generator":
                    continue
                key = str(it.attrs.get(name))
                pair = d.setdefault(key, [0, 0])
                pair[0] += 1
                if it.status == "a":
                    pair[1] += 1
            cells[name] = d
        return {"chunk": c, "raw": self.cs, "records": len(items),
                "accepted": sum(1 for it in items if it.status == "a"),
                "funnel": funnel, "cells": cells,
                "seconds": round(seconds, 4), "generated": n_gen}

    # -- main loop -------------------------------------------------------------------------
    def run(self, *, limit: int | None = None) -> dict[str, Any]:
        spec = self.spec
        raw_limit = spec.raw_limit()
        selection = Selection(self.final_ops, spec.size)
        done_chunks: list[ChunkResult] = []
        stop_reason: str | None = None
        raw_this_call = 0
        scheduled_new = 0
        c = 0
        t0 = time.perf_counter()
        workers = spec.workers if isinstance(spec.workers, int) else 1
        if workers > 1:
            self.start_workers(workers)
        try:
            while not selection.done():
                if self.pool is not None:
                    self._prefetch(c, -(-raw_limit // self.cs))
                if c * self.cs >= raw_limit:
                    stop_reason = "raw_limit"
                    selection.relax()
                    break
                cached = self.store is not None and self.store.chunk_done(self.chunk_id(c))
                if limit is not None and not cached and scheduled_new >= limit:
                    stop_reason = "limit"
                    break
                if not cached:
                    scheduled_new += self.cs
                res = self.run_chunk(c)
                done_chunks.append(res)
                if not res.cached:
                    raw_this_call += self.cs
                for it in res.items:
                    if it.status == "a":
                        selection.offer(c, it)
                if self.progress is not None:
                    self._safe_progress(res, selection, time.perf_counter() - t0)
                c += 1
                reason = self.rt.clock.check()
                if reason and not selection.done():
                    stop_reason = "max_time"
                    self.rt.event("max_time", reason=reason)
                    break
        except RunStopped as e:
            stop_reason = e.reason
            self.rt.event(e.reason, reason=str(e.message), hint=e.hint)
        finally:
            self.stop_workers()
        return {"selection": selection, "chunks": done_chunks, "stop_reason": stop_reason,
                "raw_this_call": raw_this_call, "seconds": time.perf_counter() - t0,
                "timing": dict(self._timing)}

    def _safe_progress(self, res: ChunkResult, selection: Selection, elapsed: float) -> None:
        try:
            self.progress.on_chunk(res, selection, elapsed, self.rt.clock)
        except Exception:  # noqa: BLE001, S110 - observability never breaks generation
            pass


# ---------------------------------------------------------------- finalize


def _split_view(item: Item, split: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {"_id": item.id}
    if split.stratify:
        rec[split.stratify] = item.rec.get(split.stratify, item.attrs.get(split.stratify))
    if split.group and split.group not in ("parent", "group"):
        rec[split.group] = item.rec.get(split.group, item.attrs.get(split.group))
    if split.by_length:
        f = split.by_length["field"]
        rec[f] = item.rec.get(f)
    if split.holdout:
        rec["_attrs"] = {k: item.attrs.get(k, item.rec.get(k)) for k in split.holdout}
    return rec


def finalize(spec: Any, runtime: Any, result: dict[str, Any], *,
             complete: bool) -> Any:
    """Write the version (records, rejected, lineage, manifest) and return the dataset."""
    from topologies.experiments.cache import stable_hash
    from topologies.experiments.store import now_iso, write_json_atomic
    from topologies.synth.dataset import SynthDataset
    from topologies.synth.store import ShardWriter

    store = runtime.store
    selection: Selection = result["selection"]
    chunks: list[ChunkResult] = sorted(result["chunks"], key=lambda r: r.index)
    chosen_ids = {rid for _, rid in selection.chosen}
    order = {rid: n for n, (_, rid) in enumerate(selection.chosen)}
    last_chunk = max((c for c, _ in selection.chosen), default=-1)
    if not complete and chunks:
        last_chunk = max(last_chunk, chunks[-1].index)
    used = [r for r in chunks if r.index <= last_chunk]
    split = spec.splits
    # Pass 1: split assignment over the chosen records.
    views: list[tuple[dict, dict]] = [({}, {})] * len(selection.chosen)
    for r in used:
        for it in r.items:
            if it.status == "a" and it.id in chosen_ids:
                views[order[it.id]] = (_split_view(it, split), it.meta())
    assignment = split.assign(views) if views else []
    split_of = {selection.chosen[n][1]: s for n, s in enumerate(assignment)}
    complete_flag = complete and selection.done()
    vid = spec.version_id() if complete_flag else stable_hash(
        {"spec": spec.fingerprint(), "size": spec.size, "partial": len(used),
         "n": len(selection.chosen)}, n=12)
    if store.has_version(vid) and complete_flag:
        return SynthDataset(store, vid)
    counts: Counter = Counter()
    funnel: dict[str, dict[str, int]] = {}
    cells: dict[str, dict[str, list[int]]] = {}
    evidence_mix: Counter = Counter()
    rejected_by: Counter = Counter()
    raw = 0
    blob_refs: list[dict] = []
    with store.new_version(vid) as tmp:
        writers = {s: ShardWriter(tmp / "records" / s, compress=spec.compress)
                   for s in split.names}
        rejected = ShardWriter(tmp / "rejected", compress=spec.compress)
        lineage = ShardWriter(tmp / "lineage", compress=spec.compress)
        # Records are written in the selection order (raw-index order), per split.
        pending_by_split: dict[str, list[tuple[int, Item]]] = defaultdict(list)
        for r in used:
            raw += int(r.summary.get("raw", spec.chunk_size))
            for name, d in (r.summary.get("funnel") or {}).items():
                f = funnel.setdefault(name, {})
                for k, v in d.items():
                    f[k] = f.get(k, 0) + int(v)
            for attr, d in (r.summary.get("cells") or {}).items():
                ca = cells.setdefault(attr, {})
                for val, (g, a) in d.items():
                    pair = ca.setdefault(val, [0, 0])
                    pair[0] += g
                    pair[1] += a
            for it in r.items:
                if it.status == "r":
                    rejected_by[it.op or "?"] += 1
                    rejected.add({"_id": it.id, **it.rec, "_attrs": it.attrs, "_op": it.op,
                                  "_verdict": it.verdict, "_index": it.i})
                    continue
                if it.id in chosen_ids:
                    pending_by_split[split_of[it.id]].append((order[it.id], it))
                elif it.id in selection.dropped:
                    rejected_by[selection.dropped[it.id]] += 1
                    rejected.add({"_id": it.id, **it.rec, "_attrs": it.attrs,
                                  "_op": selection.dropped[it.id],
                                  "_verdict": {"ok": False, "reason": "balance quota full"},
                                  "_index": it.i})
        dups: dict[str, list[int]] = defaultdict(list)
        for rid, k in selection.dups:
            dups[rid].append(k)
        for s, lst in pending_by_split.items():
            for _, it in sorted(lst, key=lambda x: x[0]):
                writers[s].add({"_id": it.id, **it.rec, "_attrs": it.attrs,
                                "_seed": spec.seed, "_index": it.i,
                                "_fanout_index": list(it.j)})
                counts[s] += 1
                lineage.add({"_id": it.id, "split": s, "index": it.i, "sub": list(it.j),
                             **it.lin})
                for k in dups.get(it.id, []):
                    did = hashlib.blake2b(f"{it.id}:dup:{k}".encode(), digest_size=8).hexdigest()
                    writers[s].add({"_id": did, **it.rec, "_attrs": it.attrs})
                    counts[s] += 1
                    lineage.add({"_id": did, "split": s, "index": it.i, "sub": list(it.j),
                                 **it.lin, "parents": [it.id], "via": "balance.upsample",
                                 "group": it.lin.get("group") or it.id})
                ev = it.lin.get("evidence") or []
                from topologies.synth.verdict import strongest

                evidence_mix[strongest(list(ev)) or "none"] += 1
                _collect_blobs(it.rec, blob_refs)
        files = {s: w.close() for s, w in writers.items()}
        rej_files = rejected.close()
        lin_files = lineage.close()
        for ref in blob_refs:
            store.link_blob(ref, tmp)
        import shutil

        if (store.dir / "taxonomy.yaml").is_file():
            shutil.copy2(store.dir / "taxonomy.yaml", tmp / "taxonomy.yaml")
        if (store.dir / "pools").is_dir():
            shutil.copytree(store.dir / "pools", tmp / "pools", dirs_exist_ok=True)
        for name, d in selection_funnel(selection, spec).items():
            funnel[name] = d
        total = sum(counts.values())
        manifest = {
            "format": "topologies-synth-version", "version": 1, "version_id": vid,
            "name": spec.name, "label": spec.version_label, "created_at": now_iso(),
            "parent": None, "complete": complete_flag, "stop_reason": result.get("stop_reason"),
            "spec_hash": spec.fingerprint(), "gen_fingerprint": spec.gen_fingerprint(),
            "stream_fingerprint": spec.stream_fingerprint(), "size": spec.size,
            "total": total, "splits": dict(counts), "files": files,
            "rejected_files": rej_files, "lineage_files": lin_files,
            "raw": raw, "chunks": len(used), "funnel": funnel, "rejected_by": dict(rejected_by),
            "cells": cells, "evidence": dict(evidence_mix),
            "time": {"seconds": round(sum(float(r.summary.get("seconds") or 0.0)
                                          for r in used), 3),
                     "wall_this_run": round(result.get("seconds", 0.0), 3),
                     "ops": {k: round(v, 4) for k, v in (result.get("timing") or {}).items()},
                     "nodes": _node_times(used)},
            "schema": _full_schema(spec.schema, used),
            "schema_inferred": spec.schema is None,
            "attributes": spec.attributes.describe(), "sampler": spec.sampler_spec.describe(),
            "splits_config": split.config(), "generator": spec.generator.node_kind(),
            "pipeline": [op.name for op in spec.pipeline], "seed": spec.seed,
            "library": _library(),
        }
        write_json_atomic(tmp / "manifest.json", manifest)
    store.set_latest(vid)
    store.log_event("version", version_id=vid, total=total, complete=complete_flag,
                    stop_reason=result.get("stop_reason"))
    return SynthDataset(store, vid)


def selection_funnel(selection: Selection, spec: Any) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    dropped = Counter(selection.dropped.values())
    for op in spec.final_ops():
        out[op.name] = {"dropped": dropped.get(op.name, 0)}
    out["select"] = {"selected": len(selection.chosen), "target": selection.target,
                     "surplus": selection.surplus}
    return out


def _collect_blobs(obj: Any, out: list[dict]) -> None:
    if isinstance(obj, dict):
        if "$blob" in obj:
            out.append(obj)
            return
        for v in obj.values():
            _collect_blobs(v, out)
    elif isinstance(obj, list):
        for v in obj[:1000]:
            _collect_blobs(v, out)


def _full_schema(declared: Any, chunks: list[ChunkResult]) -> dict[str, Any] | None:
    """The declared schema plus inferred types for the fields it does not declare."""
    from topologies.synth.schema import Schema

    rows = [it.rec for r in chunks for it in r.items if it.status == "a"][:500]
    inferred = None
    if rows:
        try:
            inferred = Schema.infer(rows)
        except Exception:  # noqa: BLE001
            inferred = None
    if declared is None:
        return inferred.to_config() if inferred is not None else None
    return declared.merged(inferred).to_config()


def _node_times(chunks: list[ChunkResult]) -> dict[str, list[float]]:
    """Per-node ``[seconds, calls]`` summed over chunks (kept out of lineage, which stays
    byte-identical between runs)."""
    out: dict[str, list[float]] = {}
    for r in chunks:
        for p, (t, n) in (r.summary.get("nodes") or {}).items():
            cur = out.setdefault(p, [0.0, 0])
            cur[0] = round(cur[0] + float(t), 6)
            cur[1] += int(n)
    return out


def _library() -> dict[str, Any]:
    from topologies.experiments.trial import environment

    env = environment()
    return {k: env.get(k) for k in ("topologies", "torch", "python", "platform", "git")}


# ---------------------------------------------------------------- entry point


class Progress:
    """Line / live progress for runs (display failures are swallowed)."""

    def __init__(self, spec: Any, mode: str, stream: Any = None) -> None:
        self.spec = spec
        self.stream = stream if stream is not None else sys.stdout
        from topologies.experiments.render.live import _is_tty

        self.tty = _is_tty(self.stream)
        if mode == "auto":
            mode = "live" if self.tty else "lines"
        if mode == "live" and not self.tty:
            mode = "lines"
        self.mode = mode
        self.last = 0.0
        self.rates: list[float] = []
        self.recent: list[str] = []
        self._line: Any = None
        if mode == "live":
            from topologies.visualization.terminal.progress import ProgressLine

            self._line = ProgressLine(self.stream)

    def on_chunk(self, res: ChunkResult, selection: Selection, elapsed: float,
                 clock: Any) -> None:
        if self.mode == "none":
            return
        from topologies.synth.report import live_line

        acc = res.summary.get("accepted", 0) / max(1, res.summary.get("raw", 1))
        self.rates.append(acc)
        for it in res.items:
            if it.status == "r" and it.verdict:
                self.recent = ([f"{it.op}: {it.verdict.get('reason', '')[:80]}"]
                               + self.recent)[:3]
        now = time.monotonic()
        if self.mode == "lines" and now - self.last < 2.0 and not selection.done():
            return
        self.last = now
        if self.mode == "live" and self._line is not None:
            from topologies.synth.report import progress_line

            self._line.update(progress_line(self.spec, res, selection, elapsed, clock,
                                            self.rates, self._line.width()),
                              force=selection.done())
            return
        line = live_line(self.spec, res, selection, elapsed, clock, self.rates, self.recent,
                         tty=self.tty)
        self.stream.write(line.splitlines()[0] + "\n")
        self.stream.flush()

    def finish(self, ds: Any, result: dict[str, Any]) -> None:
        if self.mode == "none":
            return
        try:
            from topologies.synth.report import run_summary

            summary = run_summary(ds, result)
            if self.mode == "live" and self._line is not None:
                self._line.close(summary.splitlines())
                return
            self.stream.write(summary + "\n")
            self.stream.flush()
        except Exception:  # noqa: BLE001, S110 - progress output is best-effort
            pass


def run_spec(spec: Any, *, limit: int | None = None, rerun: str | None = None,
             monitor: str | None = None, progress: Any = None) -> Any:
    """``Spec.run`` implementation (see module docstring)."""
    from topologies.synth.runtime import Runtime

    store = spec.store()
    store.dir.mkdir(parents=True, exist_ok=True)
    with store.lock():
        if rerun is None and store.has_version(spec.version_id()):
            from topologies.synth.dataset import SynthDataset

            store.set_latest(spec.version_id())
            return SynthDataset(store, spec.version_id())
        store.write_spec(spec.describe())
        if rerun == "all":
            store.clear_chunks(prefix=spec.stream_fingerprint()[:16])
        rt = Runtime(spec, store)
        mon = monitor or spec.monitor
        prog = progress if progress is not None else Progress(spec, mon)
        rt.event("start", spec_hash=spec.fingerprint(), size=spec.size, limit=limit)
        setup_parts(spec, rt)
        ex = Executor(spec, rt, progress=prog)
        try:
            result = ex.run(limit=limit)
        except KeyboardInterrupt:
            rt.event("interrupted")
            raise
        complete = result["stop_reason"] is None and result["selection"].done()
        if result["stop_reason"] == "raw_limit" and result["selection"].done():
            complete = True
        ds = finalize(spec, rt, result, complete=complete)
        if result["stop_reason"] == "raw_limit" and not result["selection"].done():
            rt.event("raw_limit", selected=len(result["selection"].chosen), target=spec.size)
        _finish(prog, ds, result)
        return ds


def setup_parts(spec: Any, rt: Any) -> None:
    """Attribute, node and op ``setup`` hooks before the first record."""
    from topologies.synth.core import iter_nodes

    for a in spec.attributes.attrs.values():
        setup = getattr(a, "setup", None)
        if callable(setup):
            setup(rt)
    for n in iter_nodes(spec.generator):
        n.setup(rt)
    for op in spec.pipeline:
        op.setup(rt)


def _finish(prog: Any, ds: Any, result: dict[str, Any]) -> None:
    fin = getattr(prog, "finish", None)
    if callable(fin):
        try:
            fin(ds, result)
        except Exception:  # noqa: BLE001, S110 - progress output is best-effort
            pass


def estimate_acceptance(summaries: list[dict[str, Any]]) -> float | None:
    raw = sum(int(s.get("raw", 0)) for s in summaries)
    acc = sum(int(s.get("accepted", 0)) for s in summaries)
    return acc / raw if raw else None


def eta_seconds(done: int, target: int, elapsed: float) -> float | None:
    if done <= 0 or elapsed <= 0:
        return None
    rate = done / elapsed
    return max(0.0, (target - done) / rate) if math.isfinite(rate) and rate > 0 else None


__all__ = ["ChunkResult", "Executor", "Item", "Progress", "Selection", "estimate_acceptance",
           "eta_seconds", "finalize", "record_id", "run_spec"]


# ---------------------------------------------------------------- ops outside a run


def _null_generator() -> Any:
    from topologies.synth.core import Const

    return Const({}, name="records")


def run_ops_on_records(records: Sequence[dict], ops: Sequence[Any], *, name: str = "ops",
                       seed: int = 0, on_error: str = "skip", root: Any = None,
                       return_rejected: bool = False) -> Any:
    """Apply ``ops`` (per-record and streaming aggregates) to existing records.

    Used by relabeling, label-issue pruning and the closed loop. Records keep
    their ``_id`` / ``_attrs``; rejected records carry ``_op`` and ``_verdict``.
    """
    from types import SimpleNamespace

    from topologies.synth.attributes import AttributeSpace
    from topologies.synth.ops.base import as_op
    from topologies.synth.runtime import Runtime

    op_list = [as_op(o) for o in ops]
    fake = SimpleNamespace(
        name=name, generator=_null_generator(), gen_fingerprint=lambda: "records",
        stream_fingerprint=lambda: "records", schema=None, stream_ops=lambda: op_list,
        final_ops=list, seed=seed, on_error=on_error, chunk_size=max(1, len(records)),
        attributes=AttributeSpace({}), root=root, sampler=None, max_time=None)
    rt = Runtime(fake, None)
    for op in op_list:
        op.setup(rt)
    ex = Executor(fake, rt)
    items = []
    for n, r in enumerate(records):
        rid = r.get("_id") or record_id("records", n, (0,))
        items.append(Item(rec={k: v for k, v in r.items() if not str(k).startswith("_")},
                          id=rid, i=n, j=(0,), attrs=dict(r.get("_attrs") or {})))
    items = ex.process(items)
    ok = [{"_id": it.id, **it.rec, "_attrs": it.attrs,
           **({"_lineage": it.lin} if it.lin.get("verdicts") else {})}
          for it in items if it.status == "a"]
    bad = [{"_id": it.id, **it.rec, "_attrs": it.attrs, "_op": it.op,
            "_verdict": it.verdict} for it in items if it.status == "r"]
    return (ok, bad) if return_rejected else ok
