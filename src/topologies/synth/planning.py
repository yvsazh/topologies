"""``spec.plan()``: validate, preview and measure before generating.

The plan validates the spec, samples attribute assignments, checks pairwise
coverage of the design, runs a *pilot* (the first ``pilot`` raw records through
the whole pipeline, nothing written) and reports:

* pass rates per pipeline op and the measured acceptance rate;
* measured throughput (raw records/s) and bytes per stored record, projected
  to the full ``size`` (time and disk);
* a **determinism self-check**: a few indices are generated again, in a
  different order and chunking, and their stored bytes compared;
* a preview of accepted and rejected pilot records.

    plan = spec.plan(pilot=200)
    plan.ok, plan.records_per_s, plan.projected_seconds, plan.projected_bytes
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from topologies.synth.errors import SynthError


@dataclass
class Plan:
    """Result of ``spec.plan()``; ``str(plan)`` renders it, ``plan.ok`` gates a run."""

    spec: Any
    samples: list[dict[str, Any]] = field(default_factory=list)
    raw_estimate: int = 0
    acceptance: float | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    pilot: dict[str, Any] | None = None
    coverage: dict[str, Any] = field(default_factory=dict)
    deterministic: bool | None = None
    records_per_s: float | None = None
    bytes_per_record: float | None = None
    projected_seconds: float | None = None
    projected_bytes: float | None = None
    preview: int = 3
    tree: str = ""

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise SynthError("The spec plan has errors:\n- " + "\n- ".join(self.errors),
                             stage="plan")

    def __str__(self) -> str:
        from topologies.synth.report import format_records

        s = self.spec
        lines = [f"plan  {s.name}  (spec {s.fingerprint()}, version {s.version_id()})",
                 f"  generator: {s.generator.node_kind()}",
                 (f"  attributes: {', '.join(f'{n}:{a.kind}' for n, a in s.attributes.attrs.items()) or '—'}"
                  f"  · sampler {s.sampler_spec.name}"),
                 f"  pipeline: {' → '.join(op.name for op in s.pipeline) or '—'}",
                 *(["  node tree:"] + ["    " + ln for ln in self.tree.splitlines()[:40]]
                   if self.tree else []),
                 f"  splits: {s.splits.describe()}",
                 f"  size: {s.size:,} accepted · raw estimate {self.raw_estimate:,}"
                 + ("" if self.acceptance is None else
                     f" (acceptance {self.acceptance:.0%})" if self.acceptance <= 1 else
                     f" (yield {self.acceptance:.2f} records per raw index)")]
        if self.records_per_s:
            lines.append(f"  throughput: {self.records_per_s:,.0f} raw records/s · "
                         f"{_fmt_bytes(self.bytes_per_record)}/record → projected "
                         f"{_fmt_s(self.projected_seconds)}, {_fmt_bytes(self.projected_bytes)}")
        if s.max_time:
            lines.append(f"  max_time: {_fmt_s(s.max_time)}")
        if self.deterministic is not None:
            lines.append("  determinism self-check: " + ("ok (same bytes in a different order "
                                                         "and chunking)" if self.deterministic
                                                         else "FAILED"))
        if self.coverage:
            c = self.coverage
            lines.append(f"  pair coverage per block: {c.get('covered')}/{c.get('pairs')} "
                         "attribute-value pairs")
        if self.samples and any(self.samples):
            lines.append("  sample attribute assignments:")
            for a in self.samples[:5]:
                shown = ", ".join(f"{k}={_short(v)}" for k, v in a.items())
                lines.append(f"    · {shown}")
        if self.pilot is not None:
            pl = self.pilot
            lines.append(f"  pilot: {pl['raw']} raw → {pl['accepted']} accepted "
                         f"({pl['acceptance']:.0%}) in {_fmt_s(pl['seconds'])}")
            for name, d in pl.get("funnel", {}).items():
                if d.get("in"):
                    lines.append(f"    {name:<24} {d['out']}/{d['in']} pass "
                                 f"({d['out'] / d['in']:.0%})")
            if pl.get("examples") and self.preview:
                from topologies.synth.previews import preview_records

                lines.append("  pilot samples:")
                lines += ["    " + ln for ln in preview_records(
                    pl["examples"][:self.preview], s.schema).splitlines()]
            if pl.get("rejections") and self.preview:
                lines.append(format_records(pl["rejections"][:self.preview],
                                            title="  pilot rejections:"))
        for w in self.warnings:
            lines.append(f"  ! {w}")
        for e in self.errors:
            lines.append(f"  ✗ {e}")
        lines.append("  ok" if self.ok else "  plan has errors: fix them before run()")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return str(self)


def _short(v: Any, n: int = 40) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _fmt_s(s: float | None) -> str:
    from topologies.experiments.table import fmt_seconds

    return fmt_seconds(s) if s else "0s"


def _fmt_bytes(b: float | None) -> str:
    if b is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:,.0f} {unit}" if unit == "B" else f"{b:,.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def make_plan(spec: Any, *, pilot: int = 200, preview: int = 3) -> Plan:
    from topologies.synth.rng import SynthRandom, derive_seed

    plan = Plan(spec, preview=int(preview))
    try:
        from topologies.synth.introspect import describe

        plan.tree = describe(spec.generator, echo=False)
    except Exception as e:  # noqa: BLE001
        plan.errors.append(f"node graph: {type(e).__name__}: {e}")
    try:
        for i in range(min(8, spec.size)):
            base = spec.sampler.assign(i)
            plan.samples.append(spec.attributes.complete(
                base, SynthRandom(derive_seed(spec.seed, i, "attrs")), i))
    except SynthError as e:
        plan.errors.append(f"attributes: {e.message} ({e.hint or ''})")
    try:
        cats = [a for a in spec.attributes.categorical
                if len(spec.attributes.values(a)) <= 24]
        if len(cats) >= 2:
            from topologies.synth.sampling import pair_coverage

            n = spec.sampler.block_size()
            rows = [spec.sampler.assign(i) for i in range(n)]
            plan.coverage = pair_coverage(rows, spec.attributes, cats)
            if plan.coverage.get("missing"):
                plan.warnings.append(
                    f"{len(plan.coverage['missing'])} attribute-value pairs never appear in a "
                    "block; use sampler='pairwise' (or 'auto' with >= 4 categoricals).")
    except Exception:  # noqa: BLE001, S110 - coverage is informational
        pass
    if pilot:
        try:
            plan.pilot = run_pilot(spec, int(pilot))
            plan.acceptance = plan.pilot["acceptance"]
            secs = plan.pilot["seconds"]
            plan.records_per_s = plan.pilot["raw"] / secs if secs > 0 else None
            plan.bytes_per_record = plan.pilot.get("bytes_per_record")
            plan.deterministic = plan.pilot.get("deterministic")
            if plan.deterministic is False:
                plan.errors.append("determinism self-check failed: the same indices produced "
                                   "different bytes in a different order / chunking. A node "
                                   "uses global randomness (random.random, torch.rand without "
                                   "generator=ctx.rng.torch) or mutable shared state.")
        except SynthError as e:
            plan.errors.append(f"pilot failed: {e.message}" + (f" ({e.hint})" if e.hint else ""))
        except Exception as e:  # noqa: BLE001
            plan.errors.append(f"pilot failed: {type(e).__name__}: {e}")
    acc = plan.acceptance if plan.acceptance else 1.0
    plan.raw_estimate = min(int(spec.size / max(acc, 1e-3)), spec.raw_limit())
    if plan.records_per_s:
        plan.projected_seconds = plan.raw_estimate / plan.records_per_s
    if plan.bytes_per_record is not None:
        plan.projected_bytes = plan.bytes_per_record * spec.size
    if plan.pilot is not None and plan.acceptance is not None and plan.acceptance < 0.05:
        plan.warnings.append(f"only {plan.acceptance:.0%} of pilot records pass the pipeline: "
                             "check pilot rejections and loosen filters.")
    return plan


def _stored_bytes(items: list[Any]) -> list[str]:
    from topologies.synth.store import dumps

    return [dumps({"rec": it.rec, "attrs": it.attrs, "s": it.status}) for it in items]


def run_pilot(spec: Any, n: int) -> dict[str, Any]:
    """Run the first ``n`` raw records through the whole pipeline (no files written)."""
    from topologies.synth.pipeline import Executor, setup_parts
    from topologies.synth.runtime import Runtime

    rt = Runtime(spec, None)
    setup_parts(spec, rt)
    ex = Executor(spec, rt)
    ex.cs = n
    t0 = time.perf_counter()
    res = ex.run_chunk(0)
    seconds = time.perf_counter() - t0
    acc = [it for it in res.items if it.status == "a"]
    rej = [it for it in res.items if it.status == "r"]
    lines = _stored_bytes(acc)
    bytes_per = sum(len(x.encode("utf-8")) for x in lines) / len(lines) if lines else None
    # Determinism: a few indices again, reversed and one at a time, in a fresh executor.
    deterministic: bool | None = None
    k = min(n, 8)
    if k:
        ex2 = Executor(spec, Runtime(spec, None))
        again = [x for i in reversed(range(k)) for x in ex2.generate(i)]
        again = ex2.process(again)
        for it in again:
            it.rec = ex2.encode(it.rec)
        first = [it for it in res.items if it.i < k]
        deterministic = _stored_bytes(sorted(first, key=lambda x: x.key)) == \
            _stored_bytes(sorted(again, key=lambda x: x.key))
    return {"raw": n, "accepted": len(acc), "acceptance": len(acc) / max(1, n),
            "funnel": res.summary.get("funnel", {}), "seconds": seconds,
            "bytes_per_record": bytes_per, "deterministic": deterministic,
            "examples": [{"_id": it.id, **it.rec, "_attrs": it.attrs} for it in acc[:5]],
            "rejections": [{"_id": it.id, **it.rec, "_attrs": it.attrs, "_op": it.op,
                            "_verdict": it.verdict} for it in rej[:5]]}


__all__ = ["Plan", "make_plan", "run_pilot"]
