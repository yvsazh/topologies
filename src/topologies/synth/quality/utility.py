"""Downstream utility - the headline metric: train on synthetic, test on real (TSTR).

    res = ds.evaluate(real_eval, model="TransformerClassifier", size="small", seeds=3)
    res.show()            # TSTR ± CI, TRTR and the ratio (with real_train=...), majority
                          # baseline, per-class report, confusion, per-slice errors on real data
    synth.scaling_curve(spec, sizes=[1000, 4000, 16000], real_eval=real_eval)   # + power law
    synth.mix_curve(ds, real_train, ratios=[0, .25, .5, .75, 1], real_eval=real_eval)
    synth.ablate(spec, ops=["dedup.near", "verify"], real_eval=real_eval)    # op worth it?

All of these run an ``Experiment`` over a ``SynthTask`` (resumable, cached,
parallelizable, reported like any experiment); the result objects add the
synthetic-data views. ``evaluate`` refuses to run when the real eval set
overlaps the synthetic training data above ``contamination_threshold``, unless
``allow_contamination=True``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from topologies.experiments.table import Table
from topologies.synth.errors import SynthError
from topologies.synth.records import as_records


def _root(root: Any, ds_or_spec: Any) -> Path:
    if root is not None:
        return Path(root)
    store = ds_or_spec.store() if callable(ds_or_spec.store) else ds_or_spec.store
    return Path(store.root) / ".experiments"


def _experiment(name: str, task: Any, *, axes: dict[str, Any], model: Any, size: Any,
                train_for: Any, seeds: Any, train: dict | None, root: Path, monitor: str,
                **kw: Any) -> Any:
    from topologies.experiments.core import Experiment

    tr = {"batch_size": 32, **dict(train or {})}
    return Experiment(name, task=task, axes=axes, model=model, size=size, train_for=train_for,
                      seeds=seeds, train=tr, root=root, monitor=monitor, **kw)


def _summary(rs: Any, labels: dict[str, Any]) -> dict[str, Any] | None:
    metric = rs.goal[0]
    for s in rs.summaries(metric):
        if all(str(s.labels.get(k)) == str(v) for k, v in labels.items()):
            return {"mean": s.mean, "std": s.std, "ci": s.ci, "n": s.n,
                    "synth_seconds": s.cost("synth_seconds")}
    return None


def check_contamination(train: Sequence[dict], real: Sequence[dict], *, threshold: float,
                        allow: bool) -> dict[str, Any]:
    from topologies.synth.ops.contamination import overlap

    found = overlap(real, train)
    share = len(found) / max(1, len(real))
    info = {"overlapping": len(found), "share": share, "examples": found[:5]}
    if share > threshold and not allow:
        raise SynthError(f"{len(found)} of {len(real)} real eval records ({share:.1%}) overlap "
                         "the synthetic training data: TSTR would be inflated.",
                         stage="evaluate",
                         hint="Drop them with synth.ops.contamination.check(real_eval) in the "
                              "pipeline, or pass allow_contamination=True to evaluate anyway.")
    return info


class EvalResult:
    """TSTR result (see module docstring); ``results`` is the experiment ``ResultSet``."""

    def __init__(self, results: Any, *, goal: str, tstr: dict | None, trtr: dict | None,
                 majority: float | None, per_class: dict | None, confusion: list | None,
                 classes: list | None, slices: list[dict], tagger: dict | None,
                 contamination: dict) -> None:
        self.results, self.goal, self.tstr, self.trtr = results, goal, tstr, trtr
        self.majority, self.per_class, self.confusion = majority, per_class, confusion
        self.classes, self.slices, self.tagger = classes, slices, tagger
        self.contamination = contamination

    @property
    def ratio(self) -> float | None:
        if not self.tstr or not self.trtr or not self.trtr.get("mean"):
            return None
        return self.tstr["mean"] / self.trtr["mean"]

    def summary_table(self) -> Table:
        rows = []
        for name, d in (("TSTR (train synthetic, test real)", self.tstr),
                        ("TRTR (train real, test real)", self.trtr)):
            if d:
                ci = d.get("ci")
                rows.append([name, d["mean"], d.get("std"), f"[{ci[0]:.4f}, {ci[1]:.4f}]"
                             if ci else "—", d.get("n")])
        if self.majority is not None:
            rows.append(["majority-class baseline", self.majority, None, "—", None])
        if self.ratio is not None:
            rows.append(["TSTR / TRTR", self.ratio, None, "—", None])
        return Table(["", self.goal, "std", "95% CI", "seeds"], rows,
                     meta={"title": "utility on real data"})

    def slice_table(self, n: int = 12) -> Table:
        rows = [[s["attribute"], s["value"], s["n"], s["error_rate"], s["lift"]]
                for s in self.slices[:n]]
        return Table(["attribute", "value", "n", "error rate", "vs overall"], rows,
                     meta={"title": "worst real-data slices" + (
                         " (attributes tagged by model)" if self.tagger else "")})

    def confusion_table(self) -> Table | None:
        if not self.confusion or not self.classes:
            return None
        rows = [[c, *r] for c, r in zip(self.classes, self.confusion)]
        return Table(["true \\ pred", *self.classes], rows,
                     meta={"title": "confusion matrix (real eval, first seed)",
                           "heatmap": {"values": self.confusion, "mode": "max",
                                       "row_offset": 0, "col_offset": 1}})

    def show(self) -> None:
        print(self.summary_table())
        if self.per_class:
            rows = [[k, v["precision"], v["recall"], v["f1"], v.get("support")]
                    for k, v in self.per_class.items() if isinstance(v, dict)]
            print(Table(["class", "precision", "recall", "f1", "support"], rows,
                        meta={"title": "per-class report (real eval)"}))
        ct = self.confusion_table()
        if ct is not None:
            print(ct)
        if self.slices:
            print(self.slice_table())
        if self.contamination.get("overlapping"):
            print(f"! {self.contamination['overlapping']} real eval records overlap the "
                  "synthetic training data")

    def to_json(self) -> dict[str, Any]:
        from topologies.experiments.store import jsonable

        return jsonable({"goal": self.goal, "tstr": self.tstr, "trtr": self.trtr,
                         "ratio": self.ratio, "majority": self.majority,
                         "per_class": self.per_class, "confusion": self.confusion,
                         "classes": self.classes, "slices": self.slices,
                         "tagger": self.tagger, "contamination": self.contamination})

    def __repr__(self) -> str:
        t = self.tstr or {}
        return (f"EvalResult({self.goal}: TSTR {t.get('mean')}" +
                (f", TRTR {self.trtr.get('mean')}" if self.trtr else "") + ")")


def evaluate(ds: Any, real_eval: Any, *, model: Any = "TransformerClassifier",
             size: Any = "small", train_for: Any = "5 epochs", seeds: Any = 3,
             recipe: str | None = None, tokenizer: Any = "word:8k", max_len: int = 128,
             real_train: Any = None, baselines: Sequence[str] = ("real-only", "majority"),
             train: dict | None = None, name: str | None = None, root: Any = None,
             tagger: bool = True,
             allow_contamination: bool = False, contamination_threshold: float = 0.02,
             monitor: str = "none", **exp_kw: Any) -> EvalResult:
    """TSTR of ``ds`` (a ``SynthDataset``) on ``real_eval`` (see module docstring)."""
    from topologies.experiments.space import Variant
    from topologies.synth.tasks import SynthTask

    real = as_records(real_eval, what="real_eval")
    real_tr = as_records(real_train, what="real_train") if real_train is not None else []
    train_recs = ds.to_records("train" if "train" in ds.splits else None)
    contamination = check_contamination(train_recs, real, threshold=contamination_threshold,
                                        allow=allow_contamination)
    task = SynthTask(ds, eval=real, real_train=real_tr or None, recipe=recipe,
                     tokenizer=tokenizer, max_len=max_len)
    axes: dict[str, Any] = {}
    if real_tr and "real-only" in baselines:
        axes["data"] = [Variant("synthetic", None, **{"synth.mix.real": 0.0}),
                        Variant("real-only", None, **{"synth.mix.real": 1.0})]
    exp = _experiment(name or f"tstr-{ds.name}-{ds.version_id}", task, axes=axes,
                      model=model, size=size, train_for=train_for, seeds=seeds, train=train,
                      root=_root(root, ds), monitor=monitor, **exp_kw)
    rs = exp.run()
    goal = rs.goal[0]
    tstr = _summary(rs, {"data": "synthetic"} if axes else {})
    trtr = _summary(rs, {"data": "real-only"}) if axes else None
    majority = None
    lf = (ds.schema.label_field() if ds.schema is not None else None)
    if "majority" in baselines and lf and real and lf in real[0]:
        top = Counter(str(r.get(lf)) for r in train_recs).most_common(1)
        if top:
            majority = sum(1 for r in real if str(r.get(lf)) == top[0][0]) / len(real)
    syn_trials = [r for r in rs.records if r.status == "completed" and
                  (not axes or str(r.labels.get("data")) == "synthetic")]
    first = syn_trials[0] if syn_trials else None
    per_class = (first.extras or {}).get("per_class_report") if first else None
    confusion = (first.extras or {}).get("confusion_matrix") if first else None
    classes = (first.extras or {}).get("classes") if first else None
    slices: list[dict] = []
    tag_info = None
    if syn_trials and classes and lf:
        from topologies.synth.quality.slices import AttributeTagger, slice_errors

        tagged = real
        if tagger and not any(r.get("_attrs") for r in real):
            tg = AttributeTagger().fit(train_recs)
            tagged = tg.tag(real)
            tag_info = {"kind": "model", "accuracy": tg.accuracy}
        votes = [0] * len(real)
        for t in syn_trials:
            preds = (t.extras or {}).get("predictions") or []
            for i, p in enumerate(preds[: len(real)]):
                votes[i] += int(classes[p] == str(real[i].get(lf)))
        correct = [v * 2 > len(syn_trials) for v in votes]
        slices = slice_errors(tagged, correct)
    return EvalResult(rs, goal=goal, tstr=tstr, trtr=trtr, majority=majority,
                      per_class=per_class, confusion=confusion, classes=classes,
                      slices=slices, tagger=tag_info, contamination=contamination)


# ---------------------------------------------------------------- curves


def fit_power_law(sizes: Sequence[float], errors: Sequence[float]) -> dict[str, float] | None:
    """``err(n) = a * n^-b + c`` by grid search over ``c`` and log-log least squares."""
    pts = [(float(n), float(e)) for n, e in zip(sizes, errors) if n > 0 and e is not None]
    if len(pts) < 3:
        return None
    best = None
    lo = min(e for _, e in pts)
    for k in range(50):
        c = lo * k / 50
        xs = [math.log(n) for n, e in pts if e - c > 1e-12]
        ys = [math.log(e - c) for n, e in pts if e - c > 1e-12]
        if len(xs) < 3:
            continue
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs) or 1e-12
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        icpt = my - slope * mx
        res = sum((y - (icpt + slope * x)) ** 2 for x, y in zip(xs, ys))
        tot = sum((y - my) ** 2 for y in ys) or 1e-12
        r2 = 1 - res / tot
        if best is None or r2 > best["r2"]:
            best = {"a": math.exp(icpt), "b": -slope, "c": c, "r2": r2}
    return best


class Curve:
    def __init__(self, results: Any, axis: str, rows: list[dict[str, Any]], goal: tuple,
                 fit: dict[str, float] | None = None) -> None:
        self.results, self.axis, self.rows, self.goal, self.fit = results, axis, rows, goal, fit

    def table(self) -> Table:
        rows = [[r["x"], r["mean"], r["std"], r.get("synth_seconds")] for r in self.rows]
        return Table([self.axis, self.goal[0], "std", "gen s"], rows,
                     meta={"title": f"{self.goal[0]} vs {self.axis}"})

    def size_for(self, target: float) -> float | None:
        """Dataset size the power-law fit predicts for ``target`` (accuracy-like metrics
        are converted to error ``1 - target``)."""
        f = self.fit
        if not f or f["b"] <= 0:
            return None
        err = 1 - target if self.goal[1] == "max" else target
        if err <= f["c"]:
            return math.inf
        return ((err - f["c"]) / f["a"]) ** (-1 / f["b"])

    def plateau(self) -> bool:
        """The last doubling of data gained less than one point."""
        if len(self.rows) < 2:
            return False
        a, b = self.rows[-2]["mean"], self.rows[-1]["mean"]
        if a is None or b is None:
            return False
        gain = (b - a) if self.goal[1] == "max" else (a - b)
        return gain < 0.01

    def show(self) -> None:
        print(self.table())
        if self.fit:
            f = self.fit
            print(f"power-law fit: error = {f['a']:.3g} * n^-{f['b']:.3f} + {f['c']:.3g} "
                  f"(R² {f['r2']:.3f})")

    def __repr__(self) -> str:
        return str(self.table())


def _curve_rows(rs: Any, axis: str) -> list[dict[str, Any]]:
    out = []
    for s in rs.summaries(rs.goal[0]):
        x = s.labels.get(axis)
        try:
            xv: Any = float(x)
        except (TypeError, ValueError):
            xv = x
        out.append({"x": xv, "mean": s.mean, "std": s.std, "n": s.n,
                    "synth_seconds": s.cost("synth_seconds")})
    return sorted(out, key=lambda r: (isinstance(r["x"], str), r["x"]))


def scaling_curve(spec: Any, sizes: Sequence[int], *, real_eval: Any, model: Any =
                  "TransformerClassifier", size: Any = "tiny", train_for: Any = "5 epochs",
                  seeds: Any = 1, tokenizer: Any = "word:8k", max_len: int = 128,
                  recipe: str | None = None, train: dict | None = None, root: Any = None,
                  monitor: str = "none", name: str | None = None, **kw: Any) -> Curve:
    """Utility vs synthetic dataset size, with a fitted power law."""
    from topologies.synth.tasks import SynthTask

    task = SynthTask(spec, eval=as_records(real_eval), tokenizer=tokenizer, max_len=max_len,
                     recipe=recipe)
    exp = _experiment(name or f"scaling-{spec.name}", task,
                      axes={"synth.size": [int(s) for s in sizes]}, model=model, size=size,
                      train_for=train_for, seeds=seeds, train=train, root=_root(root, spec),
                      monitor=monitor, **kw)
    rs = exp.run()
    rows = _curve_rows(rs, "synth.size")
    errors = [(1 - r["mean"]) if rs.goal[1] == "max" else r["mean"] for r in rows
              if r["mean"] is not None]
    fit = fit_power_law([r["x"] for r in rows if r["mean"] is not None], errors)
    return Curve(rs, "synth.size", rows, rs.goal, fit)


def mix_curve(source: Any, real_train: Any, *, real_eval: Any,
              ratios: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0), model: Any =
              "TransformerClassifier", size: Any = "tiny", train_for: Any = "5 epochs",
              seeds: Any = 1, tokenizer: Any = "word:8k", max_len: int = 128,
              recipe: str | None = None, train: dict | None = None, root: Any = None,
              monitor: str = "none", name: str | None = None, **kw: Any) -> Curve:
    """Utility vs the share of real records mixed into the synthetic training data."""
    from topologies.synth.tasks import SynthTask

    task = SynthTask(source, eval=as_records(real_eval), real_train=as_records(real_train),
                     tokenizer=tokenizer, max_len=max_len, recipe=recipe)
    nm = getattr(source, "name", "data")
    exp = _experiment(name or f"mix-{nm}", task, axes={"synth.mix.real": list(ratios)},
                      model=model, size=size, train_for=train_for, seeds=seeds, train=train,
                      root=_root(root, source), monitor=monitor, **kw)
    rs = exp.run()
    return Curve(rs, "synth.mix.real", _curve_rows(rs, "synth.mix.real"), rs.goal)


def ablate(spec: Any, ops: Sequence[str], *, real_eval: Any, model: Any =
           "TransformerClassifier", size: Any = "tiny", train_for: Any = "5 epochs",
           seeds: Any = 1, tokenizer: Any = "word:8k", max_len: int = 128,
           recipe: str | None = None, train: dict | None = None, root: Any = None,
           monitor: str = "none", name: str | None = None, **kw: Any) -> Curve:
    """Utility (and generation time) with each pipeline op removed vs the full pipeline."""
    from topologies.experiments.space import Variant
    from topologies.synth.tasks import SynthTask

    for op in ops:
        spec.with_(**{f"ops.{op}": None})  # validates the op name (with hints)
    task = SynthTask(spec, eval=as_records(real_eval), tokenizer=tokenizer, max_len=max_len,
                     recipe=recipe)
    variants = [Variant("full", None)] + [Variant(f"-{op}", None, **{f"synth.ops.{op}": None})
                                          for op in ops]
    exp = _experiment(name or f"ablate-{spec.name}", task, axes={"ablation": variants},
                      model=model, size=size, train_for=train_for, seeds=seeds, train=train,
                      root=_root(root, spec), monitor=monitor, **kw)
    rs = exp.run()
    return Curve(rs, "ablation", _curve_rows(rs, "ablation"), rs.goal)


__all__ = ["Curve", "EvalResult", "ablate", "check_contamination", "evaluate",
           "fit_power_law", "mix_curve", "scaling_curve"]
