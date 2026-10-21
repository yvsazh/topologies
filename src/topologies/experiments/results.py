"""Results: ``TrialRecord`` (one ``result.json``), ``ConfigSummary`` (seeds aggregated),
``Table`` (one value object for every view) and ``ResultSet``.

    res = topologies.experiments.load("arch-x-tokenizer")
    res.show()                                  # leaderboard, pivot, marginals, insights
    res.table(top=10)                           # leaderboard Table
    res.best()[0].labels                        # {"model": ..., "tokenizer": ...}
    res.filter(tokenizer=["bpe:1k", "bpe:4k"]).pivot("model", "tokenizer")
    res.to_csv("results.csv"); res.to_pandas()  # flat exports

Records are plain data, so a remote tracking exporter only needs
``to_records()`` (one flat dict per trial).
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from topologies.experiments import stats as st
from topologies.experiments.metrics import MetricInfo, metric_info
from topologies.experiments.table import Table, fmt_params, fmt_rate, fmt_seconds

# ---------------------------------------------------------------- records


@dataclass
class TrialRecord:
    """One trial's ``result.json`` as attributes (``raw`` keeps the dict)."""

    raw: dict[str, Any]
    dir: Path | None = None

    def __getattr__(self, name: str) -> Any:
        if name in ("raw", "dir") or name.startswith("__"):
            raise AttributeError(name)
        raw = self.__dict__.get("raw") or {}
        if name in raw:
            return raw[name]
        if name in ("final", "best", "cost", "extras", "labels", "config", "goal"):
            return {}
        if name in ("history", "adjustments"):
            return []
        if name in ("error", "build", "status_reason", "stop_reason", "env"):
            return None
        raise AttributeError(name)

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, dir: Path | None = None) -> TrialRecord:
        return cls(raw=dict(d), dir=dir)

    @property
    def seed(self) -> Any:
        return (self.raw.get("labels") or {}).get("seed")

    @property
    def config_labels(self) -> dict[str, Any]:
        return {k: v for k, v in (self.raw.get("labels") or {}).items() if k != "seed"}

    @property
    def config_key(self) -> tuple:
        return tuple(sorted((k, str(v)) for k, v in self.config_labels.items()))

    @property
    def value(self) -> float | None:
        v = (self.raw.get("goal") or {}).get("value")
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None

    def metric(self, name: str | None) -> float | None:
        """Value of ``name``: goal (respects ``select``) > final > cost > extras > history."""
        goal = self.raw.get("goal") or {}
        if name is None or name == goal.get("metric"):
            return self.value
        for src in (
            self.raw.get("final") or {},
            self.raw.get("cost") or {},
            self.raw.get("extras") or {},
        ):
            v = src.get(name)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v) if math.isfinite(float(v)) else None
        for row in reversed(self.raw.get("history") or []):
            v = row.get(name)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                return float(v)
        return None

    def series(self, name: str) -> list[tuple[int, float]]:
        out = []
        for row in self.raw.get("history") or []:
            v = row.get(name)
            if isinstance(v, (int, float)) and math.isfinite(float(v)) and "epoch" in row:
                out.append((int(row["epoch"]), float(v)))
        return out

    def describe(self) -> str:
        parts = [str(v) for v in self.config_labels.values()]
        return (" × ".join(parts) or self.trial_id) + f" · seed {self.seed}"

    def __repr__(self) -> str:
        return (
            f"TrialRecord({self.raw.get('trial_id')}, {self.config_labels}, "
            f"seed={self.seed}, status={self.raw.get('status')}, value={self.value})"
        )


@dataclass
class ConfigSummary:
    """All seeds of one configuration under one metric."""

    labels: dict[str, Any]
    trials: list[TrialRecord]
    metric: str
    mode: str

    @property
    def completed(self) -> list[TrialRecord]:
        return [t for t in self.trials if t.status == "completed"]

    @property
    def values(self) -> list[float]:
        vals = [t.metric(self.metric) for t in self.completed]
        return [v for v in vals if v is not None]

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float | None:
        return st.mean(self.values)

    @property
    def std(self) -> float | None:
        return st.std(self.values)

    @property
    def ci(self) -> tuple[float, float] | None:
        return st.ci95(self.values)

    @property
    def statuses(self) -> Counter:
        return Counter(t.status for t in self.trials)

    def cost(self, name: str) -> float | None:
        return st.mean([(t.cost or {}).get(name) for t in self.completed or self.trials])

    def curve(self, metric: str | None = None) -> list[float]:
        """Mean of ``metric`` (default: the summary metric) per epoch over seeds."""
        metric = metric or self.metric
        by_epoch: dict[int, list[float]] = {}
        for t in self.completed or self.trials:
            for e, v in t.series(metric):
                by_epoch.setdefault(e, []).append(v)
        return [sum(v) / len(v) for _, v in sorted(by_epoch.items())]

    def label(self, axes: Sequence[str] | None = None) -> str:
        keys = axes or list(self.labels)
        return " × ".join(str(self.labels.get(k, "—")) for k in keys) or "(all)"

    def __repr__(self) -> str:
        m = "—" if self.mean is None else f"{self.mean:.4g}"
        return f"ConfigSummary({self.labels}, {self.metric}={m}, n={self.n})"


def _matches(value: Any, want: Any) -> bool:
    if isinstance(want, (list, tuple, set, frozenset)):
        return any(_matches(value, w) for w in want)
    return str(value) == str(want)


def parse_selector(text: str) -> dict[str, str]:
    """``"model=X,tokenizer=Y"`` -> ``{"model": "X", "tokenizer": "Y"}``."""
    out: dict[str, str] = {}
    for part in str(text).split(","):
        if "=" in part:
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------- ResultSet


class ResultSet:
    """Records of one (or several merged) experiments with analysis views."""

    def __init__(
        self,
        records: Iterable[TrialRecord],
        *,
        goal: tuple[str, str] | None = None,
        axes: Sequence[str] | None = None,
        name: str | None = None,
        baseline: dict[str, Any] | None = None,
        recipe: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.records = list(records)
        self.name = name
        self.baseline = dict(baseline) if baseline else None
        self.meta = dict(meta or {})
        if goal is None:
            g = next((r.goal for r in self.records if r.goal), {}) or {}
            goal = (
                g.get("metric") or "val_loss",
                g.get("mode") or metric_info(g.get("metric") or "val_loss").mode,
            )
        self.goal = (str(goal[0]), str(goal[1]))
        if axes is None:
            seen: dict[str, None] = {}
            for r in self.records:
                for k in r.config_labels:
                    seen.setdefault(k, None)
            axes = list(seen)
        self.axes = list(axes)
        self.recipe = recipe

    # -- basics ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def analyze(self, analysis: Any, context: Any = None) -> Any:
        """Analyze a read-only generic snapshot of these trial records."""
        from topologies.analysis import AnalysisContext, analyze
        from topologies.analysis.integrations import from_result_set

        records = from_result_set(self)
        result = analyze(records, analysis, context=context or AnalysisContext())
        result.provenance.setdefault("source", records.source_provenance)
        result.provenance.setdefault("goal", records.source_provenance["goal"])
        return result

    def visualize(self, **options: Any) -> Any:
        """Explicitly render this result set with the shared visualization API."""
        from topologies.visualization import visualize

        return visualize(self, **options)

    def __iter__(self):
        return iter(self.records)

    def __repr__(self) -> str:
        c = Counter(r.status for r in self.records)
        return (
            f"ResultSet({self.name!r}, {len(self.records)} trials, "
            f"{dict(c)}, goal={self.goal[0]} {'↓' if self.goal[1] == 'min' else '↑'})"
        )

    def _derive(self, records: Iterable[TrialRecord]) -> ResultSet:
        return ResultSet(
            records,
            goal=self.goal,
            axes=self.axes,
            name=self.name,
            baseline=self.baseline,
            recipe=self.recipe,
            meta=self.meta,
        )

    def info(self, metric: str | None = None) -> MetricInfo:
        m = metric or self.goal[0]
        info = metric_info(m, recipe=self.recipe)
        if m == self.goal[0] and info.mode != self.goal[1]:
            from dataclasses import replace

            info = replace(info, mode=self.goal[1], inferred=False)
        return info

    def mode(self, metric: str | None = None) -> str:
        return self.info(metric).mode

    def varies(self, axis: str) -> bool:
        return (
            len({str(r.config_labels.get(axis)) for r in self.records if axis in r.config_labels})
            > 1
        )

    def not_comparable(self, metric: str | None = None) -> bool:
        """Per-token metric while the tokenizer varies (marked ⚠ in tables)."""
        return self.info(metric).per_token and any(
            self.varies(a) for a in self.axes if a == "tokenizer" or a.startswith("tokenizer")
        )

    def metric_header(self, metric: str | None = None) -> str:
        m = metric or self.goal[0]
        info = self.info(m)
        arrow = "↓" if info.mode == "min" else "↑"
        flag = " ⚠" if self.not_comparable(m) else ""
        return f"{m} {arrow}{'?' if info.inferred else ''}{flag}"

    # -- selection -----------------------------------------------------------
    def filter(self, **criteria: Any) -> ResultSet:
        """Keep records whose labels match (``model="X"``, ``tokenizer=["a", "b"]``,
        ``status="completed"``, ``seed=0``)."""

        def ok(r: TrialRecord) -> bool:
            for k, want in criteria.items():
                if k == "status":
                    if not _matches(r.status, want):
                        return False
                    continue
                if k == "seed":
                    if not _matches(r.seed, want):
                        return False
                    continue
                if k not in r.config_labels or not _matches(r.config_labels[k], want):
                    return False
            return True

        return self._derive(r for r in self.records if ok(r))

    def where(self, fn: Callable[[TrialRecord], bool]) -> ResultSet:
        """Keep records for which ``fn(record)`` is true (errors count as false)."""

        def ok(r: TrialRecord) -> bool:
            try:
                return bool(fn(r))
            except Exception:  # noqa: BLE001
                return False

        return self._derive(r for r in self.records if ok(r))

    def completed(self) -> ResultSet:
        return self._derive(r for r in self.records if r.status == "completed")

    def find(self, selector: Any) -> list[TrialRecord]:
        """Trials matching a trial id, a label dict, or ``"model=X,tokenizer=Y"``."""
        if isinstance(selector, str) and "=" not in selector:
            return [r for r in self.records if str(r.trial_id).startswith(selector)]
        if isinstance(selector, str):
            sel: dict[str, Any] = parse_selector(selector)
        else:
            sel = dict(selector)
        seed = sel.pop("seed", None)
        out = [
            r
            for r in self.records
            if all(_matches(r.config_labels.get(k), v) for k, v in sel.items())
        ]
        if seed is not None:
            out = [r for r in out if _matches(r.seed, seed)]
        return out

    # -- aggregation ----------------------------------------------------------
    def summaries(
        self,
        metric: str | None = None,
        *,
        include_failed: bool = False,
        include_pruned: bool = False,
    ) -> list[ConfigSummary]:
        """Configs (seeds grouped), best first; empty configs last when included."""
        metric = metric or self.goal[0]
        mode = self.mode(metric)
        groups: dict[tuple, list[TrialRecord]] = {}
        for r in self.records:
            groups.setdefault(r.config_key, []).append(r)
        out = []
        for recs in groups.values():
            s = ConfigSummary(labels=recs[0].config_labels, trials=recs, metric=metric, mode=mode)
            if s.n == 0:
                all_pruned = all(t.status == "pruned" for t in recs)
                if not (include_pruned if all_pruned else include_failed):
                    continue
            out.append(s)
        sign = 1.0 if mode == "min" else -1.0
        return sorted(
            out, key=lambda s: (s.mean is None, sign * (s.mean or 0.0), s.label(self.axes))
        )

    def best(self, n: int = 1, metric: str | None = None) -> list[ConfigSummary]:
        return [s for s in self.summaries(metric) if s.mean is not None][:n]

    def baseline_summary(self, metric: str | None = None) -> ConfigSummary | None:
        if not self.baseline:
            return None
        for s in self.summaries(metric):
            if all(
                str(s.labels.get(k)) == str(v) for k, v in self.baseline.items() if k in s.labels
            ):
                return s
        return None

    # -- leaderboard -------------------------------------------------------------
    def table(
        self,
        metric: str | None = None,
        top: int | None = None,
        group_seeds: bool = True,
        columns: Sequence[str] | None = None,
        include_failed: bool = False,
        ci: bool = False,
    ) -> Table:
        from topologies.experiments.analysis import leaderboard

        return leaderboard(
            self,
            metric=metric,
            top=top,
            group_seeds=group_seeds,
            columns=columns,
            include_failed=include_failed,
            ci=ci,
        )

    # -- analysis views (implemented in analysis.py) ------------------------------
    def pivot(self, row: str, col: str, metric: str | None = None, agg: str = "mean") -> Table:
        from topologies.experiments.analysis import pivot

        return pivot(self, row, col, metric=metric, agg=agg)

    def marginals(self, axis: str | None = None, metric: str | None = None) -> dict[str, Table]:
        from topologies.experiments.analysis import marginals

        return marginals(self, axis, metric=metric)

    def importance(self, metric: str | None = None) -> Table:
        from topologies.experiments.analysis import importance_table

        return importance_table(self, metric=metric)

    def consistency(self, axis: str, metric: str | None = None) -> Table:
        from topologies.experiments.analysis import consistency

        return consistency(self, axis, metric=metric)

    def compare(self, a: Any, b: Any, metric: str | None = None) -> Any:
        from topologies.experiments.analysis import compare

        return compare(self, a, b, metric=metric)

    def pareto(self, x: str = "wall_s", y: str | None = None) -> list[ConfigSummary]:
        from topologies.experiments.analysis import pareto

        return pareto(self, x=x, y=y)

    def curves(self, metric: str = "val_loss", by: str | None = None, x: str = "epoch") -> Any:
        from topologies.experiments.analysis import curves

        return curves(self, metric=metric, by=by, x=x)

    def failures(self) -> Table:
        from topologies.experiments.analysis import failures

        return failures(self)

    def insights(self) -> list[Any]:
        from topologies.experiments.insights import generate

        return generate(self)

    # -- display -------------------------------------------------------------------
    def summary_lines(
        self, *, width: int | None = None, color: bool | None = None, top: int = 15
    ) -> list[str]:
        from topologies.experiments.render.console import render_summary

        return render_summary(self, width=width, color=color, top=top)

    def show(self, *, top: int = 15) -> None:
        """Print leaderboard, pivot (2+ axes), marginals, importance and insights.

        ``exp.run()`` already prints this summary; on the ``ResultSet`` it returns,
        the first ``show()`` is skipped so ``exp.run().show()`` prints it once.
        """
        if self.meta.pop("summary_printed", False):
            return
        print("\n".join(self.summary_lines(top=top)))

    def _repr_html_(self) -> str:
        from topologies.experiments.render.html import summary_html

        return summary_html(self)

    # -- exports ---------------------------------------------------------------------
    def to_records(self) -> list[dict[str, Any]]:
        """One flat dict per trial: labels, status, goal, final metrics, costs."""
        out = []
        for r in self.records:
            d: dict[str, Any] = {
                "experiment": r.experiment,
                "trial_id": r.trial_id,
                **{k: v for k, v in r.config_labels.items()},
                "seed": r.seed,
                "status": r.status,
                "goal_metric": (r.goal or {}).get("metric"),
                "goal": r.value,
                "stop_reason": r.stop_reason,
            }
            d.update(
                {
                    k: v
                    for k, v in (r.final or {}).items()
                    if isinstance(v, (int, float, str, bool)) or v is None
                }
            )
            d.update({f"cost.{k}": v for k, v in (r.cost or {}).items()})
            if r.build:
                d["build.params"] = r.build.get("params")
                d["build.rel_error"] = r.build.get("rel_error")
            if r.error:
                d["error"] = f"{r.error.get('type')}: {r.error.get('message', '')}"[:300]
            out.append(d)
        return out

    def to_json(self, path: str | Path | None = None) -> str:
        from topologies.experiments.store import jsonable

        text = json.dumps(jsonable(self.to_records()), indent=1)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_csv(self, path: str | Path | None = None) -> str:
        recs = self.to_records()
        cols: dict[str, None] = {}
        for d in recs:
            for k in d:
                cols.setdefault(k, None)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(cols))
        w.writeheader()
        for d in recs:
            w.writerow({k: ("" if d.get(k) is None else d.get(k)) for k in cols})
        text = buf.getvalue()
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_markdown(self) -> str:
        from topologies.experiments.render.markdown import report_markdown

        return report_markdown(self)

    def to_pandas(self) -> Any:
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("to_pandas() needs pandas: pip install pandas") from e
        return pd.DataFrame(self.to_records())

    def report(self, path: str | Path = "report.html") -> Path:
        """Write a self-contained HTML report (or Markdown for ``.md``)."""
        p = Path(path)
        if p.suffix.lower() in (".md", ".markdown"):
            p.write_text(self.to_markdown(), encoding="utf-8")
        else:
            from topologies.experiments.render.html import report_html

            p.write_text(report_html(self), encoding="utf-8")
        return p

    def load_model(self, which: Any = "best", *, model: Any = None,
                   trusted_modules: Sequence[str] = (), trust_remote_code: bool = False) -> Any:
        """``load_pretrained`` of a saved trial (needs ``save="best"`` or ``"all"``).

        ``trusted_modules`` / ``trust_remote_code`` are passed to
        ``load_pretrained``: trials whose model class lives in your own package
        need ``trusted_modules=["my_pkg"]``.

        Trials of the default model stage save self-contained bundles. A custom
        ``build_model(ctx)`` that constructs its module directly saves weights
        only (``result.extras["checkpoint"] == "weights-only"``): pass
        ``model=`` (an instance or class) to load those.
        """
        from topologies.build.config import UntrustedTargetError
        from topologies.pretrained import load_pretrained

        if which == "best":
            best = self.best(1)
            if not best:
                raise ValueError("No completed trials.")
            cands = sorted(
                best[0].completed,
                key=lambda t: (
                    (t.value if self.goal[1] == "min" else -(t.value or 0.0))
                    if t.value is not None
                    else math.inf
                ),
            )
        else:
            cands = self.find(which)
        for t in cands:
            if t.dir is not None and (t.dir / "model" / "config.json").is_file():
                try:
                    return load_pretrained(t.dir / "model", model=model,
                                           trusted_modules=trusted_modules,
                                           trust_remote_code=trust_remote_code)
                except UntrustedTargetError:
                    raise
                except ValueError as e:
                    raise ValueError(
                        f"{e} For experiment trials: load_model(which, model=...); the "
                        "trial's model came from a custom build_model(ctx) (return "
                        "topologies.build(MyModel, ...) there to save self-contained "
                        "bundles)."
                    ) from e
        raise FileNotFoundError(
            "No saved model for that trial. Run the experiment with save='best' or save='all'."
        )


# ---------------------------------------------------------------- loading


def load(name: str, root: str | Path | None = None, *, scope: str = "all") -> ResultSet:
    """``ResultSet`` of experiment ``name`` from disk (``scope="all"`` or ``"plan"``)."""
    from topologies.experiments.store import ExperimentStore

    store = ExperimentStore(root, name)
    if not store.dir.is_dir():
        from topologies.experiments.errors import ExperimentError
        from topologies.experiments.store import list_experiments

        raise ExperimentError(
            f"No experiment {name!r} under {store.root}.",
            stage="results",
            available=list_experiments(store.root),
        )
    spec = store.read_spec() or {}
    ids = store.trial_ids()
    if scope == "plan" and spec.get("plan"):
        keep = set(spec["plan"])
        ids = [i for i in ids if i in keep]
    records = []
    for tid in ids:
        data = store.read_result(tid)
        if data is not None:
            records.append(TrialRecord.from_dict(data, dir=store.trial_dir(tid)))
    goal = spec.get("goal") or {}
    axes = list((spec.get("axes") or {}).keys()) or None
    task = spec.get("task") or {}
    recipe = (task.get("describe") or {}).get("recipe") if isinstance(task, dict) else None
    res = ResultSet(
        records,
        goal=(goal["metric"], goal["mode"]) if goal.get("metric") else None,
        axes=axes,
        name=spec.get("name", name),
        baseline=spec.get("baseline_labels"),
        recipe=recipe,
        meta={"spec": spec, "dir": str(store.dir)},
    )
    return res


def load_all(root: str | Path | None = None, names: Sequence[str] | None = None) -> ResultSet:
    """Union of several experiments; axes missing from an experiment get the label ``—``."""
    from topologies.experiments.store import list_experiments

    names = list(names) if names is not None else list_experiments(root)
    sets = [load(n, root) for n in names]
    axes: dict[str, None] = {}
    for s in sets:
        for a in s.axes:
            axes.setdefault(a, None)
    records = []
    for s in sets:
        for r in s.records:
            raw = dict(r.raw)
            labels = dict(raw.get("labels") or {})
            for a in axes:
                labels.setdefault(a, "—")
            raw["labels"] = {**{a: labels[a] for a in axes}, "seed": labels.get("seed")}
            records.append(TrialRecord(raw=raw, dir=r.dir))
    goal = sets[0].goal if sets else None
    return ResultSet(
        records,
        goal=goal,
        axes=list(axes),
        name="+".join(names) or None,
        recipe=sets[0].recipe if sets else None,
    )


__all__ = [
    "ConfigSummary",
    "ResultSet",
    "Table",
    "TrialRecord",
    "fmt_params",
    "fmt_rate",
    "fmt_seconds",
    "load",
    "load_all",
    "parse_selector",
]
