"""``Experiment.plan()``: validate every trial before spending compute.

Checks, in order: key routing (typos, with suggestions), trainer / optimizer /
recipe values (``validate_with_hints``), tokenizers (trained once and
cached), model construction on the ``meta`` device with parameter matching,
model/data compatibility (vocabulary, context length, hidden position
tables), picklability for parallel runs, and optionally a 2-step smoke run
per unique config. Per-trial problems do not abort the experiment: the
affected trials are recorded as ``failed`` with the fix, the rest run.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from topologies.experiments.errors import ExperimentError
from topologies.experiments.spec import canonical_json
from topologies.experiments.trial import Trial

MODEL_PREFIXES = ("model", "size", "target_params", "search", "tolerance", "param_count",
                  "tokenizer", "task")


def _is_model_key(k: str) -> bool:
    return any(k == p or k.startswith(p + ".") for p in MODEL_PREFIXES)


def model_group_key(trial: Trial) -> str:
    return canonical_json({k: v for k, v in trial.config.items() if _is_model_key(k)})


@dataclass
class PlanRow:
    labels: dict[str, Any]
    trials: list[Trial]
    params: int | None = None
    target: int | None = None
    rel_error: float | None = None
    knobs: dict[str, Any] | None = None
    embedding_share: float | None = None
    error: str | None = None
    note: str | None = None


@dataclass
class Plan:
    """Result of ``Experiment.plan()``; ``print(plan)`` shows the trial matrix."""

    name: str
    trials: list[Trial]
    rows: list[PlanRow] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    statuses: dict[str, str] = field(default_factory=dict)
    estimate_s: float | None = None
    smoke: dict[str, str] | None = None
    axes: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def pending(self) -> list[Trial]:
        return [t for t in self.trials if self.statuses.get(t.trial_id) != "completed"]

    def raise_for_errors(self) -> None:
        if self.errors:
            first = next(iter(self.errors.values()))
            raise ExperimentError(f"{len(self.errors)} trial(s) fail the plan checks; first: "
                                  f"{first}", stage="plan")

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "trials": len(self.trials),
                "configs": len({tuple(sorted(t.config_labels.items())) for t in self.trials}),
                "seeds": self.seeds, "cached": sum(1 for s in self.statuses.values()
                                                   if s == "completed"),
                "estimate_s": self.estimate_s, "errors": self.errors, "notes": self.notes,
                "smoke": self.smoke,
                "models": [{"labels": r.labels, "params": r.params, "target": r.target,
                            "rel_error": r.rel_error, "knobs": r.knobs, "error": r.error,
                            "embedding_share": r.embedding_share, "note": r.note}
                           for r in self.rows]}

    def __str__(self) -> str:
        from topologies.experiments.render.console import render_plan

        return "\n".join(render_plan(self))

    def __repr__(self) -> str:
        return (f"Plan({self.name!r}, {len(self.trials)} trials, {len(self.errors)} errors, "
                f"{len(self.pending)} to run)")


def _validate_values(exp: Any, trial: Trial) -> str | None:
    """Trainer / optimizer / recipe values, with ``validate_with_hints`` messages."""
    from topologies.training.configs import OptimizerConfig, TrainerConfig, validate_with_hints
    from topologies.training.errors import TrainingError

    p = trial.params
    try:
        tcfg = {k.split(".", 1)[1]: v for k, v in p.items() if k.startswith("trainer.")}
        tcfg = {k: v for k, v in tcfg.items() if k in TrainerConfig.model_fields}
        validate_with_hints(TrainerConfig, tcfg)
        from topologies.experiments.pipeline import optimizer_args, recipe_config

        ctx = SimpleNamespace(params=p)
        opt = optimizer_args(ctx).get("optimizer")
        if isinstance(opt, dict):
            entry = {("name" if k == "optimizer" else k): v for k, v in opt.items()}
            validate_with_hints(OptimizerConfig, entry)
        recipe = p.get("recipe") or getattr(exp.task, "recipe", None)
        rcfg = recipe_config(ctx)
        if recipe and rcfg:
            from topologies.training.recipe_base import get_recipe

            schema = type(get_recipe(str(recipe))).ConfigSchema
            if "loss" in rcfg and "loss" not in schema.model_fields:
                return f"recipe {recipe!r} does not support a loss override."
            validate_with_hints(schema, rcfg)
    except TrainingError as e:
        return str(e).split("\n")[0] + (f" {e.hint}" if getattr(e, "hint", None) and
                                         "Did you mean" in str(e.hint) else "")
    except ExperimentError as e:
        return str(e).split("\n")[0]
    return None


def _check_models(exp: Any, trials: list[Trial], plan: Plan) -> None:
    """One ``meta``-device build per unique model config (tokenizer included)."""
    from topologies.build.params import count_params, embedding_params
    from topologies.experiments.pipeline import build_model, build_report, check_compat
    from topologies.experiments.trial import TrialContext

    groups: dict[str, list[Trial]] = {}
    for t in trials:
        groups.setdefault(model_group_key(t), []).append(t)
    runtime = exp.runtime()
    relevant = [a.name for a in exp.axes if any(
        _is_model_key(k) for v in a.values for k in v.assignments(a.name)) or
        (a.name == "model")]
    for members in groups.values():
        t0 = members[0]
        row = PlanRow(labels={a: t0.labels.get(a) for a in relevant}, trials=members)
        plan.rows.append(row)
        if exp.free_form:
            continue
        ctx = TrialContext(runtime, t0, device="cpu", cache=exp.cache)
        try:
            _ = ctx.tokenizer
            model = build_model(ctx, meta=True)
            if exp.build_model is None:
                check_compat(ctx, model)
            row.params = count_params(model)
            emb = embedding_params(model)
            row.embedding_share = (emb / row.params) if row.params else None
            rep = build_report(model)
            if rep:
                row.target, row.rel_error, row.knobs = rep["target"], rep["rel_error"], \
                    rep["knobs"]
            if ctx.info.get("param_matching"):
                row.note = ctx.info["param_matching"]
            del model
        except ExperimentError as e:
            row.error = e.message + (f" Hint: {e.hint}" if e.hint else "")
        except ImportError as e:
            row.error = f"{e}"
        except Exception as e:  # noqa: BLE001
            if exp.build_model is not None:
                row.note = f"dry build skipped ({type(e).__name__}: {e})"[:200]
            else:
                row.error = f"{type(e).__name__}: {e}".split("\n")[0][:400]
        if row.error:
            for t in members:
                plan.errors.setdefault(t.trial_id, row.error)


def _smoke(exp: Any, trials: list[Trial], plan: Plan) -> None:
    """Two optimizer steps per unique config, in a scratch directory."""
    from topologies.experiments.pipeline import run_pipeline
    from topologies.experiments.trial import TrialContext

    plan.smoke = {}
    if exp.free_form:
        plan.notes.append("smoke test skipped for trial_fn experiments")
        return
    seen: set[str] = set()
    tmp = tempfile.mkdtemp(prefix="topologies-smoke-")
    try:
        runtime = exp.runtime(budget={"steps": 2})
        runtime.exp_dir, runtime.save, runtime.rung = tmp, "none", None
        runtime.pruner = None
        for t in trials:
            key = canonical_json(t.config)
            if key in seen or t.trial_id in plan.errors:
                continue
            seen.add(key)
            ctx = TrialContext(runtime, t, device=_plan_device(exp, t), cache=exp.cache)
            ctx.dir.mkdir(parents=True, exist_ok=True)
            try:
                run_pipeline(ctx)
                plan.smoke[t.trial_id] = "ok"
            except Exception as e:  # noqa: BLE001
                msg = f"smoke: {type(e).__name__}: " + (str(e).splitlines() or [""])[0]
                plan.smoke[t.trial_id] = msg
                for u in trials:
                    if canonical_json(u.config) == key:
                        plan.errors.setdefault(u.trial_id, msg[:400])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _plan_device(exp: Any, trial: Trial) -> str:
    from topologies.experiments.runner import resolve_device

    return resolve_device(exp.devices[0] if exp.devices else trial.params.get(
        "trainer.device"))


def _estimate(exp: Any, plan: Plan) -> None:
    from topologies.experiments.core import estimate_seconds

    done: dict[tuple, list[float]] = {}
    overall: list[float] = []
    for tid in exp.store.trial_ids():
        res = exp.store.read_result(tid, warn=False)
        if not res or res.get("status") != "completed":
            continue
        w = (res.get("cost") or {}).get("wall_s")
        if not isinstance(w, (int, float)):
            continue
        labels = {k: v for k, v in (res.get("labels") or {}).items() if k != "seed"}
        done.setdefault(tuple(sorted((k, str(v)) for k, v in labels.items())), []).append(w)
        overall.append(w)
    pending = [t for t in plan.trials if plan.statuses.get(t.trial_id) != "completed"]
    plan.estimate_s = estimate_seconds(pending, done, overall)
    if exp.workers > 1 and plan.estimate_s is not None:
        plan.estimate_s /= exp.workers


def make_plan(exp: Any, *, smoke: bool = False) -> Plan:
    t_start = time.monotonic()
    trials = exp.trials()
    plan = Plan(name=exp.name, trials=trials, axes=[a.name for a in exp.axes],
                seeds=sorted({t.seed for t in trials}))
    if exp.workers > 1 or (exp.devices and len(exp.devices) > 1):
        from topologies.experiments.parallel import check_picklable

        check_picklable(exp, trials)
    for t in trials:
        res = exp.store.read_result(t.trial_id, warn=False)
        if res is not None:
            plan.statuses[t.trial_id] = res.get("status", "?")
    if not exp.free_form:
        for t in trials:
            err = _validate_values(exp, t)
            if err:
                plan.errors[t.trial_id] = err
        hook = getattr(exp.task, "plan_check", None)
        skip = set(hook(exp, trials, plan) or ()) if callable(hook) else set()
        _check_models(exp, [t for t in trials if t.trial_id not in skip], plan)
        _embedding_notes(exp, plan)
    if smoke:
        _smoke(exp, trials, plan)
    _estimate(exp, plan)
    for t in trials:
        t.plan_error = plan.errors.get(t.trial_id)
    plan.notes.append(f"planned in {time.monotonic() - t_start:.1f}s")
    return plan


def _embedding_notes(exp: Any, plan: Plan) -> None:
    if exp.target_params is None and not any(
            "target_params" in t.params for t in plan.trials):
        return
    big = [r for r in plan.rows if r.embedding_share is not None and r.embedding_share > 0.30]
    if big:
        labels = ", ".join(" × ".join(str(v) for v in r.labels.values()) for r in big[:4])
        plan.notes.append(f"embeddings exceed 30% of the parameters for {labels}; consider "
                          "param_count='non_embedding' for a fairer architecture comparison")


__all__ = ["Plan", "PlanRow", "make_plan", "model_group_key"]
