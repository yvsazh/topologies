"""``Experiment``: declare what varies; plan, run (resumably) and analyze every trial.

    exp = Experiment(
        "arch-x-tokenizer",
        task=LanguageModeling(train_docs, val_docs, block_size=128),
        axes={"model": [TransformerLM, Variant("rope", TransformerLM, pos="rope")],
              "tokenizer": ["char", "bpe:1k", "bpe:4k"]},
        target_params="1M", train_for="3 epochs", train=dict(batch_size=32, lr=1e-3),
        seeds=2,
    )
    exp.plan()          # trial matrix, params per model, compatibility, time estimate
    res = exp.run()     # live board; resumable; returns a ResultSet
    res.show()

Level 2: ``build_model=`` / ``build_data=`` / ``make_trainer=`` / ``evaluate=``
replace single stages. Level 3: ``@experiment(...)`` wraps a whole trial
function. Spaces, seeds, storage, results and the CLI work the same for all.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

_VISUALIZERS_UNSET = object()

from topologies.experiments.cache import Cache
from topologies.experiments.errors import ExperimentError
from topologies.experiments.routing import Router, parse_duration, parse_train_for
from topologies.experiments.space import (
    Axis,
    apply_filters,
    grid_points,
    labels_of,
    normalize_axes,
    ofat_points,
    random_points,
    resolve_seeds,
    trial_order,
)
from topologies.experiments.spec import canonical, trial_hash
from topologies.experiments.store import ExperimentStore
from topologies.experiments.trial import FINAL_FAILURES, Runtime, Trial

_CHOICES = {
    "select": ("last", "best"),
    "space": ("grid", "random", "ofat"),
    "strategy": ("full", "halving"),
    "save": ("none", "best", "all"),
    "on_error": ("continue", "raise"),
    "on_oom": ("skip", "halve_batch"),
    "monitor": ("live", "lines", "plain", "dashboard", "none"),
    "param_count": ("total", "trainable", "non_embedding"),
}


def _check_choice(name: str, value: Any) -> None:
    allowed = _CHOICES[name]
    if value not in allowed:
        sug = difflib.get_close_matches(str(value), allowed, n=1)
        raise ExperimentError(f"{name}={value!r} is not one of {list(allowed)}.", stage="config",
                              key=name, hint=f"Did you mean {sug[0]!r}?" if sug else None)


def _monitoring_options(monitoring: dict | None) -> dict[str, Any]:
    """Dashboard options. ``charts`` lists series such as ``"loss"`` and ``"goal"``."""
    if monitoring is None:
        return {}
    if not isinstance(monitoring, dict):
        raise ExperimentError("monitoring must be a dict of dashboard options.",
                              stage="config", key="monitoring")
    options = dict(monitoring)
    if "charts" not in options:
        return options
    from topologies.visualization.experiment_views import normalize_charts

    try:
        options["charts"] = list(normalize_charts(options["charts"]))
    except ValueError as exc:
        raise ExperimentError(str(exc), stage="config", key="monitoring.charts",
                              hint="Use charts=['loss', 'goal'] or the goal metric name.") from exc
    return options


def resolve_goal(goal: Any, task: Any) -> tuple[str, str]:
    from topologies.experiments.metrics import metric_info

    if goal is None:
        goal = getattr(task, "default_goal", None) or ("val_loss", "min")
    if isinstance(goal, str):
        return goal, metric_info(goal).mode
    if isinstance(goal, (tuple, list)) and len(goal) == 2 and goal[1] in ("min", "max"):
        return str(goal[0]), str(goal[1])
    raise ExperimentError(f"Invalid goal {goal!r}.", stage="config", key="goal",
                          hint="Use goal='val_bpb' or goal=('val_acc', 'max').")


class Experiment:
    """A set of trials over declared axes (see module docstring and docs/experiments.md)."""

    def __init__(
        self,
        name: str,
        *,
        task: Any = None,
        axes: dict[str, Any] | None = None,
        fixed: dict[str, Any] | None = None,
        model: Any = None,
        size: str | None = None,
        target_params: str | int | None = None,
        param_count: str = "total",
        train_for: str | int | dict | None = None,
        train: dict | None = None,
        seeds: int | Sequence[int] = 1,
        goal: str | tuple[str, str] | None = None,
        select: str = "last",
        space: str = "grid",
        n_trials: int | None = None,
        baseline: dict | None = None,
        where: Callable[[dict], bool] | None = None,
        exclude: list[dict] | None = None,
        include: list[dict] | None = None,
        strategy: str = "full",
        prune: Any = None,
        trial_timeout: str | float | None = None,
        workers: int = 1,
        devices: Sequence[str] | None = None,
        threads_per_worker: int | None = None,
        save: str = "none",
        root: str | Path | None = None,
        on_error: str = "continue",
        on_oom: str = "skip",
        monitor: str = "live",
        monitoring: dict | None = None,
        trial_fn: Callable | None = None,
        build_model: Callable | None = None,
        build_data: Callable | None = None,
        make_trainer: Callable | None = None,
        evaluate: Callable | None = None,
        tags: dict[str, str] | None = None,
        description: str = "",
        eta: int = 3,
        min_budget: str | int | None = None,
        visualizer: Any = _VISUALIZERS_UNSET,
        visualizers: Any = _VISUALIZERS_UNSET,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ExperimentError("Experiment needs a non-empty name.", stage="config")
        for k, v in (("select", select), ("space", space), ("strategy", strategy),
                     ("save", save), ("on_error", on_error), ("on_oom", on_oom),
                     ("monitor", monitor)):
            _check_choice(k, v)
        if visualizer is not _VISUALIZERS_UNSET and visualizers is not _VISUALIZERS_UNSET:
            raise ExperimentError("Pass visualizer or visualizers, not both.", stage="config")
        self._visualizer_selection = visualizers if visualizers is not _VISUALIZERS_UNSET else visualizer
        if not callable(param_count):
            _check_choice("param_count", param_count)
        self.name = name
        self.task = task
        self.trial_fn = trial_fn
        self.free_form = trial_fn is not None
        self.axes_spec = dict(axes or {})
        self.fixed_spec = dict(fixed or {})
        self.model, self.size, self.target_params = model, size, target_params
        self.param_count = param_count
        self.train_for = train_for
        self.train = dict(train or {})
        self.seeds = resolve_seeds(seeds)
        self.goal = resolve_goal(goal, task)
        self._goal_given = goal is not None
        self.select, self.space, self.n_trials = select, space, n_trials
        self.baseline = dict(baseline) if baseline else None
        self.where, self.exclude, self.include = where, list(exclude or []), list(include or [])
        self.strategy, self.eta, self.min_budget = strategy, int(eta), min_budget
        self.pruner = self._make_pruner(prune)
        if self.pruner is not None and getattr(self.pruner, "mode", "set") is None:
            self.pruner.mode = self.goal[1]
        self.trial_timeout = parse_duration(trial_timeout) if trial_timeout is not None else None
        self.workers = max(1, int(workers))
        self.devices = list(devices) if devices else None
        if threads_per_worker is not None and int(threads_per_worker) < 1:
            raise ExperimentError("threads_per_worker must be >= 1.", stage="config",
                                  key="threads_per_worker")
        self.threads_per_worker = int(threads_per_worker) if threads_per_worker else None
        self.save, self.root = save, root
        self.on_error, self.on_oom, self.monitor = on_error, on_oom, monitor
        self.monitoring = _monitoring_options(monitoring)
        self.build_model, self.build_data = build_model, build_data
        self.make_trainer, self.evaluate = make_trainer, evaluate
        self.tags, self.description = dict(tags or {}), description
        self.budget = parse_train_for(train_for) if not self.free_form or train_for else \
            {"epochs": 1}
        if strategy == "halving" and self.free_form:
            raise ExperimentError("strategy='halving' needs the default pipeline (Trainer).",
                                  stage="config", key="strategy")
        if space == "random" and not n_trials:
            raise ExperimentError("space='random' needs n_trials.", stage="config",
                                  key="n_trials", hint="Experiment(..., n_trials=20)")
        self._axes: list[Axis] = normalize_axes(self.axes_spec, free_form=self.free_form)
        self.store = ExperimentStore(root, name)
        self._cache: Cache | None = None
        self._trials: list[Trial] | None = None
        self._spec_path: str | None = None

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _make_pruner(prune: Any) -> Any:
        if prune is None:
            return None
        if prune == "median":
            from topologies.experiments.pruners import MedianPruner

            return MedianPruner()
        if hasattr(prune, "should_prune"):
            return prune
        raise ExperimentError(f"Unknown prune={prune!r}.", stage="config", key="prune",
                              hint="Use prune='median' or an object with should_prune(...).")

    @property
    def cache(self) -> Cache:
        if self._cache is None:
            self._cache = Cache(self.store.root)
        return self._cache

    @property
    def axes(self) -> list[Axis]:
        return list(self._axes)

    def _fixed_raw(self) -> dict[str, Any]:
        """Experiment-level settings as raw (unrouted) keys; axes override them."""
        raw: dict[str, Any] = {}
        axis_names = {a.name for a in self._axes}
        if self.model is not None and "model" not in axis_names:
            raw["model"] = self.model
        if self.size is not None:
            raw["size"] = self.size
        if self.target_params is not None:
            raw["target_params"] = self.target_params
        if self.param_count != "total":
            raw["param_count"] = self.param_count
        if not self.free_form and "train_for" not in axis_names:
            raw["train_for"] = dict(self.budget)  # part of every trial's identity
        if self.free_form:
            raw.update(self.train)
        else:
            from topologies.experiments.routing import trainer_fields

            tf = trainer_fields()
            for k, v in self.train.items():
                raw[f"trainer.{k}" if (k in tf and "." not in k) else k] = v
        for k, v in self.fixed_spec.items():
            raw[k] = v.value if hasattr(v, "value") and type(v).__name__ == "Fixed" else v
        return raw

    def router(self) -> Router:
        recipes: list[str | None] = [getattr(self.task, "recipe", None)]
        models: list[Any] = []
        raw = self._fixed_raw()
        if raw.get("recipe"):
            recipes.append(raw["recipe"])
        if raw.get("model") is not None:
            models.append(raw["model"])
        for ax in self._axes:
            if ax.name == "recipe":
                recipes += [v.value for v in ax.values if isinstance(v.value, str)]
            if ax.name == "model":
                models += [v.value for v in ax.values if v.has_value]
            for v in ax.values:
                if "model" in v.overrides:
                    models.append(v.overrides["model"])
        task_params = self.task.param_names() if hasattr(self.task, "param_names") else set()
        return Router(recipes=recipes, models=models, task_params=task_params,
                      free_form=self.free_form,
                      synth=bool(getattr(self.task, "accepts_synth", False)))

    # -- trials ------------------------------------------------------------------
    def _points(self) -> list[dict]:
        if self.space == "grid":
            pts = grid_points(self._axes)
        elif self.space == "random":
            pts = random_points(self._axes, int(self.n_trials or 0), name=self.name)
        else:
            pts = ofat_points(self._axes, self.baseline)
        return apply_filters(pts, self._axes, where=self.where, exclude=self.exclude,
                             include=self.include, baseline=self.baseline)

    def trials(self) -> list[Trial]:
        """Every planned trial (routing errors raise here, before any compute)."""
        if self._trials is not None:
            return list(self._trials)
        router = self.router()
        fixed_raw = self._fixed_raw()
        task_fp = self.task.fingerprint() if self.task is not None and hasattr(
            self.task, "fingerprint") else None
        seed_axis = any(a.name == "seed" for a in self._axes)
        if seed_axis and len(self.seeds) > 1:
            raise ExperimentError("Use either seeds=... or a 'seed' axis, not both.",
                                  stage="space", key="seed")
        trials: list[Trial] = []
        seen: set[str] = set()
        for point, seed in trial_order(self._points(), [0] if seed_axis else self.seeds,
                                       name=self.name):
            items = [(k, v, ax.name) for ax in self._axes
                     for k, v in point[ax.name].assignments(ax.name).items()]
            model = fixed_raw.get("model")
            for k, v, _src in items:
                if k == "model":
                    model = v
            scope = [model] if model is not None else None
            params = router.route(fixed_raw, scope)
            setters: dict[str, str] = {}
            for k, v, src in items:
                ck = router.resolve(k, scope)
                if ck in setters and setters[ck] != src and \
                        canonical(params.get(ck), warn=False) != canonical(v, warn=False):
                    raise ExperimentError(
                        f"Axes {setters[ck]!r} and {src!r} both set {ck!r} "
                        f"(labels {labels_of(point)}).", stage="space", key=ck,
                        hint="Give the key to one axis, or exclude the conflicting "
                             "combinations with exclude=[...].")
                params[ck] = v
                setters[ck] = src
            s = int(params.pop("seed", seed))
            config = canonical(params)
            tid = trial_hash(task_fp, config, s)
            if tid in seen:
                continue
            seen.add(tid)
            labels = {**labels_of(point), "seed": s}
            trials.append(Trial(trial_id=tid, labels=labels, params=params, config=config,
                                seed=s, index=len(trials)))
        if not trials:
            raise ExperimentError("The experiment has no trials (filters removed everything).",
                                  stage="space")
        self._trials = trials
        return list(trials)

    # -- spec --------------------------------------------------------------------
    def spec(self) -> dict[str, Any]:
        """JSON-able snapshot (what ``experiment.json`` stores)."""
        task = None
        if self.task is not None:
            task = {"type": getattr(self.task, "type", type(self.task).__name__),
                    "fingerprint": self.task.fingerprint() if hasattr(self.task,
                                                                      "fingerprint") else None,
                    "describe": self.task.describe() if hasattr(self.task, "describe") else {}}
        axes = {}
        for ax in self._axes:
            axes[ax.name] = ax.labels if ax.values else [ax.distribution.describe()
                                                         if ax.distribution else "?"]
        fixed = canonical(self._fixed_raw(), warn=False)
        out = {"description": self.description, "tags": self.tags, "task": task, "axes": axes,
               "fixed": fixed, "goal": {"metric": self.goal[0], "mode": self.goal[1]},
               "select": self.select, "space": self.space, "n_trials": self.n_trials,
               "strategy": self.strategy, "seeds": self.seeds,
               "train_for": self.budget, "baseline": canonical(self.baseline, warn=False),
               "prune": type(self.pruner).__name__ if self.pruner is not None else None,
               "trial_timeout": self.trial_timeout, "save": self.save,
               "level": 3 if self.free_form else (2 if any(
                   (self.build_model, self.build_data, self.make_trainer, self.evaluate))
                   else 1)}
        if self.trial_fn is not None:
            out["trial_fn"] = canonical(self.trial_fn, warn=False)
        if self.baseline:
            labels = {}
            for ax in self._axes:
                if ax.name in self.baseline:
                    av = ax.find(self.baseline[ax.name])
                    labels[ax.name] = av.label if av is not None else str(self.baseline[ax.name])
            out["baseline_labels"] = labels
        return out

    def runtime(self, *, budget: dict | None = None, rung: dict | None = None) -> Runtime:
        return Runtime(
            name=self.name, task=self.task, goal=self.goal, select=self.select,
            budget=dict(self.budget), budget_override=dict(budget) if budget else None,
            root=str(self.store.root),
            exp_dir=str(self.store.dir), trial_timeout=self.trial_timeout,
            pruner=self.pruner, save=self.save, on_oom=self.on_oom, trial_fn=self.trial_fn,
            build_model=self.build_model, build_data=self.build_data,
            make_trainer=self.make_trainer, evaluate=self.evaluate,
            default_device=self.devices[0] if self.devices else None,
            free_form=self.free_form, rung=rung)

    # -- plan / run / results ------------------------------------------------------
    def plan(self, smoke: bool = False, *, verbose: bool | None = None) -> Any:
        """Validate every trial before training (see ``Plan``); prints when verbose."""
        from topologies.experiments.planning import make_plan

        plan = make_plan(self, smoke=smoke)
        if verbose if verbose is not None else self.monitor != "none":
            print(plan)
        return plan

    def run(self, *, resume: bool = True, rerun: str | None = None,
            limit: int | None = None, visualizer: Any = _VISUALIZERS_UNSET,
            visualizers: Any = _VISUALIZERS_UNSET) -> Any:
        """Run every pending trial; returns the ``ResultSet`` of the plan.

        ``resume=True`` skips completed trials and keeps failed ones (re-run them with
        ``rerun="failed"``); ``rerun="all"`` (or ``resume=False``) re-runs everything,
        archiving old results as ``result.<timestamp>.json``. ``limit`` caps how many
        trials this call runs.
        """
        from topologies.experiments.coordinator import run_experiment

        if rerun not in (None, "failed", "all"):
            raise ExperimentError(f"rerun={rerun!r} must be None, 'failed' or 'all'.",
                                  stage="config", key="rerun")
        if not resume and rerun is None:
            rerun = "all"
        if visualizer is not _VISUALIZERS_UNSET and visualizers is not _VISUALIZERS_UNSET:
            raise ExperimentError("Pass visualizer or visualizers, not both.", stage="config")
        selection = visualizers if visualizers is not _VISUALIZERS_UNSET else visualizer
        if selection is not _VISUALIZERS_UNSET:
            return run_experiment(self, rerun=rerun, limit=limit, visualizer=selection)
        return run_experiment(self, rerun=rerun, limit=limit)

    def results(self, scope: str = "plan") -> Any:
        """``ResultSet`` of the current plan's trials (``scope="all"``: everything on disk)."""
        from topologies.experiments.results import ResultSet, TrialRecord

        if scope not in ("plan", "all"):
            raise ExperimentError("scope must be 'plan' or 'all'.", stage="results")
        ids = [t.trial_id for t in self.trials()] if scope == "plan" else self.store.trial_ids()
        recs = []
        for tid in ids:
            data = self.store.read_result(tid)
            if data is not None:
                recs.append(TrialRecord.from_dict(data, dir=self.store.trial_dir(tid)))
        spec = self.spec()
        return ResultSet(recs, goal=self.goal, axes=[a.name for a in self._axes] or None,
                         name=self.name, baseline=spec.get("baseline_labels"),
                         recipe=getattr(self.task, "recipe", None),
                         meta={"spec": spec, "dir": str(self.store.dir)})

    def analyze(self, analysis: Any, context: Any = None) -> Any:
        """Analyze current persisted results only; does not run pending trials."""
        return self.results().analyze(analysis, context=context)

    def pending(self, rerun: str | None = None) -> list[Trial]:
        out = []
        for t in self.trials():
            res = self.store.read_result(t.trial_id)
            status = (res or {}).get("status")
            if rerun == "all" or status is None:
                out.append(t)
            elif status == "completed" or status in FINAL_FAILURES and rerun != "failed":
                continue
            else:
                out.append(t)
        return out

    def __call__(self, **overrides: Any) -> Any:
        """Shortcut: ``exp()`` runs it (``@experiment`` functions read naturally)."""
        return self.run(**overrides)

    def __repr__(self) -> str:
        axes = ", ".join(f"{a.name}[{len(a.values) or '∞'}]" for a in self._axes)
        return f"Experiment({self.name!r}, axes=({axes}), seeds={self.seeds})"


def experiment(_fn: Callable | None = None, *, name: str | None = None,
               **options: Any) -> Any:
    """Decorator: a fully custom trial function becomes an ``Experiment``.

    >>> @experiment(axes={"lr": loguniform(1e-4, 1e-2), "width": [64, 128]},
    ...             space="random", n_trials=12, seeds=2, goal=("val_acc", "max"))
    ... def mlp_sweep(ctx, lr, width):
    ...     ...
    ...     return trainer.evaluate(val, prefix="val_")
    >>> res = mlp_sweep.run()
    """
    def deco(fn: Callable) -> Experiment:
        return Experiment(name or fn.__name__, trial_fn=fn, **options)

    if _fn is not None and callable(_fn):
        return deco(_fn)
    return deco


def estimate_seconds(pending: Sequence[Trial], done: dict[tuple, list[float]],
                     overall: list[float]) -> float | None:
    """Per-config mean wall time of finished trials, else the overall mean."""
    if not overall:
        return None
    fallback = sum(overall) / len(overall)
    total = 0.0
    for t in pending:
        key = tuple(sorted((k, str(v)) for k, v in t.config_labels.items()))
        vals = done.get(key)
        total += (sum(vals) / len(vals)) if vals else fallback
    return total if math.isfinite(total) else None


__all__ = ["Experiment", "estimate_seconds", "experiment", "resolve_goal"]
