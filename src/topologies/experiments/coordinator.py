"""Coordinator: decide what runs, run it (sequential / process pool / successive
halving), keep ``trials.jsonl`` and the live display up to date, and return the
plan's ``ResultSet``. Only the coordinator appends to ``trials.jsonl``.
"""

from __future__ import annotations

import shutil
import sys
from typing import Any

from topologies.experiments.cache import stable_hash
from topologies.experiments.errors import ExperimentError
from topologies.experiments.runner import RESUME_HINT, execute_trial, goal_series, resolve_device
from topologies.experiments.trial import Trial

_RUN_SELECTION_UNSET = object()


def trial_device(exp: Any, trial: Trial) -> str:
    return resolve_device(exp.devices[0] if exp.devices else trial.params.get("trainer.device"))


def stored_peers(exp: Any, exclude: set[str]) -> dict[str, dict[int, float]]:
    """Goal-per-epoch of finished trials of this experiment (pruning reference)."""
    peers: dict[str, dict[int, float]] = {}
    for tid in exp.store.trial_ids():
        if tid in exclude:
            continue
        res = exp.store.read_result(tid, warn=False)
        if res and res.get("status") in ("completed", "pruned", "timeout") and \
                (res.get("goal") or {}).get("metric") == exp.goal[0]:
            peers[tid] = goal_series(res)
    return peers


def _write_spec(exp: Any, trials: list[Trial]) -> None:
    spec = exp.spec()
    spec_hash = stable_hash(spec)
    spec["plan"] = [t.trial_id for t in trials]
    exp.store.write_spec(spec, spec_hash)


def _after_trial(exp: Any, state: Any, trial: Trial, result: dict,
                 peers: dict[str, dict[int, float]]) -> None:
    exp.store.log_event(trial.trial_id, result["status"],
                        value=(result.get("goal") or {}).get("value"),
                        reason=result.get("status_reason"))
    peers[trial.trial_id] = goal_series(result)
    state.trial_done(trial, result)
    if exp.on_error == "raise" and result["status"] not in ("completed", "pruned"):
        err = result.get("error") or {}
        raise ExperimentError(
            f"Trial {trial.trial_id} ({trial.describe()}) ended {result['status']}: "
            f"{err.get('type', '')}: {err.get('message', result.get('status_reason'))}",
            stage="run", trial=trial.trial_id,
            hint="on_error='continue' (default) records failures and keeps going.")


def stamp_plan(exp: Any, trial: Trial) -> None:
    """Record where ``trial`` sits in the full plan, and how many are already done."""
    trials = exp.trials()
    done = 0
    for item in trials:
        if item.trial_id == trial.trial_id:
            continue
        res = exp.store.read_result(item.trial_id, warn=False)
        if res and res.get("status") == "completed":
            done += 1
    trial.plan_total = len(trials)
    trial.plan_done = done


def run_sequential(exp: Any, pending: list[Trial], state: Any,
                   peers: dict[str, dict[int, float]], *, runtime: Any = None) -> None:
    runtime = runtime or exp.runtime()
    for t in pending:
        stamp_plan(exp, t)
        exp.store.log_event(t.trial_id, "running")
        state.trial_start(t)
        try:
            result = execute_trial(runtime, t, device=trial_device(exp, t), cache=exp.cache,
                                   store=exp.store, emit=state.handle, peers=peers)
        except KeyboardInterrupt:
            exp.store.log_event(t.trial_id, "interrupted")
            state.trial_interrupted(t)
            raise
        _after_trial(exp, state, t, result, peers)


def run_halving(exp: Any, pending: list[Trial], state: Any,
                peers: dict[str, dict[int, float]]) -> None:
    """Successive halving: rung budgets ``min, min*eta, ...``; keep ``ceil(n/eta)`` configs.

    Promoted trials continue from ``<trial>/ckpt/last.pt``; the others end
    ``pruned`` with ``status_reason="halving rung k"``.
    """
    from topologies.experiments.pruners import halving_rungs, keep_count, rank_configs
    from topologies.experiments.routing import parse_train_for
    from topologies.experiments.stats import mean

    final = exp.budget.get("epochs")
    if not final or set(exp.budget) - {"epochs"}:
        raise ExperimentError("strategy='halving' needs an epoch budget, e.g. "
                              "train_for='9 epochs'.", stage="config", key="train_for")
    min_ep = 1
    if exp.min_budget is not None:
        mb = parse_train_for(exp.min_budget)
        if set(mb) != {"epochs"}:
            raise ExperimentError("min_budget must be in epochs.", stage="config")
        min_ep = int(mb["epochs"])
    rungs = halving_rungs(int(final), eta=exp.eta, min_epochs=min_ep)
    state.info["rungs"] = rungs
    pending_ids = {t.trial_id for t in pending}
    all_trials = [t for t in exp.trials()]
    configs: dict[tuple, list[Trial]] = {}
    for t in all_trials:
        configs.setdefault(tuple(sorted((k, str(v)) for k, v in t.config_labels.items())),
                           []).append(t)
    active = list(configs)
    metric, mode = exp.goal
    for k, budget in enumerate(rungs):
        is_last = k == len(rungs) - 1
        runtime = exp.runtime(rung={"epochs": budget, "final_epochs": int(final),
                                    "resume": k > 0, "index": k})
        todo = []
        for key in active:
            for t in configs[key]:
                res = exp.store.read_result(t.trial_id, warn=False)
                epochs_done = len((res or {}).get("history") or [])
                status = (res or {}).get("status")
                if status == "completed" or (epochs_done >= budget and not is_last
                                             and status == "pruned"):
                    continue
                if t.trial_id not in pending_ids and status in ("failed", "diverged", "oom",
                                                                "timeout"):
                    continue
                todo.append(t)
        state.info["rung"] = f"{k + 1}/{len(rungs)} ({budget} epochs)"
        run_sequential(exp, todo, state, peers, runtime=runtime)
        if is_last:
            break
        scores: dict[tuple, float | None] = {}
        for key in active:
            vals = []
            for t in configs[key]:
                res = exp.store.read_result(t.trial_id, warn=False) or {}
                hist = [h for h in res.get("history") or [] if int(h.get("epoch", -1)) <
                        budget]
                if hist and isinstance(hist[-1].get(metric), (int, float)):
                    vals.append(float(hist[-1][metric]))
            scores[key] = mean(vals)
        ranked = rank_configs(scores, mode)
        keep = ranked[:keep_count(len(active), exp.eta)]
        for key in ranked[len(keep):]:
            for t in configs[key]:
                res = exp.store.read_result(t.trial_id, warn=False)
                if res and res.get("status") in ("pruned", "completed"):
                    res["status"] = "pruned"
                    res["status_reason"] = f"halving rung {k}"
                    exp.store.write_result(t.trial_id, res)
                    exp.store.log_event(t.trial_id, "pruned", reason=f"halving rung {k}")
        state.info.setdefault("kept", []).append(len(keep))
        active = keep


def prune_saved_models(exp: Any) -> None:
    """``save="best"``: keep only the best completed trial's ``model/`` bundle."""
    res = exp.results()
    best = res.best(1)
    keep = None
    if best:
        sign = 1.0 if exp.goal[1] == "min" else -1.0
        cands = [t for t in best[0].completed if t.value is not None]
        if cands:
            keep = min(cands, key=lambda t: sign * t.value).trial_id
    for t in exp.trials():
        d = exp.store.trial_dir(t.trial_id) / "model"
        if t.trial_id != keep and d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


class _Info(dict):
    """``state.info`` mapping that also announces changes as ``experiment.info`` events."""

    def __init__(self, emit: Any) -> None:
        super().__init__()
        self._emit = emit

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        self._emit({"type": "info", key: value})


class RunEvents:
    """Coordinator-side event emitter (the ``state`` the runners talk to).

    ``trial_start``/``handle``/``trial_done``/``trial_interrupted`` turn runner
    callbacks into normalized visualization events; nothing here renders.
    """

    def __init__(self, exp: Any, shared: Any, trials: list[Trial], pending: list[Trial]) -> None:
        self.exp = exp
        self.shared = shared
        self.trials = trials
        self.pending = list(pending)
        self.run_id = f"{exp.store.root}:{exp.name}"
        self.index = {t.trial_id: i + 1 for i, t in enumerate(trials)}
        self.running: dict[str, Trial] = {}
        self.started = 0
        self.best: float | None = None
        self.info: dict[str, Any] = _Info(self._emit_raw)

    def _emit_raw(self, event: dict[str, Any], trial_id: str | None = None) -> None:
        from topologies.visualization import normalize_event

        self.shared.handle_safe(normalize_event(event, source="experiment", run_id=self.run_id,
                                                trial_id=trial_id))

    def _better(self, value: float, other: float | None) -> bool:
        return other is None or (value < other if self.exp.goal[1] == "min" else value > other)

    # -- experiment lifecycle ----------------------------------------------------
    def started_event(self) -> None:
        exp = self.exp
        pending_ids = {t.trial_id for t in self.pending}
        cached = []
        for t in self.trials:
            if t.trial_id in pending_ids:
                continue
            res = exp.store.read_result(t.trial_id, warn=False)
            if res is None:
                continue
            goal = res.get("goal") or {}
            value = goal.get("value")
            cached.append({"trial_id": t.trial_id, "index": self.index[t.trial_id],
                           "labels": dict(t.labels), "seed": t.seed,
                           "status": res.get("status"), "goal_value": value,
                           "wall_s": (res.get("cost") or {}).get("wall_s")})
            if res.get("status") == "completed" and goal.get("metric") == exp.goal[0] and \
                    isinstance(value, (int, float)) and self._better(value, self.best):
                self.best = value
        axes = {a.name: list(a.labels) for a in exp.axes}
        self._emit_raw({
            "type": "started", "current": 0, "total": len(self.trials), "name": exp.name,
            "goal": list(exp.goal), "axes": axes,
            "varying": {k: v for k, v in axes.items() if len(v) > 1},
            "pending_ids": [t.trial_id for t in self.pending], "cached": cached,
            "trials": [{"trial_id": t.trial_id, "index": self.index[t.trial_id],
                        "labels": dict(t.labels), "seed": t.seed} for t in self.trials],
            "workers": int(getattr(exp, "workers", 1) or 1),
            "seeds": len(getattr(exp, "seeds", ()) or ()) or 1,
            "strategy": getattr(exp, "strategy", None), "durations": {},
        })

    # -- runner callbacks ----------------------------------------------------------
    def trial_start(self, trial: Trial) -> None:
        self.started += 1
        self.running[trial.trial_id] = trial
        self._emit_raw({"type": "trial_started", "trial_id": trial.trial_id,
                        "index": self.index.get(trial.trial_id), "labels": dict(trial.labels),
                        "config": trial.config_labels, "seed": trial.seed, "trial": trial,
                        "current": self.started, "total": len(self.trials)}, trial.trial_id)

    def handle(self, event: dict[str, Any]) -> None:
        tid = event.get("trial_id")
        trial = self.running.get(tid) if tid is not None else None
        payload = {**event, "config": dict(getattr(trial, "config_labels", {}) or {})}
        self._emit_raw(payload, tid)

    def trial_done(self, trial: Trial, result: dict[str, Any]) -> None:
        self.running.pop(trial.trial_id, None)
        goal_metric, mode = self.exp.goal
        metrics = {**(result.get("final") or {})}
        value = (result.get("goal") or {}).get("value")
        if isinstance(value, (int, float)):
            metrics[goal_metric] = value
        status = result.get("status")
        self._emit_raw({"type": "trial_completed" if status == "completed" else "trial_failed",
                        "trial_id": trial.trial_id, "status": status,
                        "config": trial.config_labels, "trial": trial, "result": result,
                        "metrics": metrics, "goal_metric": goal_metric, "mode": mode},
                       trial.trial_id)
        self._emit_raw({"type": "leaderboard_changed", "trial_id": trial.trial_id,
                        "status": status}, trial.trial_id)
        if status == "completed" and isinstance(value, (int, float)) and \
                self._better(value, self.best):
            self.best = value
            self._emit_raw({"type": "best_configuration_changed", "trial_id": trial.trial_id,
                            "config": trial.config_labels, "metric": goal_metric,
                            "value": value, "mode": mode}, trial.trial_id)

    def trial_interrupted(self, trial: Trial) -> None:
        self.running.pop(trial.trial_id, None)
        self._emit_raw({"type": "trial_interrupted", "trial_id": trial.trial_id,
                        "status": "interrupted", "trial": trial,
                        "config": trial.config_labels}, trial.trial_id)
        self._emit_raw({"type": "leaderboard_changed", "trial_id": trial.trial_id,
                        "status": "interrupted"}, trial.trial_id)


def _selection_sinks(exp: Any, selection: Any) -> Any:
    from topologies.experiments.core import _VISUALIZERS_UNSET
    from topologies.visualization import create_visualizer
    from topologies.visualization.experiment_views import make_experiment_view

    if selection is _VISUALIZERS_UNSET:
        options = dict(getattr(exp, "monitoring", None) or {}) if exp.monitor == "dashboard" else {}
        view = make_experiment_view(exp.monitor, **options)
        return create_visualizer([view] if view is not None else [])
    return create_visualizer(selection)


def run_experiment(exp: Any, *, rerun: str | None = None, limit: int | None = None,
                   visualizer: Any = _RUN_SELECTION_UNSET) -> Any:
    from topologies.experiments.core import _VISUALIZERS_UNSET
    from topologies.experiments.planning import make_plan
    from topologies.visualization import SessionInfo, adapt_result

    selection = getattr(exp, "_visualizer_selection", _VISUALIZERS_UNSET)
    if visualizer is not _RUN_SELECTION_UNSET:
        selection = visualizer
    # Resolve names/types before planning can write or archive any run data.
    shared = _selection_sinks(exp, selection)
    plan = make_plan(exp)
    trials = plan.trials
    rerun_pending = exp.pending(rerun)
    pending = rerun_pending
    if limit is not None:
        pending = pending[: max(0, int(limit))]
    events = RunEvents(exp, shared, trials, pending)
    peers = stored_peers(exp, exclude={t.trial_id for t in pending})
    interrupted = False
    try:
        shared.prepare(source="experiment", context=SessionInfo(
            source="experiment", run_id=events.run_id, title=exp.name, goal=tuple(exp.goal),
            axes={a.name: tuple(a.labels) for a in exp.axes}, total_trials=len(trials),
            options={"monitor": exp.monitor},
            extra={"experiment": exp.name, "config": exp.spec()}))
        _write_spec(exp, trials)
        if rerun == "all":
            # Archive the full rerun selection, not only the limited execution
            # subset; trials beyond the limit remain pending without losing
            # their previous result. Drop checkpoints so the retry starts over.
            for trial in rerun_pending:
                exp.store.archive_result(trial.trial_id)
                exp.store.clear_checkpoint(trial.trial_id)
        elif rerun == "failed":
            from topologies.experiments.trial import FINAL_FAILURES

            for trial in rerun_pending:
                previous = exp.store.read_result(trial.trial_id, warn=False) or {}
                if previous.get("status") in FINAL_FAILURES:
                    exp.store.clear_checkpoint(trial.trial_id)
        kept = len(trials) - len(rerun_pending)
        if exp.monitor != "none":
            print(f"{exp.name}: {kept} completed, {len(pending)} remaining", file=sys.stderr)
        for t in pending:
            exp.store.log_event(t.trial_id, "queued")
        events.started_event()
        if exp.strategy == "halving":
            run_halving(exp, pending, events, peers)
        elif exp.workers > 1 or (exp.devices and len(exp.devices) > 1):
            from topologies.experiments.parallel import run_parallel

            run_parallel(exp, pending, events, peers)
        else:
            run_sequential(exp, pending, events, peers)
        if exp.save == "best":
            prune_saved_models(exp)
        res = exp.results()
        shared.render_safe(adapt_result(res))
        events._emit_raw({"type": "completed", "result": res})
    except BaseException as exc:
        events._emit_raw({"type": "error", "error": type(exc).__name__, "message": str(exc)})
        interrupted = isinstance(exc, KeyboardInterrupt)
        raise
    finally:
        shared.close()
        if interrupted:
            # After the visualizers closed, so the hint lands on the normal screen.
            print(f"\ninterrupted. {RESUME_HINT}", file=sys.stderr)
    return res


__all__ = ["RunEvents", "prune_saved_models", "run_experiment", "run_halving",
           "run_sequential", "stored_peers", "trial_device"]
