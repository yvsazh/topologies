"""Trial execution and coordination.

``execute_trial`` runs one trial (any level) with failure classification,
OOM retries and an atomic ``result.json``; it is what both the sequential
coordinator and the process-pool workers call. Statuses:
``completed | failed | diverged | oom | timeout | pruned | interrupted``.

A ``KeyboardInterrupt`` marks the running trial ``interrupted``, writes its
result and propagates: completed trials are kept and ``run()`` resumes.
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from topologies.experiments.errors import ExperimentError, Pruned, TrialTimeout
from topologies.experiments.store import ExperimentStore, now_iso
from topologies.experiments.trial import (
    Runtime,
    Trial,
    TrialContext,
    best_of,
    environment,
    error_info,
    goal_value,
    is_oom,
)

RESUME_HINT = "resume with: exp.run() / topologies exp run <spec> (completed trials are kept)"


def free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch, "mps", None)
        if mps is not None and getattr(torch.backends, "mps", None) is not None and \
                torch.backends.mps.is_available():
            mps.empty_cache()
    except Exception:  # noqa: BLE001, S110 - cache release is advisory
        pass


def resolve_device(device: str | None) -> str:
    try:
        from topologies.training.device import resolve_device as _rd

        return str(_rd(device or "auto"))
    except Exception:  # noqa: BLE001
        return "cpu"


def _halve_batch(params: dict[str, Any], adjustments: list[dict]) -> bool:
    bs = int(params.get("trainer.batch_size", 32))
    if bs <= 1:
        return False
    acc = int(params.get("trainer.accumulation_steps", 1))
    params["trainer.batch_size"] = bs // 2
    params["trainer.accumulation_steps"] = acc * 2
    adjustments.append({"type": "halve_batch", "batch_size": [bs, bs // 2],
                        "accumulation_steps": [acc, acc * 2], "reason": "out of memory"})
    return True


def assemble_result(runtime: Runtime, trial: Trial, ctx: TrialContext | None, *,
                    status: str, status_reason: str | None, stop_reason: str | None,
                    out: dict | None, error: dict | None, started_at: str,
                    wall_s: float, adjustments: list[dict]) -> dict[str, Any]:
    from topologies.experiments.costs import compute_costs

    metric, mode = runtime.goal
    out = out or {}
    final = {k: v for k, v in (out.get("final") or {}).items()}
    history = list(out.get("history") or [])
    best: dict[str, Any] = {}
    for m in dict.fromkeys([metric, "val_loss"]):
        b = best_of(history, m, mode if m == metric else "min")
        if b is not None:
            best[m] = b
    trainer, model = out.get("trainer"), out.get("model")
    cost = compute_costs(ctx, trainer, model, wall_s=wall_s, train_s=out.get("train_s"),
                         eval_s=out.get("eval_s"), fit_out=out.get("fit_out")) \
        if ctx is not None else {"wall_s": wall_s}
    hook = getattr(runtime.task, "extra_costs", None) if ctx is not None else None
    if callable(hook):
        try:
            cost.update(hook(ctx) or {})
        except Exception:  # noqa: BLE001, S110 - costs are best-effort
            pass
    extras = dict(ctx.extras) if ctx is not None else {}
    if ctx is not None and ctx.info.get("param_matching"):
        extras["param_matching"] = ctx.info["param_matching"]
    device = ctx.device if ctx is not None else None
    precision = ctx.info.get("precision") if ctx is not None else None
    return {
        "format": "topologies-trial", "version": 1, "trial_id": trial.trial_id,
        "experiment": runtime.name, "labels": dict(trial.labels), "config": dict(trial.config),
        "status": status, "status_reason": status_reason, "stop_reason": stop_reason,
        "goal": {"metric": metric, "mode": mode, "select": runtime.select,
                 "value": goal_value(final, history, metric, mode, runtime.select)
                 if status != "failed" else None},
        "final": final, "best": best, "history": history, "cost": cost,
        "build": out.get("build"), "extras": extras, "adjustments": list(adjustments),
        "env": environment(device, precision), "started_at": started_at,
        "finished_at": now_iso(), "error": error,
    }


def execute_trial(runtime: Runtime, trial: Trial, *, device: str, cache: Any,
                  store: ExperimentStore, emit: Any = None,
                  peers: dict | None = None) -> dict[str, Any]:
    """Run one trial to a final status and write ``result.json`` (see module docstring)."""
    from topologies.experiments.costs import reset_memory
    from topologies.experiments.pipeline import call_trial_fn, run_pipeline

    started = now_iso()
    t0 = time.monotonic()
    store.write_config(trial.trial_id, {"trial_id": trial.trial_id, "experiment": runtime.name,
                                        "labels": trial.labels, "config": trial.config,
                                        "seed": trial.seed})
    params = dict(trial.params)
    adjustments: list[dict] = []
    attempt = 0
    while True:
        tr = replace(trial, params=dict(params))
        ctx = TrialContext(runtime, tr, device=device, cache=cache, emit=emit, peers=peers)
        ctx.info["t0"] = t0
        ctx.adjustments = adjustments
        ctx.dir.mkdir(parents=True, exist_ok=True)
        reset_memory(ctx)
        ctx.emit({"type": "running", "attempt": attempt})
        status, reason, error, out = "completed", None, None, None
        interrupted: BaseException | None = None
        try:
            if trial.plan_error:
                raise ExperimentError(trial.plan_error, stage="plan")
            out = call_trial_fn(ctx) if runtime.trial_fn is not None else run_pipeline(ctx)
            if out.get("paused"):
                status, reason = "pruned", "halving rung (paused)"
            elif ctx._status_hint is not None:
                status = ctx._status_hint
                reason = ctx.info.get("divergence") or ctx._stop_reason
        except KeyboardInterrupt as e:
            status, reason, interrupted = "interrupted", "KeyboardInterrupt", e
        except Pruned as e:
            status, reason = "pruned", e.reason
        except TrialTimeout as e:
            status, reason, error = "timeout", str(e), None
        except Exception as e:  # noqa: BLE001 - one failing trial never kills the experiment
            if is_oom(e):
                if runtime.on_oom == "halve_batch" and attempt < 2 and \
                        runtime.trial_fn is None and _halve_batch(params, adjustments):
                    attempt += 1
                    del ctx, out
                    free_memory()
                    continue
                status, reason = "oom", "out of memory"
            elif ctx._status_hint is not None:
                status, reason = ctx._status_hint, ctx.info.get("divergence") or str(e)
            else:
                status, reason = "failed", f"{type(e).__name__}: " + (
                    str(e).strip().splitlines() or [""])[0][:300]
            error = error_info(e)
        stop_reason = None
        if out is not None:
            fo = out.get("fit_out") or {}
            stop_reason = fo.get("stop_reason") or ctx._stop_reason or (
                "returned" if runtime.trial_fn is not None else None)
        elif ctx._stop_reason:
            stop_reason = ctx._stop_reason
        if status == "completed" and runtime.save in ("all", "best") and out and \
                out.get("trainer") is not None:
            try:
                ctx.extras["checkpoint"] = save_trial_model(out["trainer"], ctx.dir / "model",
                                                            trial.trial_id)
            except Exception as e:  # noqa: BLE001
                ctx.extras["save_error"] = f"{type(e).__name__}: {e}"
        result = assemble_result(runtime, trial, ctx, status=status, status_reason=reason,
                                 stop_reason=stop_reason, out=out, error=error,
                                 started_at=started, wall_s=time.monotonic() - t0,
                                 adjustments=adjustments)
        store.write_result(trial.trial_id, result)
        if out is not None:
            out.pop("trainer", None)
            out.pop("model", None)
        del out, ctx
        free_memory()
        if interrupted is not None:
            raise interrupted
        return result


def save_trial_model(trainer: Any, path: Path, trial_id: str) -> str:
    """Save a trial's ``model/`` bundle; returns how it can be reloaded.

    ``"self-contained"``: the constructor call is recorded (default model
    stage, ``build()``, ``Configurable`` classes), ``load_model()`` rebuilds
    it. ``"weights-only"``: a custom ``build_model(ctx)`` returned a directly
    constructed module, so reloading needs ``load_model(model=...)``; return
    ``topologies.build(MyModel, ...)`` from ``build_model`` to avoid that.
    """
    import warnings

    from topologies.pretrained import read_config, self_contained

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # reported via the return value
        trainer.save_pretrained(path, trial_id=trial_id)
    return "self-contained" if self_contained(read_config(path).get("model")) else \
        "weights-only"


def goal_series(result: dict) -> dict[int, float]:
    """``{epoch: goal value}`` from a result's history (for pruning)."""
    metric = (result.get("goal") or {}).get("metric")
    out: dict[int, float] = {}
    for row in result.get("history") or []:
        v = row.get(metric)
        if isinstance(v, (int, float)) and math.isfinite(float(v)) and "epoch" in row:
            out[int(row["epoch"])] = float(v)
    return out


__all__ = ["RESUME_HINT", "assemble_result", "execute_trial", "free_memory", "goal_series",
           "resolve_device", "save_trial_model"]
