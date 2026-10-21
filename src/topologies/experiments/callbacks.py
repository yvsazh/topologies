"""Trainer callbacks that connect a trial to the experiment.

- ``ProgressBridge``: live progress + per-epoch rows (with derived metrics)
  to the coordinator; samples peak memory.
- ``DivergenceGuard``: non-finite loss / grad norm, or loss above 1e4 x its
  first value -> stop, status ``diverged``.
- ``TimeBudget``: wall-clock budgets (``train_for="10m"``, stop reason
  ``"time budget"``) and ``trial_timeout`` (status ``timeout``).
- ``PruneCallback``: asks the pruner at every epoch end (status ``pruned``).
- ``RungStop``: successive-halving rung boundary (training pauses there).

All of them follow the observability contract: an exception inside a
callback is swallowed by ``CallbackList`` and never breaks training.
"""

from __future__ import annotations

import math
import time
from typing import Any

from topologies.training.callbacks import Callback


def _finite(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return True


class ProgressBridge(Callback):
    """Forwards progress to the coordinator (throttled to ``max_hz``)."""

    def __init__(self, ctx: Any, *, max_hz: float = 4.0) -> None:
        self.ctx = ctx
        self.interval = 1.0 / max(0.1, float(max_hz))
        self._last = 0.0
        self._last_mem = 0.0

    def on_train_begin(self, trainer: Any) -> None:
        spe = None
        try:
            spe = len(trainer.train_loader)
        except TypeError:  # iterable without a length
            pass
        total = spe * int(trainer.config.max_epochs) if spe else None
        if trainer.config.max_steps is not None:
            total = min(total or trainer.config.max_steps, trainer.config.max_steps)
        self.ctx.emit({"type": "start", "max_epochs": int(trainer.config.max_epochs),
                       "steps_per_epoch": spe, "total_steps": total,
                       "global_step": int(trainer.global_step)})

    def on_step_end(self, trainer: Any, step: int, metrics: dict) -> None:
        now = time.monotonic()
        if now - self._last_mem > 2.0:
            self._last_mem = now
            from topologies.experiments.costs import sample_memory

            sample_memory(self.ctx)
        if now - self._last < self.interval:
            return
        self._last = now
        self.ctx.emit({"type": "progress", "global_step": int(trainer.global_step),
                       "epoch": int(trainer.current_epoch),
                       "metrics": {k: float(v) for k, v in metrics.items()
                                   if isinstance(v, (int, float))}})

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict) -> None:
        self.ctx.record_epoch({"epoch": int(epoch), **{k: v for k, v in metrics.items()}})


class DivergenceGuard(Callback):
    """Stops a run whose loss explodes (status ``diverged``)."""

    def __init__(self, ctx: Any, *, factor: float = 1e4) -> None:
        self.ctx = ctx
        self.factor = float(factor)
        self.first: float | None = None

    def on_step_end(self, trainer: Any, step: int, metrics: dict) -> None:
        loss, gn = metrics.get("loss"), metrics.get("grad_norm")
        reason = None
        if loss is not None and not _finite(loss):
            reason = f"non-finite loss ({loss})"
        elif gn is not None and not _finite(gn):
            reason = f"non-finite grad_norm ({gn})"
        elif loss is not None:
            lv = float(loss)
            if self.first is None:
                self.first = abs(lv)
            elif abs(lv) > self.factor * max(self.first, 1e-8):
                reason = f"loss {lv:.3g} > {self.factor:g} x first loss {self.first:.3g}"
        if reason is not None:
            self.ctx.info["divergence"] = reason
            self.ctx.request_stop("diverged", status="diverged")
            trainer.request_stop("diverged")


class TimeBudget(Callback):
    """Requests a stop once ``seconds`` have passed (since train begin or ``t0``)."""

    def __init__(self, ctx: Any, seconds: float, *, reason: str = "time budget",
                 status: str | None = None, t0: float | None = None) -> None:
        self.ctx = ctx
        self.seconds = float(seconds)
        self.reason = reason
        self.status = status
        self.t0 = t0

    def on_train_begin(self, trainer: Any) -> None:
        if self.t0 is None:
            self.t0 = time.monotonic()

    def on_step_end(self, trainer: Any, step: int, metrics: dict) -> None:
        if self.t0 is not None and time.monotonic() - self.t0 >= self.seconds:
            self.ctx.request_stop(self.reason, status=self.status)
            trainer.request_stop(self.reason)


class PruneCallback(Callback):
    """Consults the experiment's pruner at every epoch end."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict) -> None:
        row = self.ctx.history[-1] if self.ctx.history else {"epoch": epoch, **metrics}
        if self.ctx.check_prune(int(epoch), row):
            trainer.request_stop("pruned")


class RungStop(Callback):
    """Pauses training at a successive-halving rung boundary (after ``epochs`` epochs)."""

    def __init__(self, ctx: Any, epochs: int) -> None:
        self.ctx = ctx
        self.epochs = int(epochs)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict) -> None:
        if epoch + 1 >= self.epochs and epoch + 1 < int(trainer.config.max_epochs):
            self.ctx.info["rung_paused"] = True
            trainer.request_stop("halving rung")


def trial_callbacks(ctx: Any) -> list[Callback]:
    rt = ctx.runtime
    cbs: list[Callback] = [ProgressBridge(ctx), DivergenceGuard(ctx)]
    budget = ctx.budget
    if budget.get("time"):
        cbs.append(TimeBudget(ctx, budget["time"], reason="time budget"))
    if rt.trial_timeout:
        cbs.append(TimeBudget(ctx, rt.trial_timeout, reason="timeout", status="timeout",
                              t0=ctx.info.get("t0")))
    if rt.pruner is not None:
        cbs.append(PruneCallback(ctx))
    if rt.rung and rt.rung.get("epochs"):
        cbs.append(RungStop(ctx, rt.rung["epochs"]))
    return cbs


__all__ = ["DivergenceGuard", "ProgressBridge", "PruneCallback", "RungStop", "TimeBudget",
           "trial_callbacks"]
