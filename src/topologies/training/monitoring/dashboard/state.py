"""``DashboardState``: compatibility view over a :class:`TrainingTracker`.

Old attributes (``latest_metrics``, ``train_history``, ``epoch_summaries`` ...) are read
from the tracker; ``record_step``/``record_validation``/``record_checkpoint`` feed it
through the normal event path.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from topologies.visualization.core import EventKind, Progress, VisualizationEvent
from topologies.visualization.state import TrainingTracker

_MAX_EVENTS = 200


class DashboardState:
    """Legacy dashboard state backed by a ``TrainingTracker`` (``.tracker``)."""

    def __init__(self, tracker: TrainingTracker | None = None, **fields: Any) -> None:
        self.tracker = tracker if tracker is not None else TrainingTracker()
        self.system: dict[str, Any] = {}
        self.recent_events: deque = deque(maxlen=_MAX_EVENTS)
        for key, value in fields.items():
            setattr(self, key, value)

    # -- read-only views -----------------------------------------------------
    # -- progress outside the step loop / experiment plan (stored on the tracker) --
    def set_activity(self, label: str, done: int | None = None, total: int | None = None) -> None:
        self.tracker.set_activity(label, done, total)

    @property
    def activity(self) -> str | None:
        act = self.tracker.activity
        return None if act is None else act["label"]

    def _plan_part(self, index: int) -> int | None:
        plan = self.tracker.plan
        return None if plan is None else plan[index]

    def _set_plan_part(self, index: int, value: Any) -> None:
        plan = list(self.tracker.plan or (0, 0, None))
        plan[index] = None if value is None else int(value)
        self.tracker.plan = tuple(plan)

    plan_index = property(lambda self: self._plan_part(0),
                          lambda self, v: self._set_plan_part(0, v))
    plan_total = property(lambda self: self._plan_part(1),
                          lambda self, v: self._set_plan_part(1, v))
    plan_done = property(lambda self: self._plan_part(2),
                         lambda self, v: self._set_plan_part(2, v))

    @property
    def run_info(self) -> Any:
        return self.tracker.run

    @run_info.setter
    def run_info(self, value: Any) -> None:
        self.tracker.run = value

    @property
    def current_epoch(self) -> int:
        return self.tracker.epoch

    @property
    def current_step(self) -> int:
        return self.tracker.step

    @property
    def latest_metrics(self) -> dict[str, float]:
        return {k: v[0] for k, v in self.tracker._latest.items()}

    def _history(self, *names: str) -> list[float]:
        for name in names:
            series = self.tracker.series(name)
            if series:
                return list(series.y)
        return []

    @property
    def train_history(self) -> list[float]:
        return self._history("loss", "train_loss")

    @property
    def val_history(self) -> list[float]:
        return self._history("val_loss", "val")

    @property
    def lr_history(self) -> list[float]:
        return self._history("lr", "learning_rate")

    @property
    def custom_histories(self) -> dict[str, list[float]]:
        skip = {"loss", "train_loss", "lr", "learning_rate"}
        return {n: list(self.tracker.series(n).y) for n in self.tracker.metric_names("train")
                if n not in skip}

    @property
    def val_histories(self) -> dict[str, list[float]]:
        return {n: list(self.tracker.series(n).y) for n in self.tracker.metric_names("val")}

    @property
    def best_metric(self) -> float | None:
        return self.tracker.best_metric

    @best_metric.setter
    def best_metric(self, value: float | None) -> None:
        self.tracker.best_metric = value

    @property
    def best_epoch(self) -> int | None:
        return self.tracker.best_epoch

    @property
    def checkpoints(self) -> list[dict]:
        return self.tracker.checkpoints

    @property
    def elapsed(self) -> float:
        return self.tracker.elapsed

    @property
    def eta(self) -> float | None:
        return self.tracker.eta

    @property
    def total_steps(self) -> int | None:
        return self.tracker.total_steps

    @property
    def total_epochs(self) -> int | None:
        return self.tracker.total_epochs

    @property
    def epoch_step(self) -> int:
        return self.tracker.epoch_step

    @property
    def steps_per_epoch(self) -> int | None:
        return self.tracker.steps_per_epoch

    @property
    def it_per_sec(self) -> float | None:
        return self.tracker.throughput

    @property
    def finished(self) -> bool:
        return self.tracker.status == "done"

    @property
    def epoch_summaries(self) -> list[dict]:
        return [{"epoch": r.epoch, "metrics": dict(r.metrics), "duration": r.duration or 0.0}
                for r in self.tracker.epochs]

    # -- legacy writers ------------------------------------------------------
    def _update(self, kind: str, *, metrics: dict | None = None, **data: Any) -> None:
        step = data.get("global_step")
        progress = Progress(step, self.tracker.total_steps) if step is not None else None
        self.tracker.update(VisualizationEvent("training", kind, metrics=metrics or {},
                                               progress=progress, data=data))

    def record_step(self, step: int, epoch: int, metrics: dict, *, epoch_step: int | None = None,
                    steps_per_epoch: int | None = None) -> None:
        if self.tracker.started is None:
            self.tracker.started = self.tracker.clock()
            self.tracker.status = "training"
        try:
            self._update(EventKind.STEP, metrics=dict(metrics or {}), global_step=int(step),
                         epoch=int(epoch), epoch_step=epoch_step,
                         steps_per_epoch=steps_per_epoch)
        except Exception:  # noqa: BLE001, S110
            pass

    def record_validation(self, metrics: dict) -> None:
        try:
            self._update(EventKind.VALIDATION_END, metrics=dict(metrics or {}),
                         global_step=self.tracker.step)
        except Exception:  # noqa: BLE001, S110
            pass

    def record_checkpoint(self, path: str, epoch: int, step: int, is_best: bool) -> None:
        try:
            self._update(EventKind.CHECKPOINT, path=str(path), epoch=int(epoch),
                         global_step=int(step), is_best=bool(is_best))
        except Exception:  # noqa: BLE001, S110
            pass

    def pause_throughput(self) -> None:
        """Exclude the next step delta (validation/checkpoint pauses) from it/s."""
        self.tracker._paused = True

    def eta_seconds(self) -> float | None:
        return self.tracker.eta

    def log_event(self, text: str) -> None:
        try:
            self.recent_events.append(str(text))
            self.tracker.add_log(str(text), "event")
        except Exception:  # noqa: BLE001, S110
            pass
