"""Shared run-state trackers built from visualization events (stdlib-only).

Built-in and custom visualizers read progress, ETA, throughput, metric histories, best
values and experiment leaderboards from these trackers instead of re-deriving them
from raw events. ``update(event)`` is cheap (microseconds) and runs on the producer
thread under ``tracker.lock``; renderers read under the same lock.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from topologies.visualization.core import EventKind
from topologies.visualization.format import direction

RAW_POINTS = 512
HISTORY_POINTS = 4096


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


@dataclass(frozen=True)
class Series:
    """A metric history: ``x``/``y`` (bucket means for old points), ``split`` train/val."""

    name: str
    x: tuple[float, ...] = ()
    y: tuple[float, ...] = ()
    split: str = "train"
    ymin: tuple[float, ...] = ()
    ymax: tuple[float, ...] = ()

    @property
    def points(self) -> list[tuple[float, float]]:
        return list(zip(self.x, self.y))

    def __len__(self) -> int:
        return len(self.x)

    def __bool__(self) -> bool:
        return bool(self.x)

    def thin(self, n: int) -> list[tuple[float, float]]:
        """At most ``2 n`` points: per bucket the min and max (in x order), for charts."""
        size = len(self.x)
        if size <= 2 * n or n <= 0:
            return list(zip(self.x, self.y))
        out: list[tuple[float, float]] = []
        xs, ys = self.x, self.y
        step = size / n
        for b in range(n):
            lo_i = int(b * step)
            hi_i = max(lo_i + 1, int((b + 1) * step))
            chunk = range(lo_i, min(size, hi_i))
            i_min = min(chunk, key=ys.__getitem__)
            i_max = max(chunk, key=ys.__getitem__)
            for i in sorted({i_min, i_max}):
                out.append((xs[i], ys[i]))
        return out


class SeriesStore:
    """Last ``RAW_POINTS`` points exactly + min/max/mean buckets bounded to ``HISTORY_POINTS``."""

    __slots__ = (
        "_snap",
        "bucket",
        "count",
        "hhi",
        "hlo",
        "hn",
        "hsum",
        "hx",
        "name",
        "raw",
        "split",
    )

    def __init__(self, name: str, split: str = "train") -> None:
        self.name = name
        self.split = split
        self.raw: deque[tuple[float, float]] = deque()
        # History buckets as parallel lists: last x, min, max, sum, count.
        self.hx: list[float] = []
        self.hlo: list[float] = []
        self.hhi: list[float] = []
        self.hsum: list[float] = []
        self.hn: list[int] = []
        self.bucket = 1
        self.count = 0
        self._snap: _Snap | None = None

    @property
    def history(self) -> list[list[float]]:
        return [[x, lo, hi, sm, n] for x, lo, hi, sm, n in
                zip(self.hx, self.hlo, self.hhi, self.hsum, self.hn)]

    def add(self, x: float, y: float) -> None:
        self.count += 1
        raw = self.raw
        raw.append((x, y))
        if len(raw) <= RAW_POINTS:
            return
        ox, oy = raw.popleft()
        if self.hn and self.hn[-1] < self.bucket:
            self.hx[-1] = ox
            self.hlo[-1] = min(self.hlo[-1], oy)
            self.hhi[-1] = max(self.hhi[-1], oy)
            self.hsum[-1] += oy
            self.hn[-1] += 1
            return
        self.hx.append(ox)
        self.hlo.append(oy)
        self.hhi.append(oy)
        self.hsum.append(oy)
        self.hn.append(1)
        if len(self.hn) > HISTORY_POINTS:
            self._merge()

    def _merge(self) -> None:
        n = len(self.hn) - len(self.hn) % 2
        hx = [self.hx[i + 1] for i in range(0, n, 2)]
        hlo = [min(self.hlo[i], self.hlo[i + 1]) for i in range(0, n, 2)]
        hhi = [max(self.hhi[i], self.hhi[i + 1]) for i in range(0, n, 2)]
        hsum = [self.hsum[i] + self.hsum[i + 1] for i in range(0, n, 2)]
        hn = [self.hn[i] + self.hn[i + 1] for i in range(0, n, 2)]
        if len(self.hn) % 2:
            hx.append(self.hx[-1])
            hlo.append(self.hlo[-1])
            hhi.append(self.hhi[-1])
            hsum.append(self.hsum[-1])
            hn.append(self.hn[-1])
        self.hx, self.hlo, self.hhi, self.hsum, self.hn = hx, hlo, hhi, hsum, hn
        self.bucket *= 2

    def snapshot(self) -> Series:
        snap = self._snap
        if snap is not None and snap.count == self.count:
            return snap.series
        series = self._build()
        self._snap = _Snap(self.count, series)
        return series

    def _build(self) -> Series:
        rx = [p[0] for p in self.raw]
        ry = [p[1] for p in self.raw]
        means = [sm / n for sm, n in zip(self.hsum, self.hn)]
        return Series(self.name, tuple(self.hx + rx), tuple(means + ry), self.split,
                      tuple(self.hlo + ry), tuple(self.hhi + ry))

    def __len__(self) -> int:
        return len(self.hn) + len(self.raw)


@dataclass(frozen=True)
class _Snap:
    count: int
    series: Series


@dataclass
class EpochRow:
    epoch: int
    metrics: dict[str, float] = field(default_factory=dict)
    duration: float | None = None
    is_best: bool = False
    checkpoint: str | None = None
    best_metric: float | None = None
    step: int | None = None


def _info_view(info: Any) -> Any:
    if info is None:
        return None
    if isinstance(info, Mapping):
        return SimpleNamespace(**dict(info))
    return info


class TrainingTracker:
    """Training run state: progress, timing, metric series, epochs, best values, log."""

    def __init__(self, *, ema: float = 0.9, log_size: int = 500,
                 clock: Any = time.monotonic) -> None:
        self.lock = threading.RLock()
        self.clock = clock
        self.ema_weight = ema
        self.version = 0
        self.run: Any = None
        self.status = "idle"
        self.step = 0
        self.total_steps: int | None = None
        self.epoch = 0
        self.total_epochs: int | None = None
        self.epoch_step = 0
        self.steps_per_epoch: int | None = None
        self.throughput: float | None = None
        self.epochs: list[EpochRow] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.log: deque[tuple[float, str, str]] = deque(maxlen=log_size)
        self.best_metric: float | None = None
        self.best_epoch: int | None = None
        self.monitored: str | None = None
        self.duration: float | None = None
        self.error: str | None = None
        self.last_event: str | None = None
        self.started: float | None = None
        self.finished: float | None = None
        self._series: dict[str, SeriesStore] = {}
        self._latest: dict[str, tuple[float, int, int]] = {}
        self._best: dict[str, tuple[float, int, int]] = {}
        self._ema: dict[str, float] = {}
        self._directions: dict[str, str] = {}
        self._last_step_t: float | None = None
        self._paused = True
        self._system: dict[str, Any] = {}
        self._system_t = 0.0
        self._prev_epoch_metrics: dict[str, float] = {}
        # Work outside the step loop (e.g. a long validation pass) and the trial's
        # place in an experiment plan; set through the monitor's ``note``/``set_plan``.
        self.activity: dict[str, Any] | None = None
        self.plan: tuple[int, int, int | None] | None = None

    # -- derived -----------------------------------------------------------------
    @property
    def fraction(self) -> float | None:
        if not self.total_steps:
            return None
        return min(1.0, max(0.0, self.step / self.total_steps))

    @property
    def epoch_fraction(self) -> float | None:
        if not self.steps_per_epoch:
            return None
        return min(1.0, max(0.0, self.epoch_step / self.steps_per_epoch))

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        end = self.finished if self.finished is not None else self.clock()
        return max(0.0, end - self.started)

    @property
    def eta(self) -> float | None:
        if self.status == "done" or not self.throughput or self.total_steps is None:
            return None
        remaining = max(0, self.total_steps - self.step)
        return remaining / self.throughput

    def direction(self, name: str) -> str:
        d = self._directions.get(name)
        if d is None:
            d = self._directions[name] = direction(name)
        return d

    def series(self, name: str) -> Series:
        store = self._series.get(name)
        return store.snapshot() if store is not None else Series(name)

    def metric_names(self, split: str | None = None) -> list[str]:
        return [k for k, s in self._series.items() if split is None or s.split == split]

    def latest(self, name: str) -> tuple[float, int, int] | None:
        return self._latest.get(name)

    def latest_value(self, name: str) -> float | None:
        item = self._latest.get(name)
        return None if item is None else item[0]

    def smoothed(self, name: str) -> float | None:
        """EMA of a step metric (``ema`` weight), else the latest value."""
        value = self._ema.get(name)
        return value if value is not None else self.latest_value(name)

    def best(self, name: str) -> tuple[float, int, int] | None:
        return self._best.get(name)

    def set_activity(self, label: str | None, done: int | None = None,
                     total: int | None = None) -> None:
        """Progress of work that is not a training step; ``label=None`` clears it."""
        with self.lock:
            if label is None:
                self.activity = None
            else:
                label = str(label)
                now = self.clock()
                if self.activity is None or self.activity["label"] != label:
                    self.activity = {"label": label, "t0": now}
                self.activity["done"] = int(done or 0)
                self.activity["total"] = int(total) if total is not None else None
                elapsed = now - self.activity["t0"]
                self.activity["rate"] = (self.activity["done"] / elapsed
                                         if elapsed > 0 and self.activity["done"] else None)
            self.version += 1

    def set_plan(self, index: int, total: int, done: int | None = None) -> None:
        """This trial is ``index`` (1-based) of ``total``; ``done`` others finished."""
        with self.lock:
            self.plan = (int(index), int(total), None if done is None else int(done))
            self.version += 1

    def system(self) -> dict[str, Any]:
        """Cached ``collect_system()`` (≤ 1 call / 2 s); call from render threads only."""
        now = time.monotonic()
        if now - self._system_t >= 2.0 or not self._system:
            self._system_t = now
            try:
                from topologies.training.monitoring.system import collect_system

                self._system = collect_system()
            except Exception:  # noqa: BLE001
                self._system = {}
        return dict(self._system)

    def add_log(self, text: str, level: str = "info") -> None:
        self.log.append((time.time(), level, str(text)))

    # -- updates -------------------------------------------------------------------
    def _record(self, name: str, value: float, split: str) -> None:
        store = self._series.get(name)
        if store is None:
            store = self._series[name] = SeriesStore(name, split)
        store.add(float(self.step), value)
        point = (value, self.step, self.epoch)
        self._latest[name] = point
        best = self._best.get(name)
        if best is None or (value > best[0] if self.direction(name) == "max"
                            else value < best[0]):
            self._best[name] = point

    def update(self, event: Any) -> bool:
        """Apply one event; True when the visible state changed."""
        kind = getattr(event, "kind", None)
        handler = _TRAINING_HANDLERS.get(kind)
        if handler is None:
            return False
        with self.lock:
            handler(self, event)
            self.version += 1
            self.last_event = kind
        return True

    def _on_start(self, event: Any) -> None:
        data = event.data
        self.run = _info_view(data.get("info"))
        total = event.progress.total if event.progress is not None else None
        total = total if total is not None else data.get("total_steps")
        self.total_steps = int(total) if _num(total) is not None and total else None
        info = self.run
        if info is not None:
            if getattr(info, "max_epochs", None) is not None:
                self.total_epochs = int(info.max_epochs)
            if getattr(info, "steps_per_epoch", None) is not None:
                self.steps_per_epoch = int(info.steps_per_epoch)
            if getattr(info, "global_step", None):
                self.step = int(info.global_step)
        self.status = "training"
        self.started = self.clock()
        self.finished = None
        self._paused = True
        self.add_log("training started")

    def _on_epoch_start(self, event: Any) -> None:
        data = event.data
        if data.get("epoch") is not None:
            self.epoch = int(data["epoch"])
        if data.get("total_epochs"):
            self.total_epochs = int(data["total_epochs"])
        self.epoch_step = 0
        self._paused = True
        self.status = "training"

    def _on_step(self, event: Any) -> None:
        data = event.data
        step = event.step
        if step is not None:
            self.step = int(step)
        if data.get("epoch") is not None:
            self.epoch = int(data["epoch"])
        if data.get("epoch_step") is not None:
            self.epoch_step = int(data["epoch_step"])
        if data.get("steps_per_epoch") is not None:
            self.steps_per_epoch = int(data["steps_per_epoch"])
        if event.progress is not None and event.progress.total and not self.total_steps:
            self.total_steps = int(event.progress.total)
        now = self.clock()
        if self._last_step_t is not None and not self._paused:
            dt = now - self._last_step_t
            if 0 < dt < 3600:
                rate = 1.0 / dt
                self.throughput = rate if self.throughput is None else \
                    0.8 * self.throughput + 0.2 * rate
        self._last_step_t = now
        self._paused = False
        w = self.ema_weight
        for name, value in event.metrics.items():
            f = _num(value)
            if f is None:
                continue
            self._record(str(name), f, "train")
            prev = self._ema.get(name)
            self._ema[name] = f if prev is None else w * prev + (1 - w) * f
        if self.status not in ("training", "validating"):
            self.status = "training"

    def _on_validation_start(self, event: Any) -> None:
        self.status = "validating"
        self._paused = True

    def _on_validation_end(self, event: Any) -> None:
        step = event.step
        if step is not None:
            self.step = int(step)
        for name, value in event.metrics.items():
            f = _num(value)
            if f is None:
                continue
            key = str(name) if str(name).startswith("val") else f"val_{name}"
            self._record(key, f, "val")
        self.status = "training"
        self._paused = True
        self.add_log("validation " + "  ".join(
            f"{k} {v:.4g}" for k, v in event.metrics.items() if _num(v) is not None))

    def _on_checkpoint(self, event: Any) -> None:
        data = event.data
        entry = {"path": str(data.get("path", "")), "epoch": data.get("epoch"),
                 "step": data.get("global_step"), "is_best": bool(data.get("is_best"))}
        self.checkpoints.append(entry)
        del self.checkpoints[:-100]
        self._paused = True
        self.add_log(f"checkpoint {entry['path']}" + (" (best)" if entry["is_best"] else ""))

    def _monitored_metric(self, val_metrics: Mapping[str, Any]) -> str | None:
        if self.monitored:
            return self.monitored
        for key in ("val_loss", "val"):
            if key in val_metrics:
                self.monitored = key
                return key
        for key in val_metrics:
            self.monitored = str(key)
            return self.monitored
        return None

    def _on_epoch_end(self, event: Any) -> None:
        data = event.data
        epoch = int(data.get("epoch", self.epoch) or 0)
        train = {str(k): f for k, v in dict(data.get("train_metrics") or event.metrics).items()
                 if (f := _num(v)) is not None}
        val = {str(k): f for k, v in dict(data.get("val_metrics") or {}).items()
               if (f := _num(v)) is not None}
        metrics = {**train, **val}
        best = _num(data.get("best_metric"))
        is_best = False
        if best is not None:
            is_best = self.best_metric is None or best != self.best_metric
            self.best_metric = best
        else:
            key = self._monitored_metric(val)
            if key is not None and key in metrics:
                prev = self.best_metric
                value = metrics[key]
                better = prev is None or (value > prev if self.direction(key) == "max"
                                          else value < prev)
                if better:
                    self.best_metric = value
                    is_best = True
        if is_best:
            self.best_epoch = epoch
        self._monitored_metric(val)
        row = EpochRow(epoch=epoch, metrics=metrics, duration=_num(data.get("duration")),
                       is_best=is_best, checkpoint=data.get("checkpoint_path"),
                       best_metric=self.best_metric, step=self.step)
        self.epochs.append(row)
        del self.epochs[:-1000]
        self._paused = True
        if self.steps_per_epoch:
            self.epoch_step = self.steps_per_epoch
        self.add_log(f"epoch {epoch + 1} done" + (" ★ best" if is_best else ""))

    def _on_train_end(self, event: Any) -> None:
        data = event.data
        step = data.get("global_step")
        if step is not None:
            self.step = int(step)
        self.duration = _num(data.get("duration"))
        best = _num(data.get("best_metric"))
        if best is not None:
            self.best_metric = best
        if data.get("best_epoch") is not None:
            self.best_epoch = int(data["best_epoch"])
        self.status = "done"
        self.finished = self.clock()
        self.add_log("training finished")

    def _on_error(self, event: Any) -> None:
        data = event.data
        self.status = "failed"
        self.finished = self.clock()
        self.error = f"{data.get('error_type', 'error')}: {data.get('message', '')}".strip(": ")
        self.add_log(self.error, "error")


_TRAINING_HANDLERS = {
    EventKind.TRAIN_START: TrainingTracker._on_start,
    EventKind.EPOCH_START: TrainingTracker._on_epoch_start,
    EventKind.STEP: TrainingTracker._on_step,
    EventKind.VALIDATION_START: TrainingTracker._on_validation_start,
    EventKind.VALIDATION_END: TrainingTracker._on_validation_end,
    EventKind.CHECKPOINT: TrainingTracker._on_checkpoint,
    EventKind.EPOCH_END: TrainingTracker._on_epoch_end,
    EventKind.TRAIN_END: TrainingTracker._on_train_end,
    EventKind.TRAIN_ERROR: TrainingTracker._on_error,
}


# ---------------------------------------------------------------- experiments


@dataclass
class RunningTrial:
    trial_id: str
    index: int | None = None
    labels: dict[str, Any] = field(default_factory=dict)
    seed: Any = None
    epoch: int | None = None
    max_epochs: int | None = None
    global_step: int | None = None
    total_steps: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    val: dict[str, float] = field(default_factory=dict)
    started: float = 0.0
    curve: list[tuple[float, float]] = field(default_factory=list)
    loss_curve: list[tuple[float, float]] = field(default_factory=list)

    @property
    def fraction(self) -> float | None:
        if self.total_steps and self.global_step is not None:
            return min(1.0, max(0.0, self.global_step / self.total_steps))
        if self.max_epochs and self.epoch is not None and self.max_epochs < 10**5:
            return min(1.0, max(0.0, self.epoch / self.max_epochs))
        return None


@dataclass
class TrialSummary:
    trial_id: str
    index: int | None
    labels: dict[str, Any]
    seed: Any
    status: str
    value: float | None
    wall_s: float | None = None
    error: str | None = None
    reason: str | None = None
    curve: list[tuple[float, float]] = field(default_factory=list)
    cached: bool = False
    is_best: bool = False
    loss_curve: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class ConfigRow:
    key: tuple
    labels: dict[str, Any]
    mean: float | None
    std: float | None
    n: int
    values: list[float]
    wall_s: float | None
    slot: int
    best: bool = False
    curve: list[tuple[float, float]] = field(default_factory=list)


def config_key(labels: Mapping[str, Any] | None) -> tuple:
    """Hashable config identity: sorted ``(axis, str(value))`` pairs without ``seed``."""
    return tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items() if k != "seed"))


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, None
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var)


def _goal_curve(result: Mapping[str, Any], metric: str | None) -> list[tuple[float, float]]:
    out = []
    for i, row in enumerate(result.get("history") or ()):
        if not isinstance(row, Mapping):
            continue
        v = _num(row.get(metric)) if metric else None
        if v is not None:
            out.append((float(row.get("epoch", i)), v))
    return out


class ExperimentTracker:
    """Experiment run state: counts, running trials, records, leaderboard, ETA, color slots."""

    def __init__(self, *, clock: Any = time.monotonic, log_size: int = 500) -> None:
        self.lock = threading.RLock()
        self.clock = clock
        self.version = 0
        self.name = "experiment"
        self.goal: tuple[str, str] = ("value", "min")
        self.axes: dict[str, tuple] = {}
        self.all_axes: dict[str, tuple] = {}
        self.total = 0
        self.done = 0
        self.cached = 0
        self.workers = 1
        self.seeds = 1
        self.strategy: str | None = None
        self.status = "idle"
        self.counts: Counter = Counter()
        self.running: dict[str, RunningTrial] = {}
        self.records: list[TrialSummary] = []
        self.best_so_far: list[tuple[int, float]] = []
        self.best_value: float | None = None
        self.best_labels: dict[str, Any] | None = None
        self.result: Any = None
        self.error: str | None = None
        self.log: deque[tuple[float, str, str]] = deque(maxlen=log_size)
        self.info: dict[str, Any] = {}
        self.started: float | None = None
        self.finished: float | None = None
        self.trials: dict[str, dict[str, Any]] = {}
        self.pending: set[str] = set()
        self._slots: dict[tuple, int] = {}
        self._durations: dict[tuple, list[float]] = {}
        self._overall: list[float] = []
        self._board_cache: tuple[int, int, list[ConfigRow]] | None = None
        self._index = 0
        self.charts: tuple[str, ...] = ("loss", "goal")

    # -- derived ------------------------------------------------------------------
    @property
    def metric(self) -> str:
        return self.goal[0]

    @property
    def mode(self) -> str:
        return self.goal[1]

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        end = self.finished if self.finished is not None else self.clock()
        return max(0.0, end - self.started)

    @property
    def fraction(self) -> float | None:
        return min(1.0, self.done / self.total) if self.total else None

    @property
    def failed(self) -> int:
        return sum(v for k, v in self.counts.items() if k not in ("completed", "pruned"))

    def slot(self, key: Any) -> int:
        """Stable color slot for a config (plan order / first appearance, never rank)."""
        if not isinstance(key, tuple):
            key = config_key(key)
        if key not in self._slots:
            self._slots[key] = len(self._slots)
        return self._slots[key]

    def _better(self, a: float, b: float | None) -> bool:
        return b is None or (a > b if self.mode == "max" else a < b)

    @property
    def eta(self) -> float | None:
        if self.status in ("done", "failed") or not self._overall:
            return None
        fallback = sum(self._overall) / len(self._overall)
        remaining = [tid for tid in self.pending if tid not in self.running]
        total = 0.0
        for tid in remaining:
            vals = self._durations.get(config_key(self.trials.get(tid, {}).get("labels")))
            total += (sum(vals) / len(vals)) if vals else fallback
        now = self.clock()
        for trial in self.running.values():
            key = config_key(trial.labels)
            vals = self._durations.get(key)
            expected = (sum(vals) / len(vals)) if vals else fallback
            total += max(0.0, expected - (now - trial.started))
        if not remaining and not self.running:
            unknown = max(0, self.total - self.done)
            total = unknown * fallback
        return max(0.0, total / max(1, self.workers))

    def leaderboard(self, top: int | None = 5) -> list[ConfigRow]:
        """Configs ranked by mean goal value (direction-aware); cached per change."""
        cache = self._board_cache
        if cache is not None and cache[0] == self.version and cache[1] == (top or -1):
            return cache[2]
        groups: dict[tuple, list[TrialSummary]] = {}
        for rec in self.records:
            if rec.status == "completed" and rec.value is not None:
                groups.setdefault(config_key(rec.labels), []).append(rec)
        rows = []
        for key, recs in groups.items():
            values = [r.value for r in recs if r.value is not None]
            mean, std = _mean_std(values)
            walls = [r.wall_s for r in recs if r.wall_s is not None]
            curve = _mean_curve([r.curve for r in recs])
            labels = {k: v for k, v in recs[0].labels.items() if k != "seed"}
            rows.append(ConfigRow(key, labels, mean, std, len(values), values,
                                  sum(walls) / len(walls) if walls else None, self.slot(key),
                                  curve=curve))
        sign = -1.0 if self.mode == "max" else 1.0
        rows.sort(key=lambda r: (sign * r.mean if r.mean is not None else math.inf))
        if rows:
            rows[0].best = True
        out = rows[:top] if top else rows
        self._board_cache = (self.version, top or -1, out)
        return out

    def add_log(self, text: str, level: str = "info") -> None:
        self.log.append((time.time(), level, str(text)))

    # -- updates ------------------------------------------------------------------
    def update(self, event: Any) -> bool:
        kind = getattr(event, "kind", None)
        handler = _EXPERIMENT_HANDLERS.get(kind)
        if handler is None:
            return False
        with self.lock:
            changed = handler(self, event)
            if changed is not False:
                self.version += 1
        return changed is not False

    def _add_record(self, rec: TrialSummary) -> None:
        self.records.append(rec)
        if rec.status in ("completed",) and rec.wall_s is not None:
            key = config_key(rec.labels)
            self._durations.setdefault(key, []).append(rec.wall_s)
            self._overall.append(rec.wall_s)
        self.slot(rec.labels)

    def _on_start(self, event: Any) -> None:
        data = event.data
        self.started = self.clock()
        self.finished = None
        self.status = "running"
        self.name = str(data.get("name") or data.get("experiment") or self.name)
        goal = data.get("goal")
        if isinstance(goal, (list, tuple)) and len(goal) == 2:
            self.goal = (str(goal[0]), str(goal[1]))
        elif isinstance(goal, Mapping):
            self.goal = (str(goal.get("metric", "value")), str(goal.get("mode", "min")))
        axes = data.get("axes") or {}
        if isinstance(axes, Mapping):
            self.all_axes = {str(k): tuple(v) if isinstance(v, (list, tuple)) else (v,)
                             for k, v in axes.items()}
        varying = data.get("varying")
        if isinstance(varying, Mapping):
            self.axes = {str(k): tuple(v) for k, v in varying.items()}
        else:
            self.axes = {k: v for k, v in self.all_axes.items() if len(v) > 1}
        total = data.get("total")
        if total is None and event.progress is not None:
            total = event.progress.total
        self.total = int(total or 0)
        self.workers = int(data.get("workers") or 1)
        self.seeds = int(data.get("seeds") or 1)
        self.strategy = data.get("strategy")
        for item in data.get("trials") or ():
            if isinstance(item, Mapping) and item.get("trial_id"):
                self.trials[str(item["trial_id"])] = dict(item)
                self.slot(item.get("labels"))
        self.pending = {str(t) for t in data.get("pending_ids") or ()}
        for item in data.get("cached") or ():
            if not isinstance(item, Mapping):
                continue
            status = str(item.get("status") or "?")
            rec = TrialSummary(str(item.get("trial_id")), item.get("index"),
                               dict(item.get("labels") or {}), item.get("seed"), status,
                               _num(item.get("goal_value")), _num(item.get("wall_s")),
                               cached=True)
            self._add_record(rec)
            if status == "completed":
                self.cached += 1
                if rec.value is not None and self._better(rec.value, self.best_value):
                    self.best_value = rec.value
                    self.best_labels = rec.labels
        if data.get("pending_ids") is not None:
            self.done = max(0, self.total - len(self.pending))
        for key, values in dict(data.get("durations") or {}).items():
            k = tuple(tuple(p) for p in key) if isinstance(key, (list, tuple)) else key
            self._durations.setdefault(k, []).extend(float(v) for v in values)
        self.add_log(f"{self.name}: {self.total} trials")

    def _on_trial_start(self, event: Any) -> None:
        data = event.data
        tid = str(event.trial_id or data.get("trial_id"))
        current = self.running.get(tid)
        if current is not None:  # the trial's own "start" event: merge its shape
            for key in ("max_epochs", "total_steps", "global_step"):
                if data.get(key) is not None:
                    setattr(current, key, int(data[key]))
            return
        known = self.trials.get(tid, {})
        labels = dict(data.get("labels") or data.get("config") or known.get("labels") or {})
        seed = data.get("seed", known.get("seed", labels.get("seed")))
        index = data.get("index") or known.get("index")
        if index is None:
            self._index += 1
            index = self._index
        self.running[tid] = RunningTrial(
            tid, index=int(index), labels=labels, seed=seed,
            max_epochs=data.get("max_epochs"), total_steps=data.get("total_steps"),
            started=self.clock())
        self.slot(labels)
        bits = [f"{k}={v}" for k, v in labels.items() if k != "seed"]
        self.add_log("running " + (" ".join(bits) if bits else tid[:8]))

    def _running(self, event: Any) -> RunningTrial | None:
        tid = event.trial_id
        if tid is None:
            return None
        trial = self.running.get(str(tid))
        if trial is None:
            labels = dict(event.data.get("config") or {})
            trial = self.running[str(tid)] = RunningTrial(str(tid), labels=labels,
                                                         started=self.clock())
        return trial

    def _on_progress(self, event: Any) -> bool:
        trial = self._running(event)
        if trial is None:
            return False
        data = event.data
        if data.get("global_step") is not None:
            trial.global_step = int(data["global_step"])
        if data.get("epoch") is not None and event.kind == EventKind.STEP:
            trial.epoch = int(data["epoch"]) + 1  # current epoch, 1-based
        for key in ("max_epochs", "total_steps"):
            if data.get(key) is not None:
                setattr(trial, key, int(data[key]))
        trial.metrics = {str(k): f for k, v in event.metrics.items()
                         if (f := _num(v)) is not None}
        loss = trial.metrics.get("loss")
        if loss is not None and trial.global_step is not None:
            step = float(trial.global_step)
            curve = trial.loss_curve
            if curve and curve[-1][0] == step:
                curve[-1] = (step, loss)
            else:
                curve.append((step, loss))
            if len(curve) > 4096:
                trial.loss_curve = curve[::2]
        return True

    def _on_epoch(self, event: Any) -> bool:
        trial = self._running(event)
        if trial is None:
            return False
        data = event.data
        trial.epoch = int(data.get("epoch") or 0) + 1
        trial.val = {str(k): f for k, v in event.metrics.items() if (f := _num(v)) is not None}
        value = trial.val.get(self.metric)
        if value is not None:
            trial.curve.append((float(trial.epoch), value))
        return True

    def _on_trial_end(self, event: Any) -> None:
        data = event.data
        tid = str(event.trial_id)
        running = self.running.pop(tid, None)
        result = data.get("result") or {}
        status = str(data.get("status") or result.get("status") or "?")
        goal = result.get("goal") or {}
        value = _num(goal.get("value"))
        if value is None:
            value = _num(event.metrics.get(self.metric))
        cost = result.get("cost") or {}
        labels = dict(result.get("labels") or (running.labels if running else None)
                      or data.get("config") or {})
        error = None
        if status not in ("completed", "pruned", "timeout"):
            err = result.get("error") or {}
            error = f"{err.get('type', status)}: {err.get('message', '')}".strip(": ") \
                if err else str(result.get("status_reason") or status)
        rec = TrialSummary(tid, running.index if running else None, labels,
                           labels.get("seed", running.seed if running else None), status,
                           value, _num(cost.get("wall_s")), error,
                           result.get("status_reason"), _goal_curve(result, self.metric),
                           loss_curve=list(running.loss_curve) if running else [])
        if status == "completed" and value is not None and self._better(value, self.best_value):
            self.best_value = value
            self.best_labels = labels
            rec.is_best = True
        self._add_record(rec)
        self.counts[status] += 1
        self.done += 1
        self.pending.discard(tid)
        if status == "completed" and value is not None:
            self.best_so_far.append((len(self.best_so_far) + 1, self.best_value))
        self.add_log(f"{status} {tid[:8]}" + (f" {error}" if error else ""),
                     "error" if error else "info")

    def _on_interrupted(self, event: Any) -> None:
        self.running.pop(str(event.trial_id), None)
        self.add_log(f"interrupted {str(event.trial_id)[:8]}", "warn")

    def _on_best(self, event: Any) -> bool:
        return False

    def _on_info(self, event: Any) -> None:
        self.info.update(dict(event.data))

    def _on_end(self, event: Any) -> None:
        self.result = event.data.get("result")
        self.status = "done"
        self.finished = self.clock()
        self.running.clear()

    def _on_error(self, event: Any) -> None:
        data = event.data
        self.status = "failed"
        self.finished = self.clock()
        self.error = f"{data.get('error', 'error')}: {data.get('message', '')}".strip(": ")
        self.add_log(self.error, "error")


def _mean_curve(curves: Iterable[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    by_x: dict[float, list[float]] = {}
    for curve in curves:
        for x, y in curve:
            by_x.setdefault(x, []).append(y)
    return [(x, sum(v) / len(v)) for x, v in sorted(by_x.items())]


_EXPERIMENT_HANDLERS = {
    EventKind.EXPERIMENT_START: ExperimentTracker._on_start,
    EventKind.TRIAL_START: ExperimentTracker._on_trial_start,
    EventKind.STEP: ExperimentTracker._on_progress,
    EventKind.EPOCH_END: ExperimentTracker._on_epoch,
    EventKind.TRIAL_END: ExperimentTracker._on_trial_end,
    EventKind.TRIAL_FAILED: ExperimentTracker._on_trial_end,
    EventKind.TRIAL_INTERRUPTED: ExperimentTracker._on_interrupted,
    EventKind.BEST: ExperimentTracker._on_best,
    "experiment.info": ExperimentTracker._on_info,
    EventKind.EXPERIMENT_END: ExperimentTracker._on_end,
    EventKind.EXPERIMENT_ERROR: ExperimentTracker._on_error,
}


def make_tracker(source: str | None) -> TrainingTracker | ExperimentTracker:
    """Tracker matching a session source (``"experiment"`` → ExperimentTracker)."""
    return ExperimentTracker() if source == "experiment" else TrainingTracker()


__all__ = [
    "ConfigRow",
    "EpochRow",
    "ExperimentTracker",
    "RunningTrial",
    "Series",
    "SeriesStore",
    "TrainingTracker",
    "TrialSummary",
    "config_key",
    "make_tracker",
]
