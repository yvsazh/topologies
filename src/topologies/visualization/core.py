"""Dependency-light event contracts, visualization composition and dispatch."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from types import MappingProxyType, SimpleNamespace
from typing import Any


class EventKind:
    """Event kind constants (plain strings, comparable to ``event.kind``)."""

    TRAIN_START = "training.started"
    EPOCH_START = "training.epoch_started"
    STEP = "training.step_completed"
    VALIDATION_START = "training.validation_started"
    VALIDATION_END = "training.validation_completed"
    CHECKPOINT = "training.checkpoint"
    EPOCH_END = "training.epoch_completed"
    TRAIN_END = "training.completed"
    TRAIN_ERROR = "training.error"
    EXPERIMENT_START = "experiment.started"
    TRIAL_START = "experiment.trial_started"
    TRIAL_END = "experiment.trial_completed"
    TRIAL_FAILED = "experiment.trial_failed"
    TRIAL_INTERRUPTED = "experiment.trial_interrupted"
    LEADERBOARD = "experiment.leaderboard_changed"
    BEST = "experiment.best_configuration_changed"
    EXPERIMENT_END = "experiment.completed"
    EXPERIMENT_ERROR = "experiment.error"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return tuple(v for k, v in vars(cls).items() if k.isupper())


# Guaranteed payload keys per event kind (documentation is generated from this dict).
EVENT_SCHEMA: Mapping[str, Mapping[str, str]] = MappingProxyType({
    EventKind.TRAIN_START: {"progress": "total = total steps (when known)",
                            "data": "info (RunInfo fields), total_steps"},
    EventKind.EPOCH_START: {"data": "epoch (0-based), total_epochs"},
    EventKind.STEP: {"metrics": "per-step training metrics (loss, lr, ...)",
                     "progress": "current = global step, total = total steps",
                     "data": "global_step, epoch, epoch_step, steps_per_epoch, elapsed, "
                             "throughput_steps_per_second"},
    EventKind.VALIDATION_START: {"data": "epoch"},
    EventKind.VALIDATION_END: {"metrics": "validation metrics (val_loss, ...)",
                               "progress": "current = global step of the measurement",
                               "data": "epoch, global_step, duration"},
    EventKind.CHECKPOINT: {"progress": "current = global step",
                           "data": "path, epoch, global_step, is_best"},
    EventKind.EPOCH_END: {"metrics": "epoch-mean training metrics",
                          "data": "epoch, train_metrics, val_metrics, best_metric, duration, "
                                  "checkpoint_path"},
    EventKind.TRAIN_END: {"progress": "current = final global step",
                          "data": "total_epochs, global_step, duration, best_metric, "
                                  "best_epoch, history"},
    EventKind.TRAIN_ERROR: {"data": "error_type, message"},
    EventKind.EXPERIMENT_START: {"progress": "current = 0, total = planned trials",
                                 "data": "name, goal, axes, varying, total, pending_ids, "
                                         "cached, trials, workers, durations, strategy"},
    EventKind.TRIAL_START: {"progress": "current = start order, total = planned trials",
                            "data": "index, labels, config, seed, max_epochs, total_steps"},
    EventKind.TRIAL_END: {"metrics": "final metrics + goal value",
                          "data": "status, config, result, goal_metric, mode"},
    EventKind.TRIAL_FAILED: {"metrics": "final metrics (when any)",
                             "data": "status, config, result, goal_metric, mode"},
    EventKind.TRIAL_INTERRUPTED: {"data": "status, config"},
    EventKind.LEADERBOARD: {"data": "status"},
    EventKind.BEST: {"data": "config, metric, value, mode"},
    EventKind.EXPERIMENT_END: {"data": "result (ResultSet)"},
    EventKind.EXPERIMENT_ERROR: {"data": "error, message"},
})


def event_schema_markdown() -> str:
    """Markdown table of :data:`EVENT_SCHEMA` (used by ``docs/visualization.md``)."""
    rows = ["| Kind | `EventKind` | `metrics` | `progress` | `data` |", "|---|---|---|---|---|"]
    names = {v: k for k, v in vars(EventKind).items() if k.isupper()}
    for kind, keys in EVENT_SCHEMA.items():
        rows.append(f"| `{kind}` | `{names[kind]}` | {keys.get('metrics', '—')} | "
                    f"{keys.get('progress', '—')} | {keys.get('data', '—')} |")
    return "\n".join(rows)


@dataclass(frozen=True)
class Progress:
    current: float | None = None
    total: float | None = None


@dataclass(frozen=True)
class MetricSeries:
    name: str
    values: tuple[float, ...]
    steps: tuple[float, ...] = ()


@dataclass(frozen=True)
class PlotSpec:
    kind: str
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualizationResult:
    kind: str
    data: Any
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VisualizationEvent:
    source: str
    kind: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: str | None = None
    trial_id: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    progress: Progress | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "metrics", _freeze(_snapshot(dict(self.metrics))))
        object.__setattr__(self, "data", _freeze(_snapshot(dict(self.data))))

    # -- read-only convenience accessors (computed, no stored state) --------------
    @property
    def step(self) -> Any:
        if self.progress is not None and self.progress.current is not None \
                and self.kind != EventKind.TRIAL_START:
            return self.progress.current
        return self.data.get("global_step", self.data.get("step"))

    @property
    def epoch(self) -> Any:
        return self.data.get("epoch")

    @property
    def total_steps(self) -> Any:
        if self.progress is not None and self.progress.total is not None \
                and self.is_training:
            return self.progress.total
        return self.data.get("total_steps")

    @property
    def fraction(self) -> float | None:
        step, total = self.step, self.total_steps
        try:
            return min(1.0, max(0.0, float(step) / float(total))) if total else None
        except (TypeError, ValueError):
            return None

    @property
    def status(self) -> str | None:
        status = self.data.get("status")
        if status is not None:
            return str(status)
        return {EventKind.TRIAL_END: "completed", EventKind.TRIAL_FAILED: "failed",
                EventKind.TRIAL_INTERRUPTED: "interrupted", EventKind.TRAIN_END: "completed",
                EventKind.EXPERIMENT_END: "completed"}.get(self.kind,
                                                           "error" if self.kind.endswith(
                                                               ".error") else None)

    @property
    def is_training(self) -> bool:
        return self.kind.startswith("training.")

    @property
    def is_experiment(self) -> bool:
        return self.kind.startswith("experiment.")

    @property
    def config(self) -> Mapping[str, Any]:
        value = self.data.get("labels") or self.data.get("config") or {}
        return value if isinstance(value, Mapping) else {}

    @property
    def elapsed(self) -> Any:
        return self.data.get("elapsed")

    def metric(self, name: str, default: Any = None) -> Any:
        """Metric value by name (``metrics`` first, then val/train metric mappings)."""
        if name in self.metrics:
            return self.metrics[name]
        for key in ("val_metrics", "train_metrics"):
            values = self.data.get(key)
            if isinstance(values, Mapping) and name in values:
                return values[name]
        return default


@dataclass(frozen=True)
class SessionInfo(Mapping):
    """Session context passed to ``Visualizer.prepare``; also a read-only mapping."""

    source: str = "result"
    run_id: str | None = None
    title: str | None = None
    run_info: Mapping[str, Any] | None = None
    goal: tuple[str, str] | None = None
    axes: Mapping[str, tuple] | None = None
    total_trials: int | None = None
    request_stop: Callable[[str | None], None] | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def _view(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"}
        out.update(self.options)
        out.update(self.extra)
        return out

    def __getitem__(self, key: str) -> Any:
        return self._view()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._view())

    def __len__(self) -> int:
        return len(self._view())

    __hash__ = object.__hash__


def session_info(source: str | None, context: Any = None) -> SessionInfo:
    """Coerce ``prepare(source, context)`` arguments into a :class:`SessionInfo`."""
    if isinstance(context, SessionInfo):
        return context
    ctx = dict(context or {}) if isinstance(context, Mapping) else {}
    known = {f.name for f in fields(SessionInfo)}
    kwargs = {k: ctx.pop(k) for k in list(ctx) if k in known and k not in ("extra",)}
    kwargs.setdefault("source", source or "result")
    if "title" not in kwargs:
        kwargs["title"] = ctx.get("experiment") or ctx.get("name")
    return SessionInfo(**kwargs, extra=ctx)


def _snapshot(value):
    """Copy event payloads recursively without retaining tensor/model objects."""
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _snapshot(getattr(value, item.name)) for item in fields(value)}
    if type(value).__module__.startswith("torch") and hasattr(value, "detach"):
        try:
            item = value.detach().cpu()
            return item.item() if item.numel() == 1 else item.tolist()
        except Exception:  # noqa: BLE001
            return repr(value)
    try:
        return deepcopy(value)
    except Exception:  # noqa: BLE001
        return repr(value)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _legacy_value(value):
    """Convert frozen snapshots back to mutable legacy mapping/list shapes."""
    if isinstance(value, Mapping):
        return {key: _legacy_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_legacy_value(item) for item in value)
    return value


_HOOKS = {
    EventKind.TRAIN_START: "on_train_start",
    EventKind.EPOCH_START: "on_epoch_start",
    EventKind.STEP: "on_step",
    EventKind.VALIDATION_START: "on_validation_start",
    EventKind.VALIDATION_END: "on_validation_end",
    EventKind.CHECKPOINT: "on_checkpoint",
    EventKind.EPOCH_END: "on_epoch_end",
    EventKind.TRAIN_END: "on_train_end",
    EventKind.EXPERIMENT_START: "on_experiment_start",
    EventKind.EXPERIMENT_END: "on_experiment_end",
    EventKind.TRIAL_START: "on_trial_start",
    EventKind.TRIAL_END: "on_trial_end",
    EventKind.TRIAL_FAILED: "on_trial_end",
    EventKind.TRIAL_INTERRUPTED: "on_trial_end",
    EventKind.BEST: "on_best",
    EventKind.LEADERBOARD: "on_leaderboard",
}


class Visualizer:
    """Base contract for observers of immutable visualization events/results.

    The default :meth:`handle` calls :meth:`on_event` for every event and then the
    matching hook (``on_train_start``, ``on_step``, ``on_epoch_end``, ``on_trial_end``,
    ``on_error`` ...). Subclasses that override ``handle`` keep working unchanged.
    """

    owns_terminal: bool = False

    def prepare(self, source: str | None = None, context: Mapping[str, Any] | None = None) -> None:
        """Prepare a visualization session before events are dispatched."""
        return

    def handle(self, event: VisualizationEvent) -> None:
        """Observe one immutable event snapshot (dispatches to hook methods)."""
        dispatch_hooks(self, event)

    def on_event(self, event: VisualizationEvent) -> None:
        """Called for every event before the specific hook."""
        return

    def render(self, result: VisualizationResult) -> Any:
        """Render or consume a structured final result."""
        return

    def close(self) -> None:
        """Release session resources and signal lifecycle completion."""
        return

    def __enter__(self) -> Visualizer:  # noqa: PYI034
        self.prepare()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def dispatch_hooks(visualizer: Any, event: VisualizationEvent) -> None:
    """Call ``on_event`` then the hook named after ``event.kind`` (if defined)."""
    on_event = getattr(visualizer, "on_event", None)
    if callable(on_event):
        on_event(event)
    kind = getattr(event, "kind", None)
    if not isinstance(kind, str):
        return
    name = _HOOKS.get(kind) or ("on_error" if kind.endswith(".error") else None)
    hook = getattr(visualizer, name, None) if name else None
    if callable(hook):
        hook(event)


ENTRY_POINT_GROUP = "topologies.visualizers"


class VisualizerRegistry:
    """Named factories for constructing visualization sinks.

    With ``discover=True`` an unknown name triggers a one-time load of the
    ``topologies.visualizers`` entry-point group before failing.
    """

    def __init__(self, *, discover: bool = False):
        self._factories: dict[str, Callable] = {}
        self.discover = discover
        self._discovered = False

    def register(self, name, factory):
        """Register a callable visualizer factory under ``name``."""
        if not callable(factory):
            raise TypeError("visualizer factory must be callable")
        self._factories[str(name)] = factory

    def _load_entry_points(self) -> None:
        if self._discovered:
            return
        self._discovered = True
        try:
            from importlib.metadata import entry_points

            found = entry_points(group=ENTRY_POINT_GROUP)
        except Exception:  # noqa: BLE001
            return
        for ep in found:
            if ep.name in self._factories:
                continue
            try:
                self.register(ep.name, ep.load())
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Visualizer plugin {ep.name!r} failed to load: {exc}",
                              RuntimeWarning, stacklevel=3)

    def create(self, name, **options):
        """Instantiate a registered visualizer with backend options."""
        factory = self._factories.get(name)
        if factory is None and self.discover:
            self._load_entry_points()
            factory = self._factories.get(name)
        if factory is None:
            raise ValueError(f"Unknown visualizer {name!r}; available: "
                             f"{', '.join(self.names()) or 'none'}")
        return factory(**options)

    def names(self):
        """Return registered factory names in registration order."""
        return tuple(self._factories)

    def __contains__(self, name) -> bool:
        return name in self._factories


def register_visualizer(name: str, registry: VisualizerRegistry | None = None) -> Callable:
    """Decorator registering a Visualizer class or factory under ``name``."""
    def decorator(factory):
        (registry or DEFAULT_REGISTRY).register(name, factory)
        return factory
    return decorator


class VisualizerGroup(Visualizer):
    def __init__(self, sinks=()):
        self.sinks = list(sinks)
        self._disabled = set()
        self._closed = False

    def _dispatch(self, method, *args):
        for sink in tuple(self.sinks):
            if id(sink) in self._disabled:
                continue
            try:
                getattr(sink, method)(*args)
            except Exception as exc:  # noqa: BLE001
                self._disabled.add(id(sink))
                warnings.warn(
                    f"Visualizer {sink!r} disabled after {method} failure: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _resolve_terminal_owners(self) -> None:
        owners = [sink for sink in self.sinks if getattr(sink, "owns_terminal", False)]
        if len(owners) <= 1:
            return
        for sink in owners[1:]:
            release = getattr(sink, "release_terminal", None)
            if callable(release):
                release()
        warnings.warn(
            "Two live terminal displays cannot share a terminal; keeping "
            f"{type(owners[0]).__name__} and silencing "
            f"{', '.join(type(s).__name__ for s in owners[1:])}.",
            RuntimeWarning,
            stacklevel=3,
        )

    def prepare(self, source=None, context=None):
        self._closed = False
        self._disabled.clear()
        self._resolve_terminal_owners()
        prepared = []
        try:
            for sink in tuple(self.sinks):
                sink.prepare(source, context)
                prepared.append(sink)
        except BaseException:
            for sink in tuple(self.sinks):
                try:
                    sink.close()
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(
                        f"Visualizer cleanup failed during prepare: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            self._closed = True
            raise

    def handle(self, event):
        self._dispatch("handle", event)

    def render(self, result):
        values = []
        for sink in tuple(self.sinks):
            if id(sink) in self._disabled:
                continue
            values.append(sink.render(result))
        return values[0] if len(values) == 1 else tuple(values)

    def render_safe(self, result):
        values = []
        for sink in tuple(self.sinks):
            if id(sink) in self._disabled:
                continue
            try:
                render = getattr(sink, "render_session", sink.render)
                values.append(render(result))
            except Exception as exc:  # noqa: BLE001
                self._disabled.add(id(sink))
                warnings.warn(
                    f"Visualizer {sink!r} disabled after render failure: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return values[0] if len(values) == 1 else tuple(values)

    def handle_safe(self, event):
        self._dispatch("handle", event)

    def close(self):
        if self._closed:
            return
        self._closed = True
        # Cleanup every sink, including ones disabled by a prior failure.
        for sink in self.sinks:
            try:
                sink.close()
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Visualizer cleanup failed: {exc}", RuntimeWarning, stacklevel=2)


class _Quiet(Visualizer):
    pass


class LegacyMonitorVisualizer(Visualizer):
    """Adapt a configured Trainer monitor into the shared event channel."""

    def __init__(self, monitor, factory=None):
        self.monitor = monitor
        self.factory = factory

    def prepare(self, source=None, context=None):
        if self.monitor is None and self.factory is not None:
            self.monitor = self.factory()

    def handle(self, event):
        mapping = {
            "training.started": "on_train_begin",
            "training.epoch_started": "on_epoch_begin",
            "training.step_completed": "on_step",
            "training.validation_started": "on_validation_begin",
            "training.validation_completed": "on_validation_end",
            "training.checkpoint": "on_checkpoint",
            "training.epoch_completed": "on_epoch_end",
            "training.completed": "on_train_end",
        }
        method = mapping.get(event.kind)
        if method is None:
            return
        values = {**dict(event.data), "metrics": dict(event.metrics)}
        if event.kind == "training.epoch_completed":
            values.setdefault("train_metrics", dict(event.metrics))
        callback = getattr(self.monitor, method, None)
        if callback is not None:
            values = {key: _legacy_value(value) for key, value in values.items()}
            if isinstance(values.get("info"), dict):
                values["info"] = SimpleNamespace(**values["info"])
            callback(SimpleNamespace(**values))

    def close(self):
        if self.monitor is not None:
            self.monitor.close()
            self.monitor = None


class _LegacyTerminal(Visualizer):
    """Producer-aware terminal selector.

    Trainer sessions get the training tier (``standard``/``plain``/``dashboard``);
    experiment sessions get the experiment view for ``Experiment(monitor=...)``
    (``standard``), ``ExperimentDashboard`` (``dashboard``) or lines (``plain``).
    """

    owns_terminal = True

    def __init__(self, mode="standard", **options):
        self.mode = mode
        self.options = options
        self.monitor = None
        self._view = None
        self._released = False

    def _experiment_view(self, context):
        from topologies.visualization.experiment_views import make_experiment_view

        monitor = (context or {}).get("monitor", "live") if isinstance(context, Mapping) \
            else "live"
        mode = {"standard": monitor, "dashboard": "dashboard", "plain": "lines"}[self.mode]
        options = {k: v for k, v in self.options.items()
                   if k in ("stream", "theme", "refresh_rate", "capture_output", "threaded",
                            "caps", "tty", "color", "fullscreen", "panels", "view",
                            "interactive", "status_line", "summary")}
        return make_experiment_view(mode, **options)

    def prepare(self, source=None, context=None):
        self.monitor = None
        if source == "experiment":
            self._view = self._experiment_view(context)
        else:
            if self.mode == "dashboard":
                from topologies.training.monitoring.dashboard.app import DashboardMonitor

                self.monitor = DashboardMonitor(**self.options)
            else:
                from topologies.training.monitoring.standard import StandardMonitor

                self.monitor = StandardMonitor(self.options, plain=self.mode == "plain")
            self.monitor._prepared = True
            self._view = self.monitor.visualizer
        if self._view is None:
            return
        if self._released:
            self._view.release_terminal()
        self._view.prepare(source or "training", context)

    def release_terminal(self):
        self._released = True
        if self._view is not None:
            self._view.release_terminal()

    def handle(self, event):
        if getattr(event, "source", None) == "introspection":
            if self.mode in ("standard", "plain"):
                _print_introspection_event(event)
            return
        if self._view is None:
            if self.monitor is not None or getattr(event, "source", None) == "experiment":
                return
            self.prepare()
        self._view.handle(event)

    def render(self, result):
        if getattr(result, "kind", None) == "views" and self.mode in ("standard", "plain"):
            from .views import describe_views

            print(describe_views(list(result.data.get("views", ()))))
            return None
        if self._view is not None:
            return self._view.render(result)
        return None

    def close(self):
        if self._view is not None:
            self._view.close()


def _print_introspection_event(event):
    """One terminal line per introspection progress event (scalars only)."""
    progress = event.progress
    where = ""
    if progress is not None and progress.current is not None:
        where = f" {progress.current}" + (f"/{progress.total}" if progress.total else "")
    metrics = " ".join(f"{k}={v}" for k, v in dict(event.metrics).items()
                       if isinstance(v, (int, float, str)))
    label = event.kind.removeprefix("introspection.").replace("_", " ")
    print(f"[introspection] {label}{where} {metrics}".rstrip())


class StandardVisualizer(_LegacyTerminal):
    """Public standard selector: the producer's default terminal tier."""

    def __init__(self, **options):
        super().__init__("standard", **options)


class DashboardVisualizer(_LegacyTerminal):
    """Public dashboard selector: ``TrainingDashboard`` for Trainer sessions."""

    def __init__(self, **options):
        super().__init__("dashboard", **options)


class PlainVisualizer(_LegacyTerminal):
    """Public plain selector: log-friendly lines (``PlainTrainingVisualizer`` for Trainer)."""

    owns_terminal = False

    def __init__(self, **options):
        super().__init__("plain", **options)


DEFAULT_REGISTRY = VisualizerRegistry(discover=True)
DEFAULT_REGISTRY.register("quiet", lambda **kw: _Quiet())
DEFAULT_REGISTRY.register("standard", StandardVisualizer)
DEFAULT_REGISTRY.register("dashboard", DashboardVisualizer)
DEFAULT_REGISTRY.register("plain", PlainVisualizer)


def _matplotlib_factory(**options):
    from .matplotlib import MatplotlibVisualizer

    return MatplotlibVisualizer(**options)


DEFAULT_REGISTRY.register("matplotlib", _matplotlib_factory)


class ExperimentRunStateAdapter(Visualizer):
    """Delegate normalized events to the existing experiment RunState renderer."""

    def __init__(self, run_state):
        self.run_state = run_state
        self._start = getattr(run_state, "trial_start", None)
        self._done = getattr(run_state, "trial_done", None)
        self._interrupted = getattr(run_state, "trial_interrupted", None)
        self.mode = getattr(run_state, "mode", "none")

    def _call(self, callback, *args):
        old = getattr(self.run_state, "mode", None)
        if old is not None:
            self.run_state.mode = self.mode
        try:
            callback(*args)
        finally:
            if old is not None:
                self.run_state.mode = old

    def handle(self, event):
        if event.source != "experiment":
            return
        kinds = {
            "training.step_completed": "progress",
            "training.epoch_completed": "epoch",
        }
        kind = kinds.get(event.kind)
        data = dict(event.data)
        if event.kind == "experiment.completed" and data.get("result") is not None:
            if hasattr(self.run_state, "finish"):
                self._call(self.run_state.finish, data["result"])
            return
        if event.kind == "experiment.trial_started":
            trial = self._trial(data, event.trial_id)
            if self._start and trial is not None:
                self._call(self._start, trial)
            return
        if event.kind in ("experiment.trial_completed", "experiment.trial_failed"):
            trial = self._trial(data, event.trial_id)
            if self._done and trial is not None:
                self._call(self._done, trial, data["result"])
            return
        if event.kind == "experiment.trial_interrupted":
            trial = self._trial(data, event.trial_id)
            if self._interrupted and trial is not None:
                self._call(self._interrupted, trial)
            return
        if kind:
            payload = {
                **data,
                "type": kind,
                "trial_id": event.trial_id,
                "metrics": dict(event.metrics),
            }
            self._call(getattr(self.run_state, "_visualizer_original_handle", self.run_state.handle), payload)

    def _trial(self, data, trial_id):
        return next((trial for trial in self.run_state.trials if trial.trial_id == trial_id), None)

    def close(self):
        self._call(self.run_state.close)


def _selection_items(selection) -> list:
    if isinstance(selection, tuple) and len(selection) == 2 and isinstance(selection[0], str) \
            and isinstance(selection[1], Mapping):
        return [selection]
    if isinstance(selection, Mapping):
        return [selection]
    return list(selection) if isinstance(selection, (list, tuple)) else [selection]


def create_visualizer(selection="standard", *, registry=None, **options):
    """Resolve instance/name/list/False; an empty list explicitly silences output.

    List items may be names, Visualizer instances, ``(name, options)`` tuples or
    ``{name: options}`` dicts. Top-level ``**options`` apply only when exactly one
    named sink is selected.
    """
    if selection is False or selection is None:
        return VisualizerGroup()
    if isinstance(selection, Visualizer):
        selection = [selection]
    reg = registry or DEFAULT_REGISTRY
    items: list = []
    for item in _selection_items(selection):
        if isinstance(item, Mapping):
            items.extend((str(k), dict(v or {})) for k, v in item.items())
        else:
            items.append(item)
    named = [item for item in items if isinstance(item, (str, tuple))]
    if options and len(named) != 1:
        raise TypeError(
            "Top-level visualizer options apply to exactly one named sink; with several "
            "sinks pass per-sink options, e.g. visualizers=['standard', "
            "('matplotlib', {'save': 'plots'})] or {'matplotlib': {...}}."
        )
    sinks = []
    created = []
    try:
        for item in items:
            if isinstance(item, VisualizerGroup):
                sinks.extend(item.sinks)
            elif isinstance(item, Visualizer):
                sinks.append(item)
            elif isinstance(item, str):
                sink = reg.create(item, **options)
                created.append(sink)
                sinks.append(sink)
            elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) \
                    and isinstance(item[1], Mapping):
                sink = reg.create(item[0], **{**options, **dict(item[1])})
                created.append(sink)
                sinks.append(sink)
            else:
                raise TypeError("visualizers must be names, Visualizer instances, "
                                "(name, options) tuples, {name: options} dicts, a list, or False")
    except BaseException:
        for sink in reversed(created):
            try:
                sink.close()
            except Exception:  # noqa: BLE001, S110 - best-effort cleanup while re-raising
                pass
        raise
    return VisualizerGroup(list({id(sink): sink for sink in sinks}.values()))


def normalize_event(event, *, source=None, run_id=None, trial_id=None, step_baseline=0, elapsed_baseline=0.0):
    """Normalize Trainer dataclasses or experiment event mappings without mutation."""
    if isinstance(event, VisualizationEvent):
        return event
    if isinstance(event, Mapping):
        raw = dict(event)
        src = source or "experiment"
        kind = raw.pop("kind", raw.pop("type", "event"))
        kinds = {
            "start": "experiment.trial_started",
            "progress": "training.step_completed",
            "epoch": "training.epoch_completed",
        }
        prefix = "training" if src == "training" else "experiment"
        semantic = kinds.get(kind, kind if "." in str(kind) else f"{prefix}.{kind}")
        metrics = raw.get("metrics") or {}
        tid = trial_id or raw.get("trial_id")
    else:
        raw = dict(vars(event))
        src = source or "training"
        names = {
            "TrainBeginEvent": "training.started",
            "EpochBeginEvent": "training.epoch_started",
            "StepEvent": "training.step_completed",
            "ValidationBeginEvent": "training.validation_started",
            "ValidationEvent": "training.validation_completed",
            "EpochEndEvent": "training.epoch_completed",
            "TrainEndEvent": "training.completed",
            "CheckpointEvent": "training.checkpoint",
        }
        semantic = names.get(
            type(event).__name__, f"training.{type(event).__name__.removesuffix('Event').lower()}"
        )
        metrics = raw.get("metrics") or raw.get("train_metrics") or raw.get("val_metrics") or {}
        tid = trial_id
    step = raw.get("global_step", raw.get("step", raw.get("current")))
    total = raw.get("total_steps", raw.get("total"))
    progress = Progress(step, total) if step is not None or total is not None else None
    # StepEvent.elapsed is cumulative; its quotient is a measured average rate,
    # not an inferred per-step performance estimate.
    elapsed = raw.get("elapsed")
    if src == "training" and step is not None:
        raw.setdefault("global_step", step)
    if src == "training" and step is not None and isinstance(elapsed, (int, float)) and elapsed > elapsed_baseline:
        raw.setdefault("throughput_steps_per_second", (float(step) - float(step_baseline)) / (float(elapsed) - float(elapsed_baseline)))
    return VisualizationEvent(
        src, semantic, run_id=run_id, trial_id=tid, metrics=metrics, progress=progress, data=raw
    )
