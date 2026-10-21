"""Standard terminal monitor (compatibility facade over the shared training views)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from topologies.training.monitoring.base import BaseMonitor


@dataclass
class StandardMonitorOptions:
    columns: list[str] | None = None
    show_eta: bool = True
    show_mem: bool = True
    show_lr: bool = True
    precision: int = 3
    log_every_steps: int = 10
    use_ema: bool = True


def _fmt_val(v: Any, precision: int = 3) -> str:
    """Compatibility wrapper over :func:`topologies.visualization.format.fmt_number`."""
    from topologies.visualization.format import fmt_number

    return fmt_number(v, sig=max(1, int(precision or 3)))


def _fmt_time(seconds: float) -> str:
    """Compatibility wrapper over :func:`topologies.visualization.format.fmt_clock`."""
    from topologies.visualization.format import fmt_clock

    return fmt_clock(seconds)


def _is_tty(stream: Any = None) -> bool:
    """True when ``stream`` (default ``sys.stderr``) supports live rewriting."""
    try:
        from topologies.visualization.terminal.caps import probe

        return probe(sys.stderr if stream is None else stream).is_tty
    except Exception:  # noqa: BLE001
        return False


def _term_width(default: int = 100) -> int:
    try:
        from topologies.visualization.terminal.caps import terminal_size

        return max(40, terminal_size(sys.stderr, (default, 24))[0])
    except Exception:  # noqa: BLE001
        return default


def _safe_mem() -> str | None:
    """Accelerator memory (CUDA or MPS); None on CPU-only runs."""
    try:
        from topologies.training.monitoring.system import device_memory_str

        return device_memory_str()
    except Exception:  # noqa: BLE001
        return None


def _legacy_event(event: Any, step: int | None = None) -> Any:
    """Normalize a legacy ``on_*`` payload (Trainer dataclass or namespace)."""
    from topologies.visualization import normalize_event

    name = type(event).__name__
    if name == "ValidationEvent":
        return normalize_event({**vars(event), "global_step": step,
                                "kind": "training.validation_completed"}, source="training")
    return normalize_event(event)


class LegacyFacade(BaseMonitor):
    """``BaseMonitor`` facade owning a new-style visualizer (``.visualizer``).

    When the Trainer created the facade it hands ``facade.visualizer`` to the
    visualizer group directly; standalone ``on_*`` calls are converted with
    ``normalize_event`` and forwarded.
    """

    visualizer: Any = None

    def __init__(self) -> None:
        self._prepared = False
        self._closed = False

    def _ensure(self) -> Any:
        if not self._prepared:
            self._prepared = True
            self.visualizer.prepare("training")
        return self.visualizer

    def _forward(self, event: Any) -> None:
        try:
            viz = self._ensure()
            state = getattr(viz, "state", None)
            viz.handle(_legacy_event(event, getattr(state, "step", None)))
        except Exception:  # noqa: BLE001, S110
            pass

    def on_train_begin(self, event: Any) -> None:
        self._forward(event)

    def on_epoch_begin(self, event: Any) -> None:
        self._forward(event)

    def on_step(self, event: Any) -> None:
        self._forward(event)

    def on_validation_begin(self, event: Any) -> None:
        self._forward(event)

    def on_validation_end(self, event: Any) -> None:
        self._forward(event)

    def on_validation(self, event: Any) -> None:
        self.on_validation_end(event)

    def on_checkpoint(self, event: Any) -> None:
        self._forward(event)

    def on_epoch_end(self, event: Any) -> None:
        self._forward(event)

    def on_train_end(self, event: Any) -> None:
        self._forward(event)

    # -- progress outside the step loop / position in an experiment plan -----------
    def _tracker(self) -> Any:
        return getattr(self._ensure(), "state", None)

    def note(self, label: str, *, done: int | None = None, total: int | None = None) -> None:
        """Progress of work after the training loop (e.g. a long validation pass)."""
        try:
            tracker = self._tracker()
            if tracker is None:
                return
            tracker.set_activity(label, done, total)
            finished = done is not None and total is not None and 0 < int(total) <= int(done)
            self._show_note(label, done, total, finished)
            if finished:
                tracker.set_activity(None)
        except Exception:  # noqa: BLE001, S110 - a display must never break training
            pass

    def _show_note(self, label: str, done: int | None, total: int | None,
                   finished: bool) -> None:
        self.visualizer.refresh()

    def set_plan(self, index: int, total: int, *, done: int | None = None) -> None:
        """This trial is ``index`` (1-based) of ``total`` in an experiment plan."""
        try:
            tracker = self._tracker()
            if tracker is not None:
                tracker.set_plan(index, total, done)
                self._show_plan(tracker)
        except Exception:  # noqa: BLE001, S110
            pass

    def _show_plan(self, tracker: Any) -> None:
        self.visualizer.refresh()

    def close(self) -> None:
        if self._prepared and not self._closed:
            self._closed = True
            try:
                self.visualizer.close()
            except Exception:  # noqa: BLE001, S110
                pass


def _resolve_options(options: Any, defaults: dict[str, Any]) -> StandardMonitorOptions:
    if isinstance(options, StandardMonitorOptions):
        return options
    values = dict(defaults)
    if isinstance(options, dict):
        values.update({k: options[k] for k in defaults if k in options})
    opts = StandardMonitorOptions(**values)
    try:
        opts.log_every_steps = max(1, int(opts.log_every_steps))
    except (TypeError, ValueError):
        opts.log_every_steps = 10
    return opts


class StandardMonitor(LegacyFacade):
    """Default monitor: header + one live line + aligned epoch lines.

    A facade over :class:`~topologies.visualization.training_views.
    StandardTrainingVisualizer` (``plain=True``: the log-friendly plain tier).
    """

    def __init__(
        self,
        options: StandardMonitorOptions | dict | None = None,
        *,
        columns: list[str] | None = None,
        show_eta: bool = True,
        show_mem: bool = True,
        show_lr: bool = True,
        precision: int = 3,
        log_every_steps: int = 10,
        use_ema: bool = True,
        plain: bool = False,
        **extra_options: Any,
    ) -> None:
        super().__init__()
        defaults = {"columns": columns, "show_eta": show_eta, "show_mem": show_mem,
                    "show_lr": show_lr, "precision": precision,
                    "log_every_steps": log_every_steps, "use_ema": use_ema}
        extra = dict(options) if isinstance(options, dict) else {}
        self.options = _resolve_options(options, defaults)
        from topologies.visualization.training_views import (
            PlainTrainingVisualizer,
            StandardTrainingVisualizer,
            view_options,
        )

        merged = {**{k: v for k, v in extra.items() if k not in defaults}, **extra_options}
        opts = view_options("standard", {**merged, **vars(self.options)})
        cls = PlainTrainingVisualizer if plain or merged.get("mode") in ("plain", "compact") \
            else StandardTrainingVisualizer
        self.visualizer = cls(**opts)

    _note_at = 0.0

    def _show_note(self, label: str, done: int | None, total: int | None,
                   finished: bool) -> None:
        """One line per note, at most once a second (always the first and the last)."""
        import time

        now = time.monotonic()
        if not finished and done not in (0, None) and now - self._note_at < 1.0:
            return
        self._note_at = now
        count = "?" if done is None else str(int(done))
        if total is not None:
            count = f"{count}/{int(total)}"
        self.visualizer.print(f"[{label}] {count}" + (" done" if finished else ""))

    def _show_plan(self, tracker: Any) -> None:
        from topologies.visualization.training_views import plan_text

        text = plan_text(tracker)
        if text:
            self.visualizer.print(text)

    @property
    def _is_tty(self) -> bool:
        return getattr(self.visualizer, "screen_type", None) not in ("plain", "null")

    @property
    def _total_epochs(self) -> int | None:
        state = getattr(self.visualizer, "state", None)
        return None if state is None else state.total_epochs

    @property
    def _global_step(self) -> int:
        state = getattr(self.visualizer, "state", None)
        return 0 if state is None else state.step
