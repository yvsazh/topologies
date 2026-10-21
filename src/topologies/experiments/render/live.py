"""Deprecated run-time display shims (``RunState``, ``make_monitor``).

Experiment terminal output is event-driven now: the coordinator emits events and the
views in :mod:`topologies.visualization.experiment_views` render them. ``RunState``
remains importable for code that drove the old board directly; it wraps an
``ExperimentLiveVisualizer`` (``mode="live"``) or ``ExperimentLinesVisualizer``
(``mode="lines"``) and maps the old methods onto events. Display errors are
swallowed: they never affect trials.
"""

from __future__ import annotations

import sys
from typing import Any

from topologies.experiments.render.console import Style, term_width, use_color

_ICON = {"completed": "✓", "failed": "✗", "diverged": "↯", "oom": "▲", "timeout": "◔",
         "pruned": "✂", "interrupted": "‖"}


def _is_tty(stream: Any) -> bool:
    try:
        from topologies.visualization.terminal.caps import probe

        return probe(stream).is_tty
    except Exception:  # noqa: BLE001
        return False


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    from topologies.visualization.format import fmt_duration

    return fmt_duration(seconds)


def _eta_str(seconds: float | None) -> str:
    from topologies.visualization.format import fmt_eta

    return fmt_eta(seconds).lstrip("≈~")


class RunState:
    """Deprecated: drive an experiment view through the old RunState methods."""

    def __init__(self, exp: Any, trials: list, pending: list, plan: Any, *, mode: str,
                 stream: Any = None) -> None:
        from topologies.experiments.coordinator import RunEvents
        from topologies.visualization import SessionInfo, create_visualizer
        from topologies.visualization.experiment_views import make_experiment_view

        self.exp = exp
        self.name = exp.name
        self.goal = exp.goal
        self.trials = trials
        self.total = len(trials)
        self.pending = list(pending)
        self.stream = stream if stream is not None else sys.stdout
        self.tty = _is_tty(self.stream)
        self.mode = "lines" if (mode == "live" and not self.tty) else mode
        self.color = use_color(self.stream)
        self.style = Style(self.color)
        self.view = make_experiment_view(self.mode, stream=self.stream, threaded=False,
                                         summary=False)
        self.group = create_visualizer([self.view] if self.view is not None else [])
        self.events = RunEvents(exp, self.group, trials, pending)
        self.info = self.events.info
        self.running = self.events.running
        self.group.prepare(source="experiment", context=SessionInfo(
            source="experiment", title=exp.name, goal=tuple(exp.goal),
            options={"monitor": self.mode}))
        self.events.started_event()
        self._closed = False

    @property
    def done(self) -> int:
        return self.view.state.done if self.view is not None else 0

    def handle(self, event: dict) -> None:
        self.events.handle(event)

    def trial_start(self, trial: Any) -> None:
        self.events.trial_start(trial)

    def trial_interrupted(self, trial: Any) -> None:
        self.events.trial_interrupted(trial)

    def trial_done(self, trial: Any, result: dict) -> None:
        self.events.trial_done(trial, result)

    def eta(self) -> float | None:
        return None if self.view is None else self.view.state.eta

    def frame(self) -> list[str]:
        """Current live frame as plain lines (empty for ``lines``/``none``)."""
        from topologies.visualization.terminal.text import to_plain

        if self.view is None or self.view.state is None:
            return []
        width = term_width()
        rows = 40
        if self.view._screen is not None:
            width, rows = self.view._screen.size()
        return [to_plain(x) for x in self.view._render_frame(width, rows)]

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.group.close()

    def finish(self, res: Any) -> None:
        if self.mode == "none":
            return
        self.close()
        try:
            self.stream.write("\n" + "\n".join(res.summary_lines(color=self.color)) + "\n")
            self.stream.flush()
            res.meta["summary_printed"] = True
        except Exception as e:  # noqa: BLE001
            self.stream.write(f"<summary failed: {type(e).__name__}: {e}>\n")


def make_monitor(exp: Any, trials: list, pending: list, plan: Any) -> RunState:
    """Deprecated: ``RunState`` for ``exp.monitor``."""
    return RunState(exp, trials, pending, plan, mode=exp.monitor)


__all__ = ["RunState", "make_monitor"]
