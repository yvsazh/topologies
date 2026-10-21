"""Experiment terminal views: lines (log-friendly), live board (inline), dashboard.

All consume an :class:`~topologies.visualization.state.ExperimentTracker` fed by the
coordinator's events. The terminal-owning view prints the final summary once (and
marks the ``ResultSet`` so ``exp.run().show()`` does not print it twice).
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from topologies.visualization.core import VisualizationEvent
from topologies.visualization.format import fmt_clock, fmt_duration, fmt_eta, fmt_number
from topologies.visualization.live import LiveVisualizer
from topologies.visualization.state import ExperimentTracker, TrialSummary, config_key
from topologies.visualization.terminal.caps import TerminalCaps, probe
from topologies.visualization.terminal.glyphs import UNICODE as G
from topologies.visualization.terminal.layout import Rect, grid_layout
from topologies.visualization.terminal.panels import Panel, panel, resolve_panels
from topologies.visualization.terminal.text import Line, line, styled
from topologies.visualization.terminal.widgets import (
    badge,
    box,
    fmt_percent,
    kv_line,
    line_chart,
    progress_bar,
    sparkline,
    table,
)
from topologies.visualization.training_views import compose

_STATUS_TOKEN = {"completed": "good", "pruned": "warn", "timeout": "warn",
                 "interrupted": "warn"}


def status_glyph(status: str) -> tuple[str, str]:
    """``(glyph, token)`` for a trial status; never color alone (callers add the word)."""
    return G.status(status), _STATUS_TOKEN.get(status, "bad")


def _metric_text(metric: str, value: Any) -> str:
    if value is None:
        return "—"
    try:
        from topologies.experiments.metrics import metric_info

        return metric_info(metric).format(value)
    except Exception:  # noqa: BLE001
        return fmt_number(value)


def _arrow(mode: str) -> str:
    return G.down if mode == "min" else G.up


def _cpu_percent() -> float | None:
    try:
        from topologies.training.monitoring.system import _cpu_percent

        return _cpu_percent()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- shared text


class ExperimentFormatter:
    """Header, trial lines, running rows, leaderboard rows and failures."""

    def axis_widths(self, state: ExperimentTracker) -> dict[str, int]:
        widths = {}
        for axis, values in state.axes.items():
            widths[axis] = max((len(f"{axis}={v}") for v in values), default=len(axis) + 2)
        return widths

    def labels(self, state: ExperimentTracker, labels: dict[str, Any], seed: Any, *,
               pad: bool = True) -> str:
        parts = []
        widths = self.axis_widths(state)
        axes = list(state.axes) or [k for k in labels if k != "seed"]
        for axis in axes:
            if axis not in labels:
                continue
            text = f"{axis}={labels[axis]}"
            parts.append(text.ljust(widths.get(axis, len(text))) if pad else text)
        if state.seeds > 1 and seed is not None:
            parts.append(f"seed={seed}")
        return ("  " if pad else " ").join(parts) if parts else (f"seed={seed}" if seed
                                                                 is not None else "—")

    def header(self, state: ExperimentTracker) -> Line:
        configs = len({config_key(t.get("labels")) for t in state.trials.values()}) or None
        bits = [f"{state.total} trials"]
        if configs:
            seeds = max(1, state.seeds)
            bits[0] += f" = {configs} configs × {seeds} seed{'s' if seeds != 1 else ''}"
        bits.append(f"goal {state.metric} {_arrow(state.mode)}")
        bits.append(f"cached {state.cached}")
        bits.append(f"workers {state.workers}")
        return line((state.name, "title"), (" · ", "muted"), " · ".join(bits))

    def trial_line(self, state: ExperimentTracker, rec: TrialSummary, *, width: int | None
                   ) -> Line:
        total = state.total or state.done
        counter = f"{state.done:>{len(str(total))}}/{total}"
        glyph, token = status_glyph(rec.status)
        parts: list[Any] = [" ", counter, " ", (glyph, token), " ",
                            self.labels(state, rec.labels, rec.seed), "   "]
        if rec.status in ("completed", "pruned", "timeout"):
            value = _metric_text(state.metric, rec.value)
            parts += [(state.metric, "muted"), " ", (value, "value"), " ",
                      (G.best, "best") if rec.is_best else " ", "   ",
                      (fmt_duration(rec.wall_s), "muted")]
            if rec.status != "completed":
                parts.append((f"   {rec.status}: {rec.reason or rec.status}", "warn"))
            return line(*parts)
        message = rec.error or rec.status
        parts += [(message, "bad"), (f" → trials/{rec.trial_id[:6]}…/result.json", "muted")]
        out = line(*parts)
        return out.clip(width) if width else out

    def status_line(self, state: ExperimentTracker, width: int) -> Line:
        c = state.counts
        bits = [f"{G.running} running {len(state.running)}",
                f"{state.done}/{state.total} done",
                f"{G.completed}{c.get('completed', 0) + state.cached} {G.failed}{state.failed}",
                f"{fmt_clock(state.elapsed)} elapsed", f"ETA {fmt_eta(state.eta)}"]
        return styled(" · ".join(bits), "muted").clip(width)

    def board_header(self, state: ExperimentTracker, width: int, *, cpu: float | None) -> Line:
        c = state.counts
        stats = [f"{state.done}/{state.total} trials",
                 (f"{G.completed}{c.get('completed', 0) + state.cached} "
                  f"{G.failed}{state.failed} {G.pruned}{c.get('pruned', 0)}"),
                 f"{fmt_clock(state.elapsed)} elapsed", f"ETA {fmt_eta(state.eta)}"]
        rung = state.info.get("rung")
        if rung:
            stats.append(f"rung {rung}")
        if cpu is not None:
            stats.append(f"cpu {cpu:.0f}%")
        tail = styled(" · ".join(stats), None)
        bar_w = max(0, min(30, width - tail.width - len(state.name) - 4))
        bar = progress_bar(state.fraction, bar_w) if bar_w >= 8 else Line()
        return line((state.name, "title"), "  ", bar, " " if bar else "", tail).clip(width)

    def _compact_labels(self, state: ExperimentTracker, labels: dict[str, Any], seed: Any,
                        axes: list[str]) -> str:
        parts = [f"{axis}={labels[axis]}" for axis in axes if axis in labels]
        if state.seeds > 1 and seed is not None:
            parts.append(f"seed={seed}")
        return " ".join(parts) if parts else "—"

    def label_forms(self, state: ExperimentTracker, labels: dict[str, Any], seed: Any
                    ) -> list[str]:
        """Longest axis label first, then the same text with trailing axes removed."""
        axes = [a for a in (list(state.axes) or [k for k in labels if k != "seed"])
                if a in labels]
        forms = [self.labels(state, labels, seed)]
        for n in range(len(axes), 0, -1):
            forms.append(self._compact_labels(state, labels, seed, axes[:n]))
        seen: set[str] = set()
        out = []
        for form in forms:
            if form not in seen:
                seen.add(form)
                out.append(form)
        return out or ["—"]

    def running_row(self, state: ExperimentTracker, trial: Any, width: int) -> Line:
        """Elastic running-trial row.

        Epoch text, the bar and trailing axis labels are dropped whole. The step
        fraction and the latest metric stay as long as any shorter label fits.
        """
        epoch = None
        if trial.epoch is not None and trial.max_epochs and trial.max_epochs < 10**5:
            epoch = f"epoch {trial.epoch}/{trial.max_epochs}"
        frac = trial.fraction
        metric = None
        value = trial.val.get(state.metric)
        if value is not None:
            metric = line((state.metric, "muted"), " ", _metric_text(state.metric, value))
        elif trial.metrics.get("loss") is not None:
            metric = line(("loss", "muted"), " ", fmt_number(trial.metrics["loss"], sig=3))
        # Richest decoration first. Labels shrink only after that decoration no longer fits.
        options = [(True, 12, True), (False, 8, True), (False, 0, True),
                   (False, None, True), (True, 12, False), (False, 8, False),
                   (False, 0, False), (False, None, False)]
        forms = self.label_forms(state, trial.labels, trial.seed)
        for use_epoch, bar_w, use_metric in options:
            if use_epoch and not epoch:
                continue
            if bar_w is not None and frac is None:
                continue
            if use_metric and metric is None:
                continue
            for label in forms:
                parts: list[Any] = [" ", (G.running, "accent"), " ", label]
                if use_epoch and epoch:
                    parts += ["  ", epoch]
                if bar_w is not None and frac is not None:
                    parts += ["  ", progress_bar(frac, bar_w) if bar_w else "",
                              " " if bar_w else "", fmt_percent(frac)]
                if use_metric and metric is not None:
                    parts += ["  ", metric]
                row = line(*parts)
                if row.width <= width:
                    return row
        return line(" ", (G.running, "accent"), " ", forms[-1]).clip(width)

    def running_rows(self, state: ExperimentTracker, width: int, limit: int = 8) -> list[Line]:
        running = list(state.running.values())
        out = [self.running_row(state, trial, width) for trial in running[:limit]]
        if len(running) > limit:
            out.append(styled(f"   +{len(running) - limit} more running", "muted"))
        return out

    def leaderboard_rows(self, state: ExperimentTracker, width: int, top: int = 5) -> list[Line]:
        board = state.leaderboard(top)
        if not board:
            return []
        axes = list(state.axes) or sorted({k for r in board for k in r.labels})
        header = f"{state.metric} {_arrow(state.mode)}"
        rows = []
        for rank, row in enumerate(board, 1):
            value = _metric_text(state.metric, row.mean)
            if row.std is not None:
                value += f" ± {_metric_text(state.metric, row.std)}"
            cell = line(value, (" " + G.best, "best") if row.best else "  ")
            rows.append([rank, *[str(row.labels.get(a, "—")) for a in axes], cell, row.n,
                         fmt_duration(row.wall_s) if row.wall_s is not None else None,
                         sparkline([y for _x, y in row.curve], 8)])
        return table(["#", *axes, header, "n", "time", "curve"], rows, width=width, indent=1)

    def failure_rows(self, state: ExperimentTracker, width: int, limit: int = 3) -> list[Line]:
        bad = [r for r in state.records if r.status not in ("completed", "pruned") and
               not r.cached]
        out = []
        for rec in bad[-limit:]:
            glyph, token = status_glyph(rec.status)
            out.append(line(" ", (glyph, token), " ",
                            self.labels(state, rec.labels, rec.seed, pad=False), ": ",
                            (rec.error or rec.status, "bad")).clip(width))
        return out


# ---------------------------------------------------------------- views


def _caps(stream: Any, tty: bool | None, color: str | None) -> TerminalCaps | None:
    if tty is None and color in (None, "auto"):
        return None
    caps = probe(stream)
    is_tty = caps.is_tty if tty is None else bool(tty)
    depth = caps.color if color in (None, "auto") else (0 if color == "never" else
                                                        max(16, caps.color))
    if tty is False and color in (None, "auto"):
        depth = 0
    return TerminalCaps(is_tty=is_tty, color=depth, unicode=caps.unicode, columns=caps.columns,
                        rows=caps.rows, notebook=caps.notebook and tty is not False,
                        ci=caps.ci, fullscreen_ok=caps.fullscreen_ok and is_tty,
                        windows_vt=caps.windows_vt, stream=stream, size_source=caps.size_source)


class ExperimentLinesVisualizer(LiveVisualizer):
    """One line per finished trial (+ a sticky status line on a TTY): ``monitor="lines"``."""

    live_screen = "inline"

    def __init__(self, *, status_line: bool = True, summary: bool = True, system: bool = True,
                 stream: Any = None,
                 refresh_rate: float | None = None, capture_output: bool | None = None,
                 theme: Any = "default", threaded: bool | None = None,
                 caps: TerminalCaps | None = None, tty: bool | None = None,
                 color: str | None = None, screen: str | None = None, clock: Any = None,
                 **_ignored: Any) -> None:
        super().__init__(screen=screen, stream=stream, refresh_rate=refresh_rate,
                         capture_output=capture_output, theme=theme, threaded=threaded,
                         caps=caps)
        self.status_line = status_line
        self.summary = summary
        self.system = system
        self.tty = tty
        self.color = color
        self.clock = clock
        self.fmt = ExperimentFormatter()
        self._summary_lines: list[Any] = []

    def make_state(self, source: str | None) -> ExperimentTracker:
        return ExperimentTracker(clock=self.clock) if self.clock is not None else \
            ExperimentTracker()

    def screen_kind(self, caps: TerminalCaps) -> str:
        kind = super().screen_kind(caps)
        if kind == "inline" and not self.status_line and type(self) is ExperimentLinesVisualizer:
            return "plain"
        return kind

    def prepare(self, source: str | None = None, context: Any = None) -> None:
        stream = self.stream if self.stream is not None else self.default_stream("experiment")
        if self.caps is None or getattr(self, "_derived_caps", False):
            derived = _caps(stream, self.tty, self.color)
            if derived is not None:
                self.caps = derived
                self._derived_caps = True
        self._summary_lines = []
        super().prepare("experiment" if source is None else source, context)

    @property
    def board(self) -> bool:
        return False

    def _width(self) -> int | None:
        if self._screen is None or self.screen_type == "plain":
            return None
        return self._screen.size()[0] - 1

    def on_experiment_start(self, event: VisualizationEvent) -> None:
        if not self.board:
            self.print(self.fmt.header(self.state))

    def on_trial_end(self, event: VisualizationEvent) -> None:
        if self.board or event.kind == "experiment.trial_interrupted" or not self.state.records:
            return
        self.print(self.fmt.trial_line(self.state, self.state.records[-1], width=self._width()))

    def draw(self, state: ExperimentTracker, width: int, height: int) -> list[Line]:
        if state.status != "running" or not self.status_line:
            return []
        return [self.fmt.status_line(state, width - 1)]

    def render(self, result: Any) -> Any:
        super().render(result)
        data = getattr(result, "data", None)
        rs = data.get("resultset") if isinstance(data, dict) or hasattr(data, "get") else None
        if self.summary and rs is not None and self.screen_type != "null":
            try:
                width = None
                if self._screen is not None and self.screen_type != "plain":
                    width = self._screen.size()[0] - 1
                color = bool(self._screen is not None and self._screen.color > 0)
                self._summary_lines = ["", *rs.summary_lines(width=width, color=color)]
                rs.meta["summary_printed"] = True
            except Exception as exc:  # noqa: BLE001
                self._summary_lines = [f"<summary failed: {type(exc).__name__}: {exc}>"]
        return None

    def final(self, state: ExperimentTracker) -> list[Any] | None:
        return list(self._summary_lines) or None


class ExperimentLiveVisualizer(ExperimentLinesVisualizer):
    """Inline live board (running trials, top-5 leaderboard, recent failures): ``"live"``."""

    @property
    def board(self) -> bool:
        return self.screen_type in ("inline", "notebook")

    def draw(self, state: ExperimentTracker, width: int, height: int) -> list[Line]:
        if not self.board or state.status != "running":
            return []
        w = width - 1
        lines = [self.fmt.board_header(state, w, cpu=_cpu_percent() if self.system else None)]
        lines += self.fmt.running_rows(state, w)
        lines += self.fmt.leaderboard_rows(state, w)
        lines += self.fmt.failure_rows(state, w)
        limit = max(1, min(height - 1, 4 + len(state.running) + 5 + 3))
        return lines[:limit]


# ---------------------------------------------------------------- dashboard panels


def _view(state: Any) -> Any:
    return getattr(state, "dashboard", None) or SimpleNamespace(scroll=0, log_y=False)


_FMT = ExperimentFormatter()


@panel("header", title="experiment", min_height=3, priority=100, domain="experiment",
       min_width=30)
def exp_header_panel(state: ExperimentTracker, width: int, height: int) -> list[Line]:
    c = state.counts
    frac = state.fraction
    out = [line(progress_bar(frac, max(10, width - 28)), " ", fmt_percent(frac), "  ",
                f"{state.done}/{state.total} trials")]
    out.append(kv_line([
        ("", f"{G.completed} {c.get('completed', 0) + state.cached} completed"),
        ("", f"{G.failed} {state.failed} failed" if state.failed else None),
        ("", f"{G.pruned} {c['pruned']} pruned" if c.get("pruned") else None),
        ("running", str(len(state.running))), ("elapsed", fmt_clock(state.elapsed)),
        ("ETA", fmt_eta(state.eta)), ("goal", f"{state.metric} {_arrow(state.mode)}"),
        ("workers", str(state.workers))], width))
    return out


def _exp_badge(state: ExperimentTracker) -> Line:
    if state.status == "done":
        b = badge("done", "good", G.completed)
    elif state.status == "failed":
        b = badge("failed", "bad", G.failed)
    else:
        b = badge("running", "accent", "●")
    return line(b, "  ", (fmt_clock(state.elapsed), "muted"))


exp_header_panel.title_right = _exp_badge


@panel("running", title="running", min_height=3, preferred_height=6, priority=90,
       domain="experiment", min_width=30)
def running_panel(state: ExperimentTracker, width: int, height: int) -> list[Line]:
    rows = _FMT.running_rows(state, width, limit=max(1, height - 1))
    return rows or [styled("no trial running", "muted")]


@panel("leaderboard", title="leaderboard", min_height=5, preferred_height=10, priority=80,
       domain="experiment", min_width=36)
def leaderboard_panel(state: ExperimentTracker, width: int, height: int) -> list[Line]:
    rows = _FMT.leaderboard_rows(state, width, top=max(1, height - 2))
    return rows or [styled("no completed trial yet", "muted")]


DEFAULT_CHARTS = ("loss", "goal")


def normalize_charts(charts: Any) -> tuple[str, ...]:
    """``"loss"`` is train loss versus step; ``"goal"`` or the metric name is the goal curve."""
    if charts is None:
        return DEFAULT_CHARTS
    if isinstance(charts, str):
        items = [charts]
    elif isinstance(charts, Sequence) and not isinstance(charts, (str, bytes)):
        items = list(charts)
    else:
        raise TypeError("charts must be a string or a list of strings")
    out = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("charts entries must be non-empty strings")
        out.append(item.strip())
    if not out:
        raise ValueError("charts must list at least one series")
    return tuple(out)


def _thin(points: list[tuple[float, float]], budget: int) -> list[tuple[float, float]]:
    if len(points) <= budget:
        return list(points)
    stride = max(1, (len(points) - 1) // max(1, budget - 1))
    out = list(points[::stride])
    if out[-1][0] != points[-1][0]:
        out.append(points[-1])
    return out


def _goal_series(state: ExperimentTracker) -> tuple[dict[str, Any], dict[str, int]]:
    series: dict[str, Any] = {}
    slots: dict[str, int] = {}
    for row in state.leaderboard(5):
        if row.curve:
            name = _FMT.labels(state, row.labels, None, pad=False)
            series[name] = row.curve
            slots[name] = row.slot
    for trial in state.running.values():
        if trial.curve:
            name = f"{G.running} " + _FMT.labels(state, trial.labels, trial.seed, pad=False)
            series[name] = trial.curve
            slots[name] = state.slot(trial.labels)
    return series, slots


def _loss_series(state: ExperimentTracker, budget: int) -> tuple[dict[str, Any], dict[str, int]]:
    series: dict[str, Any] = {}
    slots: dict[str, int] = {}
    for rec in state.records:
        if not rec.loss_curve:
            continue
        name = _FMT.labels(state, rec.labels, rec.seed, pad=False)
        series[name] = _thin(rec.loss_curve, budget)
        slots[name] = state.slot(rec.labels)
    for trial in state.running.values():
        if not trial.loss_curve:
            continue
        name = f"{G.running} " + _FMT.labels(state, trial.labels, trial.seed, pad=False)
        series[name] = _thin(trial.loss_curve, budget)
        slots[name] = state.slot(trial.labels)
    return series, slots


def _one_chart(state: ExperimentTracker, name: str, width: int, height: int) -> list[Line]:
    if name in ("loss", "train", "train_loss"):
        series, slots = _loss_series(state, max(8, width))
        title, empty = "loss · step", "no train loss yet"
    elif name == "goal" or name == state.metric:
        series, slots = _goal_series(state)
        title, empty = f"{state.metric} · epoch", f"no {state.metric} curve yet"
    else:
        series, slots = {}, {}
        title, empty = name, f"no {name} curve yet"
    head = styled(title, "muted")
    if height <= 1 or not series:
        return [head, styled(empty, "muted")][:height]
    body = line_chart(series, width, max(2, height - 2), slots=slots,
                      log_y=_view(state).log_y, x_integer=True)
    return [head, *body][:height]


@panel("curves", title="curves", min_height=8, preferred_height=16, priority=70,
       domain="experiment", min_width=36)
def curves_panel(state: ExperimentTracker, width: int, height: int) -> list[Line]:
    charts = normalize_charts(getattr(state, "charts", None))
    base, extra = divmod(height, len(charts))
    out: list[Line] = []
    for index, name in enumerate(charts):
        share = base + (extra if index == len(charts) - 1 else 0)
        out.extend(_one_chart(state, name, width, share))
    return out[:height]


@panel("insights", title="insights", min_height=3, preferred_height=4, priority=60,
       domain="experiment", min_width=30)
def insights_panel(state: ExperimentTracker, width: int, height: int) -> list[Line]:
    board = state.leaderboard(None)
    if not board:
        return [styled("waiting for the first completed trial", "muted")]
    best = board[0]
    out = [line((G.best + " best ", "best"), _FMT.labels(state, best.labels, None, pad=False),
                "  ", (state.metric, "muted"), " ", _metric_text(state.metric, best.mean))]
    if len(board) > 1 and best.mean is not None and board[1].mean is not None:
        margin = abs(board[1].mean - best.mean)
        out.append(line((f"{G.bullet} margin to #2 ", "muted"),
                        _metric_text(state.metric, margin)))
    if state.failed:
        out.append(line(("▲ ", "warn"), f"{state.failed} trial(s) failed"))
    return out


@panel("log", title="log", min_height=3, preferred_height=6, priority=40, domain="experiment",
       min_width=30)
def exp_log_panel(state: ExperimentTracker, width: int, height: int) -> list[Line]:
    entries = list(state.log)[-max(1, height):]
    styles = {"error": "bad", "warn": "warn", "out": None}
    return [styled(t, styles.get(lvl, "muted")) for _ts, lvl, t in entries] or \
        [styled("no events yet", "muted")]


def trials_table(state: ExperimentTracker, width: int, height: int) -> list[Line]:
    view = _view(state)
    recs = list(state.records)
    room = max(1, height - 2)
    scroll = max(0, min(int(getattr(view, "scroll", 0) or 0), max(0, len(recs) - room)))
    end = len(recs) - scroll
    shown = recs[max(0, end - room): end]
    axes = list(state.axes)
    rows = []
    for rec in shown:
        glyph, token = status_glyph(rec.status)
        rows.append([rec.index, (f"{glyph} {rec.status}", token),
                     *[str(rec.labels.get(a, "—")) for a in axes], rec.seed,
                     _metric_text(state.metric, rec.value) if rec.value is not None else None,
                     fmt_duration(rec.wall_s) if rec.wall_s is not None else None,
                     rec.error])
    return table(["#", "status", *axes, "seed", state.metric, "time", "error"], rows,
                 width=width, indent=0, hide_empty=False)


def axis_curves_panel(axis: str, metric: str = "goal metric") -> Panel:
    def render(state: ExperimentTracker, width: int, height: int) -> list[Line]:
        by_value: dict[str, list[list[tuple[float, float]]]] = {}
        for rec in state.records:
            if rec.status == "completed" and rec.curve:
                by_value.setdefault(str(rec.labels.get(axis)), []).append(rec.curve)
        if not by_value:
            return [styled("no curves yet", "muted")]
        series = {}
        for value, curves in by_value.items():
            xs: dict[float, list[float]] = {}
            for curve in curves:
                for x, y in curve:
                    xs.setdefault(x, []).append(y)
            series[f"{axis}={value}"] = [(x, sum(v) / len(v)) for x, v in sorted(xs.items())]
        return line_chart(series, width, max(2, height - 3), x_integer=True,
                          log_y=_view(state).log_y)[:height]

    return Panel(f"axis:{axis}", render, title=f"{metric} by {axis}", min_height=8,
                 preferred_height=14, priority=50, domain="experiment", min_width=30)


EXP_DEFAULT_PANELS = ("header", "running", "leaderboard", "curves", "insights", "log")
EXP_VIEWS = ("overview", "trials", "curves", "log")


class ExperimentDashboard(ExperimentLiveVisualizer):
    """Fullscreen experiment dashboard (``monitor="dashboard"``); inline board otherwise."""

    live_screen = "fullscreen"

    def __init__(self, *, fullscreen: bool | None = None, panels: Sequence[Any] | None = None,
                 view: str = "overview", interactive: bool = True,
                 charts: Sequence[str] | None = None, **options: Any) -> None:
        super().__init__(**options)
        self.fullscreen = fullscreen
        if fullscreen is False:
            self.live_screen = "inline"
        self.panels = resolve_panels(panels, "experiment", EXP_DEFAULT_PANELS)
        self.view_name = view if view in EXP_VIEWS else "overview"
        self.interactive = interactive
        self.charts = normalize_charts(charts)
        self.help = False
        self.message: str | None = None
        self._keys: Any = None

    @property
    def board(self) -> bool:
        return self.screen_type in ("inline", "notebook", "fullscreen")

    def on_session_start(self) -> None:
        self.state.dashboard = SimpleNamespace(scroll=0, log_y=False)
        if self._capture is not None:
            self._capture.sink = self._captured
        if self.screen_type == "fullscreen" and self.interactive:
            from topologies.visualization.terminal.keys import KeyReader

            self._keys = KeyReader(self.on_key)
            if not self._keys.start():
                self._keys = None
            elif self._screen is not None:
                self._screen.on_end.append(self._keys.stop)

    def _captured(self, text: str) -> None:
        if self.state is not None:
            self.state.add_log(text, "out")
        if self._driver is not None:
            self._driver.print(text)

    def on_session_end(self) -> None:
        if self._keys is not None:
            self._keys.stop()
            self._keys = None

    def on_key(self, key: str) -> None:
        view = self.state.dashboard
        if key == "?":
            self.help = not self.help
        elif key == "tab":
            self.view_name = EXP_VIEWS[(EXP_VIEWS.index(self.view_name) + 1) % len(EXP_VIEWS)]
        elif key in ("1", "2", "3", "4"):
            self.view_name = EXP_VIEWS[int(key) - 1]
        elif key == "l":
            view.log_y = not view.log_y
        elif key == "up":
            view.scroll = int(view.scroll or 0) + 1
        elif key == "down":
            view.scroll = max(0, int(view.scroll or 0) - 1)
        elif key == "p" and self._driver is not None:
            self._driver.paused = not self._driver.paused
            self.message = "paused (data still collected)" if self._driver.paused else None
        elif key == "q":
            self._leave_fullscreen()
            return
        if self._driver is not None:
            self._driver.refresh()

    def _leave_fullscreen(self) -> None:
        if self.screen_type != "fullscreen" or self._driver is None:
            return
        from topologies.visualization.terminal.driver import LiveDriver
        from topologies.visualization.terminal.screen import InlineRegion

        if self._keys is not None:
            self._keys.stop()
            self._keys = None
        old = self._screen
        self._driver.stop(render_final=False)
        self._screen = InlineRegion(old.stream, caps=old.caps, theme=self.theme)
        self._driver = LiveDriver(self._screen, self._render_frame, fps=10.0,
                                  threaded=self.threaded)
        self._driver.start()
        if self._capture is not None:
            self._capture.sink = self._driver.print

    def draw(self, state: ExperimentTracker, width: int, height: int) -> list[Line]:
        if self.screen_type != "fullscreen":
            return super().draw(state, width, height)
        return self.frame(state, width, height)

    def frame(self, state: ExperimentTracker, width: int, height: int) -> list[Line]:
        state.charts = self.charts
        if width < 60 or height < 12:
            return [line((state.name, "title"), "  ", _exp_badge(state)).clip(width),
                    line(progress_bar(state.fraction, max(4, width - 8)),
                         fmt_percent(state.fraction)).clip(width),
                    styled("enlarge terminal for the full dashboard", "muted").clip(width)]
        blocks: list[tuple[Rect, Sequence[Line]]] = []
        header = next((p for p in self.panels if p.name == "header"), None)
        top = 0
        if header is not None:
            body = exp_header_panel(state, width - 4, 2)
            hdr = box(body, f"topologies ─ {state.name}", width, len(body) + 2,
                      title_right=_exp_badge(state))
            blocks.append((Rect(0, 0, width, len(hdr)), hdr))
            top = len(hdr)
        area = Rect(0, top, width, max(0, height - 1 - top))
        if self.help:
            keys = [("?", "toggle help"), ("tab / 1-4", "overview, trials, curves, log"),
                    ("↑ ↓", "scroll trials"), ("l", "log scale"), ("p", "pause drawing"),
                    ("q", "leave fullscreen")]
            lines = [line((k.ljust(10), "key"), " ", d) for k, d in keys]
            help_box = box(lines, "keys", min(width, 60), min(area.h, len(lines) + 2))
            blocks.append((Rect(0, area.y, min(width, 60), len(help_box)), help_box))
        else:
            for panel_obj, rect in grid_layout(self._body_panels(state), area):
                blocks.append((rect, panel_obj.render(state, rect.w, rect.h)))
        hints = f"[{self.view_name}] q quit · tab view · ? help"
        notable = [t for _ts, lvl, t in state.log if lvl != "out"]
        left = self.message or (notable[-1] if notable else "")
        right = styled(hints, "key")
        room = max(0, width - 1 - right.width - 2)
        footer = line(" ", styled(left, "muted").clip(room).pad(room), " ", right)
        blocks.append((Rect(0, height - 1, width - 1, 1), [footer]))
        return compose(blocks, width, height)

    def _body_panels(self, state: ExperimentTracker) -> list[Panel]:
        if self.view_name == "trials":
            return [Panel("trials", trials_table, title="trials", min_height=4,
                          preferred_height=200, priority=100, domain="experiment")]
        if self.view_name == "curves":
            return [axis_curves_panel(axis, state.metric) for axis in state.axes] or \
                [p for p in self.panels if p.name == "curves"]
        if self.view_name == "log":
            return [Panel("log", exp_log_panel, title="log", min_height=3,
                          preferred_height=200, priority=100, domain="experiment")]
        return [p for p in self.panels if p.name != "header"]


# ---------------------------------------------------------------- factory


def make_experiment_view(mode: str, **options: Any) -> Any:
    """View for an ``Experiment(monitor=...)`` mode; ``None`` for ``"none"``."""
    mode = (mode or "live").strip().lower()
    if mode in ("none", "quiet", "off"):
        return None
    if mode in ("lines", "plain"):
        return ExperimentLinesVisualizer(**options)
    if mode == "dashboard":
        return ExperimentDashboard(**options)
    return ExperimentLiveVisualizer(**options)


__all__ = [
    "EXP_DEFAULT_PANELS",
    "ExperimentDashboard",
    "ExperimentFormatter",
    "ExperimentLinesVisualizer",
    "ExperimentLiveVisualizer",
    "make_experiment_view",
    "status_glyph",
]
