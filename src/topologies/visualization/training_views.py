"""Trainer terminal views: plain (log lines), standard (one live line), dashboard.

All three consume a :class:`~topologies.visualization.state.TrainingTracker` and are
built from the public toolkit (``terminal`` widgets, screens, driver, panels).

* :class:`PlainTrainingVisualizer` — grep-friendly ``key=value`` lines, no cursor control.
* :class:`StandardTrainingVisualizer` — header + aligned epoch lines + one sticky live line
  (behaves like plain when the stream is not a terminal).
* :class:`TrainingDashboard` — fullscreen panels with keyboard shortcuts; inline compact
  frame when fullscreen is unavailable; one snapshot per epoch off a terminal.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from topologies.visualization.core import VisualizationEvent
from topologies.visualization.format import (
    direction,
    fmt_bytes,
    fmt_clock,
    fmt_count,
    fmt_lr,
    fmt_metric,
    fmt_number,
    fmt_rate,
)
from topologies.visualization.live import LiveVisualizer
from topologies.visualization.state import EpochRow, TrainingTracker
from topologies.visualization.terminal.caps import TerminalCaps, probe
from topologies.visualization.terminal.glyphs import UNICODE as G
from topologies.visualization.terminal.layout import Rect, grid_layout
from topologies.visualization.terminal.panels import Panel, panel, resolve_panels
from topologies.visualization.terminal.text import Line, as_line, line, styled
from topologies.visualization.terminal.widgets import (
    badge,
    box,
    fmt_percent,
    kv_line,
    line_chart,
    meter,
    progress_bar,
    rule,
    sparkline,
    table,
)

_LR = ("lr", "learning_rate")
_SMOOTHING_STEPS = (0.0, 0.6, 0.9)


# ---------------------------------------------------------------- text formatting


def _info(state: Any, name: str) -> Any:
    run = getattr(state, "run", None)
    value = getattr(run, name, None) if run is not None else None
    return None if value in ("", "None") else value


def _is_lr(name: str) -> bool:
    return name in _LR or name.endswith("_lr")


def loss_key(state: TrainingTracker) -> str | None:
    names = state.metric_names("train")
    for key in ("loss", "train_loss"):
        if key in names:
            return key
    return next((k for k in names if "loss" in k), None)


def val_key(state: TrainingTracker) -> str | None:
    names = state.metric_names("val")
    if state.monitored in names:
        return state.monitored
    for key in ("val_loss", "val"):
        if key in names:
            return key
    return next(iter(names), None)


def _value_text(name: str, value: Any, sig: int = 4) -> str:
    if value is None:
        return "n/a"
    return fmt_lr(value) if _is_lr(name) else fmt_number(value, sig=sig)


class TrainingFormatter:
    """Shared text for the Trainer tiers (header, step, epoch, done and live lines)."""

    def __init__(self, *, columns: Sequence[str] | None = None, show_eta: bool = True,
                 show_mem: bool = True, show_lr: bool = True, precision: int = 3,
                 use_ema: bool = True, run_dir: str | None = None) -> None:
        self.columns = list(columns) if columns else None
        self.show_eta = show_eta
        self.show_mem = show_mem
        self.show_lr = show_lr
        self.precision = max(1, int(precision or 3))
        self.use_ema = use_ema
        self.run_dir = run_dir
        self._epoch_cols: list[tuple[str, int, int | None]] = []
        self.epoch_lines: list[Line] = []

    # -- header -----------------------------------------------------------------
    def header_fields(self, state: TrainingTracker, *, plain: bool) -> list[str]:
        out: list[str] = []
        for name in ("model_name", "recipe_name"):
            if _info(state, name):
                out.append(str(_info(state, name)))
        params = _info(state, "param_total")
        if params is not None:
            out.append(f"{fmt_count(params)} params")
        device = " ".join(str(x) for x in (_info(state, "device"), _info(state, "precision"))
                          if x)
        if device:
            out.append(device)
        opt = _info(state, "optimizer_name")
        lr = _info(state, "lr")
        if opt or lr is not None:
            lr_text = None if lr is None else (f"lr={fmt_lr(lr)}" if plain else f"lr {fmt_lr(lr)}")
            out.append(" ".join(str(x) for x in (opt, lr_text) if x))
        batch = _info(state, "batch_size")
        if batch is not None and not plain:
            out.append(f"batch {batch}")
        epochs = state.total_epochs or _info(state, "max_epochs")
        spe = state.steps_per_epoch or _info(state, "steps_per_epoch")
        def _n(count, unit):
            return f"{count} {unit}" + ("" if count == 1 else "s")

        if epochs and spe:
            out.append(f"{_n(epochs, 'epoch')} × {_n(spe, 'step')}" if plain
                       else f"{epochs} × {_n(spe, 'step')}")
        elif epochs:
            out.append(_n(epochs, "epoch"))
        elif state.total_steps:
            out.append(_n(state.total_steps, "step"))
        if plain and batch is not None:
            out.append(f"batch {batch}")
        seed = _info(state, "seed")
        if seed is not None and plain:
            out.append(f"seed {seed}")
        if plain:
            ckpt = _info(state, "checkpoint_dir") or _info(state, "run_dir")
            if ckpt:
                out.append(f"checkpoint {ckpt}")
        return out

    def plain_header(self, state: TrainingTracker) -> Line:
        return line(("[train] ", "muted"), " · ".join(self.header_fields(state, plain=True)))

    def standard_header(self, state: TrainingTracker) -> Line:
        fields = self.header_fields(state, plain=False)
        parts: list[Any] = []
        for i, f in enumerate(fields):
            if i:
                parts.append((" · ", "muted"))
            parts.append((f, "title") if i == 0 else f)
        return line(*parts)

    # -- step metrics -------------------------------------------------------------
    def step_keys(self, state: TrainingTracker) -> list[str]:
        if self.columns:
            return [c for c in self.columns if c != "val"]
        names = [n for n in state.metric_names("train") if not _is_lr(n)]
        losses = [n for n in names if "loss" in n]
        return losses[:2] if losses else names[:1]

    def lr_value(self, state: TrainingTracker) -> float | None:
        for key in _LR:
            v = state.latest_value(key)
            if v is not None:
                return v
        return None

    def step_line(self, state: TrainingTracker) -> Line:
        total = state.total_steps
        width = len(str(total)) if total else 6
        counter = f"[step {state.step:>{width}}/{total}]" if total else f"[step {state.step:>6}]"
        parts: list[Any] = [counter]
        if state.total_epochs:
            parts.append(f" epoch {state.epoch + 1}/{state.total_epochs}")
        for key in self.step_keys(state):
            value = state.latest_value(key)
            if value is not None:
                parts.append(f"  {key}={_value_text(key, value, self.precision)}")
        if self.columns and "val" in self.columns:
            vk = val_key(state)
            if vk and state.latest_value(vk) is not None:
                parts.append(f"  {vk}={fmt_number(state.latest_value(vk), sig=self.precision)}")
        lr = self.lr_value(state)
        if self.show_lr and lr is not None:
            parts.append(f"  lr={fmt_lr(lr)}")
        if state.throughput:
            parts.append(f"  {fmt_rate(state.throughput)}")
        return line(*parts)

    # -- epochs ------------------------------------------------------------------
    @staticmethod
    def epoch_label(row: EpochRow, state: TrainingTracker) -> str:
        total = state.total_epochs
        return f"{row.epoch + 1}/{total}" if total else str(row.epoch + 1)

    def _checkpoint(self, path: str | None, *, full: bool) -> str | None:
        if not path:
            return None
        if full:
            return str(path)
        base = self.run_dir
        if base:
            with contextlib.suppress(ValueError):
                return os.path.relpath(str(path), str(base))
        return str(path)

    def plain_epoch_line(self, row: EpochRow, state: TrainingTracker) -> Line:
        parts: list[Any] = [f"[epoch {self.epoch_label(row, state)}]"]
        for name, value in row.metrics.items():
            parts.append(f" {name}={_value_text(name, value)}")
        if row.duration is not None:
            parts.append(f"  time={fmt_clock(row.duration)}")
        if row.is_best:
            parts.append(("  ★ best", "best"))
        ckpt = self._checkpoint(row.checkpoint, full=True)
        if ckpt:
            parts.append(f"  checkpoint={ckpt}")
        return line(*parts)

    @staticmethod
    def _decimals(name: str, value: float) -> int | None:
        """Fixed decimals for an epoch column (None = lr/scientific formatting)."""
        if _is_lr(name) or value == 0 or not (1e-3 <= abs(value) < 1e6):
            return None
        text = fmt_number(value, sig=4)
        return len(text.split(".")[1]) if "." in text else 0

    def _cell(self, name: str, value: float | None, dec: int | None) -> str:
        if value is None:
            return "—"
        if dec is None or not (1e-3 <= abs(value) < 1e6):
            return _value_text(name, value)
        return f"{value:.{dec}f}"

    def epoch_line(self, row: EpochRow, prev: EpochRow | None, state: TrainingTracker) -> Line:
        for n, v in row.metrics.items():
            if n not in {c[0] for c in self._epoch_cols}:
                dec = self._decimals(n, v)
                self._epoch_cols.append((n, len(self._cell(n, v, dec)), dec))
        total = state.total_epochs
        label_w = len(f"epoch {total}/{total}") if total else 8
        parts: list[Any] = [(f"epoch {self.epoch_label(row, state)}".ljust(label_w), "label"),
                            "  "]
        cols = []
        for k, (name, w, dec) in enumerate(self._epoch_cols):
            value = row.metrics.get(name)
            text = self._cell(name, value, dec)
            if len(text) > w:
                self._epoch_cols[k] = (name, len(text), dec)
                w = len(text)
            cols.append((name, text.rjust(w), value))
        for name, text, value in cols:
            parts += [(name, "muted"), " ", (text, "value" if value is not None else "muted"), " "]
            parts.append(self._trend(name, value, prev))
            parts.append(" ")
        parts.append((G.best, "best") if row.is_best else " ")
        if row.duration is not None:
            parts += [" ", (fmt_clock(row.duration), "muted")]
        ckpt = self._checkpoint(row.checkpoint, full=False)
        if ckpt:
            parts += ["  ", (f"saved {ckpt}", "muted")]
        return line(*parts)

    @staticmethod
    def _trend(name: str, value: float | None, prev: EpochRow | None) -> tuple[str, Any]:
        if value is None or prev is None or prev.metrics.get(name) is None:
            return (" ", None)
        before = prev.metrics[name]
        if before == value or abs(value - before) <= 1e-3 * max(abs(before), 1e-12):
            return (G.flat, "muted")
        better = (value > before) if direction(name) == "max" else (value < before)
        arrow = G.up if value > before else G.down
        return (arrow, "good" if better else "warn")

    def done_line(self, state: TrainingTracker, *, plain: bool) -> Line:
        epochs = len(state.epochs) or (state.epoch + 1 if state.step else 0)
        elapsed = state.duration if state.duration is not None else state.elapsed
        stats = [f"{epochs} epoch{'s' if epochs != 1 else ''}", f"{state.step:,} steps",
                 fmt_clock(elapsed)]
        if state.best_metric is not None:
            name = state.monitored or "metric"
            at = f" @ epoch {state.best_epoch + 1}" if state.best_epoch is not None else ""
            sep = "=" if plain else " "
            stats.append(f"best {name}{sep}{fmt_number(state.best_metric)}{at}")
        if plain:
            tag = {"done": "[done]", "failed": "[failed]"}.get(state.status, "[stopped]")
            tail = f" · {state.error}" if state.status == "failed" and state.error else ""
            return line(tag, " ", " · ".join(stats), tail)
        if state.status == "failed":
            head = badge("failed", "bad", G.failed)
            stats.append(state.error or "error")
        elif state.status == "done":
            head = badge("done", "good", G.completed)
        else:
            head = badge("stopped", "warn", G.interrupted)
        return line(head, "  ", (" · ".join(stats), None))

    # -- live line -----------------------------------------------------------------
    def live_line(self, state: TrainingTracker, width: int, *, mem: str | None = None) -> Line:
        """Elastic segment layout: low-priority segments are dropped whole when narrow."""
        width = max(10, width)
        frac = state.epoch_fraction if state.epoch_fraction is not None else state.fraction
        segments: dict[str, Line] = {"pct": styled(fmt_percent(frac), "value")}
        if state.total_epochs:
            ep = f"epoch {state.epoch + 1}/{state.total_epochs}"
            if state.steps_per_epoch:
                ep += f" · {state.epoch_step}/{state.steps_per_epoch}"
        elif state.total_steps:
            ep = f"step {state.step:,}/{state.total_steps:,}"
        else:
            ep = f"step {state.step:,}"
        segments["epoch"] = styled(ep, None)
        clock = fmt_clock(state.elapsed)
        eta = state.eta if self.show_eta else None
        segments["time"] = line((clock, None), ((" < " + fmt_clock(eta)) if eta is not None
                                                else "", "muted"))
        metric_keys = []
        for key in self.step_keys(state):
            value = state.smoothed(key) if self.use_ema else state.latest_value(key)
            if value is None:
                continue
            metric_keys.append(f"m:{key}")
            segments[f"m:{key}"] = line((key, "muted"), " ", fmt_number(value, sig=self.precision))
        vk = val_key(state)
        if vk and (not self.columns or "val" in self.columns or vk in self.columns):
            value = state.latest_value(vk)
            if value is not None and f"m:{vk}" not in segments:
                metric_keys.append(f"m:{vk}")
                segments[f"m:{vk}"] = line((vk, "muted"), " ",
                                           fmt_number(value, sig=self.precision))
        act = getattr(state, "activity", None)
        if act:
            total = act.get("total")
            segments["rate"] = styled(f"{act['label']} {act.get('done') or 0}"
                                      + (f"/{total}" if total else ""), "accent")
        elif state.status == "validating":
            segments["rate"] = styled("validating…", "accent")
        elif state.throughput:
            segments["rate"] = styled(fmt_rate(state.throughput), None)
        lr = self.lr_value(state)
        if self.show_lr and lr is not None:
            segments["lr"] = line(("lr", "muted"), " ", fmt_lr(lr))
        if self.show_mem and mem:
            segments["mem"] = line(("mem", "muted"), " ", mem)
        priority = ["pct", "bar", "epoch", "time", *metric_keys, "rate", "lr", "mem"]
        display = ["bar", "pct", "epoch", "time", "rate", *metric_keys, "lr", "mem"]
        gap = 2
        chosen: set[str] = set()
        used = 0
        for key in priority:
            if key == "bar":
                need = 10
                if width < 70 or frac is None:
                    continue
            elif key in segments:
                need = segments[key].width
            else:
                continue
            extra = need + (gap if chosen else 0)
            if used + extra <= width:
                chosen.add(key)
                used += extra
        if "bar" in chosen:
            bar_w = min(40, 10 + (width - used))
            segments["bar"] = progress_bar(frac, bar_w)
        parts: list[Any] = []
        for key in display:
            if key in chosen:
                if parts:
                    parts.append(" " * gap)
                parts.append(segments[key])
        return line(*parts).clip(width)


# ---------------------------------------------------------------- base view


def _options_caps(stream: Any, tty: bool | None, color: str | None) -> TerminalCaps | None:
    if tty is None and color in (None, "auto"):
        return None
    caps = probe(stream)
    is_tty = caps.is_tty if tty is None else bool(tty)
    depth = caps.color if color in (None, "auto") else (0 if color == "never" else
                                                        max(16, caps.color))
    if tty is False and color in (None, "auto"):
        depth = 0
    return TerminalCaps(is_tty=is_tty, color=depth, unicode=caps.unicode, columns=caps.columns,
                        rows=caps.rows, notebook=caps.notebook and tty is not False, ci=caps.ci,
                        fullscreen_ok=caps.fullscreen_ok and is_tty, windows_vt=caps.windows_vt,
                        stream=stream, size_source=caps.size_source)


class _TrainingView(LiveVisualizer):
    """Common option handling for the Trainer tiers."""

    def __init__(self, options: Any = None, *, columns: Sequence[str] | None = None,
                 show_eta: bool = True, show_mem: bool = True, show_lr: bool = True,
                 precision: int = 3, log_every_steps: int = 10, use_ema: bool = True,
                 stream: Any = None, refresh_rate: float | None = None,
                 capture_output: bool | None = None, theme: Any = "default",
                 threaded: bool | None = None, tty: bool | None = None, width: int | None = None,
                 color: str | None = None, caps: TerminalCaps | None = None,
                 screen: str | None = None, clock: Any = None, **_ignored: Any) -> None:
        if options is not None:
            values = dict(options) if isinstance(options, dict) else dict(vars(options))
            columns = values.get("columns", columns)
            show_eta = values.get("show_eta", show_eta)
            show_mem = values.get("show_mem", show_mem)
            show_lr = values.get("show_lr", show_lr)
            precision = values.get("precision", precision)
            log_every_steps = values.get("log_every_steps", log_every_steps)
            use_ema = values.get("use_ema", use_ema)
        super().__init__(screen=screen, stream=stream, refresh_rate=refresh_rate,
                         capture_output=capture_output, theme=theme, threaded=threaded,
                         caps=caps)
        self.tty = tty
        self.width = width
        self.clock = clock
        self.color = color
        try:
            self.log_every_steps = max(1, int(log_every_steps or 10))
        except (TypeError, ValueError):
            self.log_every_steps = 10
        self.fmt = TrainingFormatter(columns=columns, show_eta=show_eta, show_mem=show_mem,
                                     show_lr=show_lr, precision=precision, use_ema=use_ema)
        self.options = SimpleNamespace(columns=columns, show_eta=show_eta, show_mem=show_mem,
                                       show_lr=show_lr, precision=precision,
                                       log_every_steps=self.log_every_steps, use_ema=use_ema)

    def make_state(self, source: str | None) -> TrainingTracker:
        kwargs = {"clock": self.clock} if self.clock is not None else {}
        return TrainingTracker(ema=0.9 if self.fmt.use_ema else 0.0, **kwargs)

    def prepare(self, source: str | None = None, context: Any = None) -> None:
        stream = self.stream if self.stream is not None else self.default_stream(source)
        if self.caps is None or getattr(self, "_derived_caps", False):
            derived = _options_caps(stream, self.tty, self.color)
            if derived is not None:
                self.caps = derived
                self._derived_caps = True
        self.fmt = TrainingFormatter(columns=self.fmt.columns, show_eta=self.fmt.show_eta,
                                     show_mem=self.fmt.show_mem, show_lr=self.fmt.show_lr,
                                     precision=self.fmt.precision, use_ema=self.fmt.use_ema)
        super().prepare(source, context)
        run_info = getattr(self.session, "run_info", None) or {}
        self.fmt.run_dir = run_info.get("run_dir") if isinstance(run_info, dict) else None

    def handle(self, event: VisualizationEvent) -> None:
        if getattr(event, "trial_id", None) is not None:
            return  # trial-scoped training events belong to experiment views
        super().handle(event)

    @property
    def plain(self) -> bool:
        return self.screen_type in ("plain", "null")

    def mem_text(self) -> str | None:
        if not self.fmt.show_mem:
            return None
        sysd = self.state.system()
        if sysd.get("device") in (None, "cpu") or sysd.get("gpu_mem_gb") is None:
            return None
        return fmt_bytes(float(sysd["gpu_mem_gb"]) * 2**30)


# ---------------------------------------------------------------- plain


class PlainTrainingVisualizer(_TrainingView):
    """Log-friendly ``key=value`` lines; never moves the cursor (``monitor="plain"``)."""

    owns_terminal = False
    screen = "plain"

    def on_train_start(self, event: VisualizationEvent) -> None:
        self.print(self.fmt.plain_header(self.state))

    def on_step(self, event: VisualizationEvent) -> None:
        step = self.state.step
        if step % self.log_every_steps == 0 or step <= 1:
            self.print(self.fmt.step_line(self.state))

    def on_epoch_end(self, event: VisualizationEvent) -> None:
        if self.state.epochs:
            self.print(self.fmt.plain_epoch_line(self.state.epochs[-1], self.state))

    def final(self, state: TrainingTracker) -> list[Line] | None:
        if state.status == "idle":
            return None
        return [self.fmt.done_line(state, plain=True)]


# ---------------------------------------------------------------- standard


class StandardTrainingVisualizer(_TrainingView):
    """Header, aligned epoch lines and one sticky live line (``monitor="standard"``)."""

    screen = "auto"
    live_screen = "inline"

    def on_train_start(self, event: VisualizationEvent) -> None:
        self.print(self.fmt.plain_header(self.state) if self.plain
                   else self.fmt.standard_header(self.state))

    def on_step(self, event: VisualizationEvent) -> None:
        if self.plain:
            step = self.state.step
            if step % self.log_every_steps == 0 or step <= 1:
                self.print(self.fmt.step_line(self.state))

    def on_epoch_end(self, event: VisualizationEvent) -> None:
        if not self.state.epochs:
            return
        row = self.state.epochs[-1]
        if self.plain:
            self.print(self.fmt.plain_epoch_line(row, self.state))
            return
        prev = self.state.epochs[-2] if len(self.state.epochs) > 1 else None
        text = self.fmt.epoch_line(row, prev, self.state)
        self.fmt.epoch_lines.append(text)
        if self._driver is not None:
            self._driver.print_now([text])

    def draw(self, state: TrainingTracker, width: int, height: int) -> list[Line]:
        if state.status in ("idle", "done", "failed"):
            return []
        return [self.fmt.live_line(state, width - 1, mem=self.mem_text())]

    def final(self, state: TrainingTracker) -> list[Line] | None:
        if state.status == "idle":
            return None
        return [self.fmt.done_line(state, plain=self.plain)]


# ---------------------------------------------------------------- dashboard panels


def _view(state: Any) -> Any:
    return getattr(state, "dashboard", None) or SimpleNamespace(
        charts=None, smoothing=0.6, log_y=False, scroll=0, system=None)


def _progress_lines(state: TrainingTracker, width: int) -> list[Line]:
    out = []
    ep_label = f"epoch {state.epoch + 1}" + (f"/{state.total_epochs}" if state.total_epochs else "")
    label_w = max(6, len(ep_label) + 1)
    bar_w = max(10, width - label_w - 26)
    total = state.total_steps
    frac = state.fraction
    count = f"{state.step:,} / {total:,} steps" if total else f"{state.step:,} steps"
    out.append(line(("total".ljust(label_w), "muted"), progress_bar(frac, bar_w), " ",
                    fmt_percent(frac), "  ", count))
    if state.steps_per_epoch:
        ef = state.epoch_fraction
        out.append(line((ep_label.ljust(label_w), "muted"), progress_bar(ef, bar_w), " ",
                        fmt_percent(ef), "  ",
                        f"{state.epoch_step:,} / {state.steps_per_epoch:,} steps"))
    rate = "validating…" if state.status == "validating" else (
        fmt_rate(state.throughput) if state.throughput else None)
    eta = state.eta
    cadence = _info(state, "eval_every")
    ckpt = _info(state, "checkpoint_dir")
    out.append(kv_line([("", rate), ("elapsed", fmt_clock(state.elapsed)),
                        ("ETA", fmt_clock(eta) if eta is not None else None),
                        ("validation every", f"{cadence} epoch{'s' if cadence != 1 else ''}"
                         if cadence else None),
                        ("checkpoint", ckpt)], width))
    return out


def _status_badge(state: TrainingTracker) -> Line:
    status = state.status
    if status == "done":
        b = badge("done", "good", G.completed)
    elif status == "failed":
        b = badge("failed", "bad", G.failed)
    elif status == "validating":
        b = badge("validating", "accent", "◌")
    elif status == "idle":
        b = badge("waiting", "muted", "◌")
    else:
        b = badge("training", "accent", "●")
    return line(b, "  ", (fmt_clock(state.elapsed), "muted"))


def _title(state: TrainingTracker) -> str:
    bits = [str(x) for x in (_info(state, "model_name"), _info(state, "recipe_name")) if x]
    return "topologies" + (" ─ " + " · ".join(bits) if bits else "")


@panel("header", title="topologies", min_height=3, priority=100, min_width=30)
def header_panel(state: TrainingTracker, width: int, height: int) -> list[Line]:
    return _progress_lines(state, width)


header_panel.title_right = _status_badge


@panel("progress", title="progress", min_height=4, priority=100, min_width=30)
def progress_panel(state: TrainingTracker, width: int, height: int) -> list[Line]:
    return _progress_lines(state, width)


def _chart_lines(series: dict[str, Any], width: int, height: int, *, styles: Any = None,
                 slots: Any = None, log_y: bool = False, background: Sequence[str] = (),
                 legend: bool = True) -> list[Line]:
    rows = height - 2 - (1 if legend else 0)
    if rows < 2:
        name = next((k for k in series if k not in background), next(iter(series)))
        ys = [p[1] for p in series[name]]
        return [line((name + " ", "muted"), sparkline(ys, max(1, width - len(name) - 1)))]
    return line_chart(series, width, rows, log_y=log_y, styles=styles, slots=slots,
                      legend=legend, direct_labels=True, background=background)[:height]


def _smoothed(values: Sequence[float], weight: float) -> list[float]:
    if not weight:
        return list(values)
    out, acc = [], None
    for v in values:
        acc = v if acc is None else weight * acc + (1 - weight) * v
        out.append(acc)
    return out


@panel("loss", title="loss · x = step", min_height=8, preferred_height=12, priority=90,
       min_width=30)
def loss_panel(state: TrainingTracker, width: int, height: int) -> list[Line]:
    view = _view(state)
    key = loss_key(state)
    if key is None:
        return [styled("waiting for the first step…", "muted")]
    raw = state.series(key)
    series: dict[str, Any] = {}
    styles: dict[str, Any] = {}
    weight = view.smoothing or 0.0
    budget = max(8, 2 * width)
    if weight and len(raw) > 2:
        series["raw"] = raw.thin(budget)
        styles["raw"] = "muted"
        smooth = _smoothed(raw.y, weight)
        stride = max(1, len(smooth) // (4 * budget))
        series["train"] = list(zip(raw.x[::stride], smooth[::stride]))
    else:
        series["train"] = raw.thin(budget)
    vk = val_key(state)
    if vk is not None and "loss" in vk:
        val = state.series(vk)
        if val:
            series["val"] = list(zip(val.x, val.y))
    return _chart_lines(series, width, height, styles=styles, background=("raw",),
                        slots={"train": 0, "val": 1}, log_y=view.log_y)


def _chart_keys(state: TrainingTracker) -> list[str]:
    view = _view(state)
    if view.charts:
        return list(view.charts)
    keys = [k for k in state.metric_names() if not k.startswith("_")]
    order = {"loss": 0, "train_loss": 0}
    return sorted(keys, key=lambda k: (order.get(k, 1), k.startswith("val"), _is_lr(k)))


@panel("metrics", title="metrics", min_height=5, preferred_height=10, priority=80,
       min_width=36)
def metrics_panel(state: TrainingTracker, width: int, height: int) -> list[Line]:
    keys = _chart_keys(state)
    rows = []
    trend_w = max(4, min(16, width - 38))
    for key in keys[: max(1, height - 2)]:
        latest = state.latest_value(key)
        best = state.best(key)
        last_text = fmt_metric(key, latest) if latest is not None else None
        best_text = None
        if best is not None and not _is_lr(key):
            best_text = fmt_metric(key, best[0])
        is_best = best is not None and latest is not None and best[0] == latest and \
            not _is_lr(key) and len(state.series(key)) > 1
        last_cell = line(last_text or "—", (" " + G.best, "best") if is_best else "  ")
        trend = sparkline(state.series(key).y, trend_w)
        rows.append([key, last_cell, best_text, trend])
    return table(["metric", "last", "best", "trend"], rows, width=width, hide_empty=False,
                 indent=0)


@panel("epochs", title="epochs", min_height=5, preferred_height=8, priority=60, min_width=30)
def epochs_panel(state: TrainingTracker, width: int, height: int) -> list[Line]:
    if not state.epochs:
        return [styled("no epoch finished yet", "muted")]
    view = _view(state)
    names: list[str] = []
    for row in state.epochs:
        for n in row.metrics:
            if n not in names:
                names.append(n)
    first_seen = list(names)
    names.sort(key=lambda n: ("loss" not in n, first_seen.index(n)))
    room = max(1, height - 2)
    scroll = max(0, min(int(getattr(view, "scroll", 0) or 0), len(state.epochs) - room))
    end = len(state.epochs) - scroll
    shown = state.epochs[max(0, end - room): end]
    rows = []
    for row in shown:
        rows.append([row.epoch + 1, *[row.metrics.get(n) for n in names],
                     fmt_clock(row.duration) if row.duration is not None else None,
                     (G.best, "best") if row.is_best else ""])
    return table(["ep", *names, "time", ""], rows, width=width, indent=0, hide_empty=False)


@panel("system", title="system", min_height=5, preferred_height=6, priority=50, min_width=30)
def system_panel(state: TrainingTracker, width: int, height: int) -> list[Line]:
    view = _view(state)
    sysd = view.system if getattr(view, "system", None) else state.system()
    device = " · ".join(str(x) for x in (_info(state, "device") or sysd.get("device"),
                                         _info(state, "precision")) if x)
    params = _info(state, "param_total")
    out = [kv_line([("device", device or "n/a"),
                    ("params", fmt_count(params) if params is not None else None)], width)]
    cpu = sysd.get("cpu_percent")
    out.append(meter("cpu".ljust(7), None if cpu is None else float(cpu) / 100.0,
                     min(width, 40)))
    rss = sysd.get("rss_gb")
    accel = sysd.get("gpu_mem_gb")
    out.append(kv_line([("rss", "n/a" if rss is None else fmt_bytes(float(rss) * 2**30)),
                        ("accel mem", None if sysd.get("device") in (None, "cpu") else
                         ("n/a" if accel is None else fmt_bytes(float(accel) * 2**30))),
                        ("gpu util", None if sysd.get("gpu_util") is None else
                         f"{sysd['gpu_util']:.0f}%")], width))
    if state.best_metric is not None:
        name = state.monitored or "best"
        out.append(kv_line([("best", f"{name} {fmt_number(state.best_metric)}"
                             + (f" @ epoch {state.best_epoch + 1}"
                                if state.best_epoch is not None else ""))], width))
    return out


@panel("log", title="log", min_height=4, preferred_height=6, priority=40, min_width=30)
def log_panel(state: TrainingTracker, width: int, height: int) -> list[Line]:
    view = _view(state)
    entries = list(state.log)
    scroll = int(getattr(view, "log_scroll", 0) or 0)
    if scroll:
        entries = entries[: max(0, len(entries) - scroll)]
    out = []
    for _t, level, text in entries[-max(1, height):]:
        style = {"error": "bad", "warn": "warn", "out": None}.get(level, "muted")
        out.append(styled(text, style))
    return out or [styled("no events yet", "muted")]


def _footer(state: TrainingTracker, width: int, hints: str, message: str | None) -> Line:
    left = message
    if left is None:
        act = getattr(state, "activity", None)
        parts = [plan_text(state)]
        if act:
            total = act.get("total")
            parts.append(f"{act['label']} {act.get('done') or 0}" + (f"/{total}" if total else ""))
        left = " · ".join(p for p in parts if p)
    if not left:
        notable = [t for _ts, lvl, t in state.log if lvl != "out"]
        left = notable[-1] if notable else ""
    right = styled(hints, "key")
    room = max(0, width - right.width - 2)
    return line(" ", styled(left, "warn" if message else "muted").clip(room).pad(room), " ",
                right).clip(width)


@panel("footer", title="", min_height=1, priority=100, border=False)
def footer_panel(state: TrainingTracker, width: int, height: int) -> list[Line]:
    return [_footer(state, width, "q quit · tab view · ? help", None)]


def _families(state: TrainingTracker) -> list[tuple[str, list[str], bool]]:
    groups: dict[str, list[str]] = {"loss": [], "accuracy": [], "learning rate": [],
                                    "grad norm": [], "other": []}
    for key in state.metric_names():
        if "loss" in key:
            groups["loss"].append(key)
        elif _is_lr(key):
            groups["learning rate"].append(key)
        elif "grad" in key:
            groups["grad norm"].append(key)
        else:
            series = state.series(key)
            if series and direction(key) == "max" and 0 <= min(series.y) and max(series.y) <= 1:
                groups["accuracy"].append(key)
            else:
                groups["other"].append(key)
    return [(name, keys, name == "learning rate") for name, keys in groups.items() if keys]


def _family_panel(name: str, keys: list[str], log_y: bool) -> Panel:
    def render(state: TrainingTracker, width: int, height: int) -> list[Line]:
        view = _view(state)
        series = {}
        slots = {}
        for key in keys[:8]:
            s = state.series(key)
            if not s:
                continue
            label = key
            if s.split == "train" and view.smoothing:
                ys = _smoothed(s.y, view.smoothing)
                stride = max(1, len(ys) // (8 * width))
                series[label] = list(zip(s.x[::stride], ys[::stride]))
            else:
                series[label] = s.thin(max(8, 2 * width))
            slots[label] = 1 if s.split == "val" else (0 if len(slots) == 0 else len(slots))
        if not series:
            return [styled("no data", "muted")]
        return _chart_lines(series, width, height, slots=slots, log_y=log_y or view.log_y)

    return Panel(f"family:{name}", render, title=f"{name} · x = step", min_height=8,
                 preferred_height=12,
                 priority=50, min_width=30)


def _help_lines(controls: bool) -> list[Line]:
    keys = [("q", "leave fullscreen, keep training with the standard line"),
            ("?", "toggle this help"), ("tab / 1-4", "switch view: overview, metrics, log, system"),
            ("↑ ↓", "scroll epochs / log"), ("l", "toggle log scale on charts"),
            ("s", "cycle smoothing (off, EMA 0.6, EMA 0.9)"), ("p", "pause / resume drawing")]
    if controls:
        keys.append(("S", "request a graceful stop (asks for confirmation)"))
    return [line((k.ljust(10), "key"), " ", d) for k, d in keys]


def compose(blocks: Sequence[tuple[Rect, Sequence[Line]]], cols: int, rows: int) -> list[Line]:
    """Paint rectangular blocks of lines into a ``rows`` × ``cols`` frame."""
    grid: list[list[tuple[int, Line]]] = [[] for _ in range(rows)]
    for rect, lines in blocks:
        for i in range(min(rect.h, len(lines))):
            y = rect.y + i
            if 0 <= y < rows:
                grid[y].append((rect.x, as_line(lines[i]).clip(rect.w).pad(rect.w)))
    out = []
    for parts in grid:
        parts.sort(key=lambda p: p[0])
        pieces: list[Any] = []
        x = 0
        for px, ln in parts:
            if px > x:
                pieces.append(" " * (px - x))
                x = px
            pieces.append(ln)
            x += ln.width
        out.append(line(*pieces).clip(cols))
    return out


def activity_line(state: TrainingTracker, width: int) -> Line | None:
    """Progress of work outside the step loop (``monitor.note``), or ``None``."""
    act = getattr(state, "activity", None)
    if not act:
        return None
    done, total = int(act.get("done") or 0), act.get("total")
    frac = done / max(1, int(total)) if total else None
    count = f"{done}/{int(total)}" if total else str(done)
    status = "  done" if total and done >= int(total) else ""
    rate = act.get("rate")
    speed = f"  {rate:.1f} it/s" if rate else ""
    bar_w = max(10, min(40, width - 60))
    return line((str(act["label"]), "label"), "  ", progress_bar(frac, bar_w), " ",
                fmt_percent(frac), "  ", count, status, (speed, "muted")).clip(width)


def plan_text(state: TrainingTracker) -> str | None:
    """``trial i/n · d/n done`` when the run is one trial of an experiment plan."""
    plan = getattr(state, "plan", None)
    if not plan:
        return None
    index, total, done = plan
    done = max(0, index - 1) if done is None else done
    return f"trial {index}/{total} · {done}/{total} done"


def compact_frame(state: TrainingTracker, width: int, *, height: int | None = None,
                  charts: Sequence[str] | None = None, system: Any = None) -> list[Line]:
    """One-column dashboard (inline fallback, non-TTY snapshots, ``render_dashboard``)."""
    width = max(40, int(width))
    view = SimpleNamespace(charts=list(charts) if charts else None, smoothing=0.6, log_y=False,
                           scroll=0, system=system)
    old = getattr(state, "dashboard", None)
    state.dashboard = view
    try:
        out: list[Line] = []
        title = " · ".join(str(x) for x in (_info(state, "model_name"),
                                            _info(state, "recipe_name")) if x)
        head = line(("topologies", "title"), (" · " + title) if title else "")
        badge_line = _status_badge(state)
        room = width - badge_line.width - 1
        out.append(line(head.clip(max(0, room)).pad(max(0, room)), " ", badge_line).clip(width))
        frac = state.fraction
        epoch = f"Epoch {state.epoch + 1}" + (f"/{state.total_epochs}"
                                              if state.total_epochs else "")
        steps = f"step {state.step:,}" + (f"/{state.total_steps:,}" if state.total_steps else "")
        eta = state.eta
        status = "done" if state.status == "done" else (
            fmt_clock(state.elapsed) + (f" < {fmt_clock(eta)}" if eta is not None else ""))
        rate = fmt_rate(state.throughput) if state.throughput else None
        bar_w = max(10, min(40, width - 60))
        out.append(line((epoch, "label"), "  ", progress_bar(frac, bar_w), " ",
                        fmt_percent(frac), "  ", steps, "  ", status,
                        ("  " + rate) if rate else "").clip(width))
        activity = activity_line(state, width)
        if activity is not None:
            out.append(activity)
        sections = [("metrics", metrics_panel(state, width - 1, 12)),
                    ("epochs", epochs_panel(state, width - 1, 6) if state.epochs else []),
                    ("system", system_panel(state, width - 1, 4))]
        for name, body in sections:
            if not body:
                continue
            out.append(rule(width, name))
            out.extend(line(" ", b).clip(width) for b in body)
        plan = plan_text(state)
        if plan:
            out.append(rule(width, "plan"))
            out.append(line(plan).clip(width))
        if height is not None and len(out) > height:
            out = out[: max(1, height)]
        return out
    finally:
        if old is None:
            with contextlib.suppress(AttributeError):
                del state.dashboard
        else:
            state.dashboard = old


# ---------------------------------------------------------------- dashboard view


DEFAULT_PANELS = ("header", "loss", "metrics", "epochs", "system", "log", "footer")
VIEWS = ("overview", "metrics", "log", "system")


class TrainingDashboard(_TrainingView):
    """Fullscreen training dashboard (``monitor="dashboard"``).

    Options (also via ``monitoring={...}``): ``fullscreen`` (None = auto), ``panels``,
    ``view``, ``interactive``, ``controls``, ``capture_output``, ``theme``,
    ``smoothing``, ``charts``, ``refresh_rate``, ``width``, ``tty``, ``stream``.
    """

    screen = "auto"
    live_screen = "fullscreen"

    def __init__(self, refresh_rate: float | None = None, charts: Sequence[str] | None = None,
                 *, fullscreen: bool | None = None, panels: Sequence[Any] | None = None,
                 view: str = "overview", interactive: bool = True, controls: bool = False,
                 smoothing: float = 0.6, **options: Any) -> None:
        super().__init__(refresh_rate=refresh_rate, **options)
        self.charts = list(charts) if charts else []
        self.fullscreen = fullscreen
        if fullscreen is False:
            self.live_screen = "inline"
        self.panels = resolve_panels(panels, "training", DEFAULT_PANELS)
        self.view_name = view if view in VIEWS else "overview"
        self.interactive = interactive
        self.controls = controls
        self.smoothing = float(smoothing or 0.0)
        self.log_y = False
        self.help = False
        self.message: str | None = None
        self._confirm_stop = False
        self._keys: Any = None
        self._standard_after_quit = False
        self._snapshots = 0

    # -- session -----------------------------------------------------------------
    def screen_kind(self, caps: TerminalCaps) -> str:
        if self.fullscreen is True and caps.is_tty and not self._released:
            return "fullscreen"
        return super().screen_kind(caps)

    def on_session_start(self) -> None:
        self.state.dashboard = SimpleNamespace(
            charts=self.charts or None, smoothing=self.smoothing, log_y=False, scroll=0,
            log_scroll=0, system=None)
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

    # -- events ------------------------------------------------------------------
    def on_epoch_end(self, event: VisualizationEvent) -> None:
        if not self.state.epochs:
            return
        row = self.state.epochs[-1]
        prev = self.state.epochs[-2] if len(self.state.epochs) > 1 else None
        text = self.fmt.epoch_line(row, prev, self.state)
        self.fmt.epoch_lines.append(text)
        vk = self.state.monitored
        bits = [f"epoch {row.epoch + 1} done"]
        if vk and vk in row.metrics:
            bits.append(f"{vk} {fmt_number(row.metrics[vk])}" + (f" {G.best}" if row.is_best
                                                                  else ""))
        if row.checkpoint:
            bits.append(f"saved {self.fmt._checkpoint(row.checkpoint, full=False)}")
        self.state.add_log(" · ".join(bits), "event")
        if self._standard_after_quit and self._driver is not None:
            self._driver.print_now([text])
        elif self.plain:
            self.print("")
            for ln in compact_frame(self.state, self.width or 100, charts=self.charts or None):
                self.print(ln)
            self._snapshots += 1

    # -- keys --------------------------------------------------------------------
    def on_key(self, key: str) -> None:
        view = self.state.dashboard
        if self._confirm_stop:
            self._confirm_stop = False
            if key == "S" and self.session is not None and self.session.request_stop:
                self.session.request_stop("dashboard")
                self.message = "stop requested: finishing the current epoch…"
            else:
                self.message = "stop cancelled"
        elif key == "q":
            self._leave_fullscreen()
            return
        elif key == "?":
            self.help = not self.help
        elif key == "tab":
            self.view_name = VIEWS[(VIEWS.index(self.view_name) + 1) % len(VIEWS)]
        elif key in ("1", "2", "3", "4"):
            self.view_name = VIEWS[int(key) - 1]
        elif key == "l":
            self.log_y = not self.log_y
            view.log_y = self.log_y
        elif key == "s":
            i = _SMOOTHING_STEPS.index(view.smoothing) if view.smoothing in _SMOOTHING_STEPS \
                else 0
            view.smoothing = _SMOOTHING_STEPS[(i + 1) % len(_SMOOTHING_STEPS)]
            self.message = f"smoothing {'off' if not view.smoothing else f'EMA {view.smoothing}'}"
        elif key == "p":
            if self._driver is not None:
                self._driver.paused = not self._driver.paused
                self.message = "paused (data still collected)" if self._driver.paused else None
        elif key == "up":
            view.scroll = int(view.scroll or 0) + 1
            view.log_scroll = int(view.log_scroll or 0) + 1
        elif key == "down":
            view.scroll = max(0, int(view.scroll or 0) - 1)
            view.log_scroll = max(0, int(view.log_scroll or 0) - 1)
        elif key == "S":
            if self.controls and self.session is not None and self.session.request_stop:
                self._confirm_stop = True
                self.message = "press S again to stop training (any other key cancels)"
        if self._driver is not None:
            self._driver.refresh()

    def _leave_fullscreen(self) -> None:
        """``q``: leave the alternate screen and continue with the standard live line."""
        if self.screen_type != "fullscreen" or self._driver is None:
            return
        from topologies.visualization.terminal.driver import LiveDriver
        from topologies.visualization.terminal.screen import InlineRegion

        if self._keys is not None:
            self._keys.stop()
            self._keys = None
        old_driver, old_screen = self._driver, self._screen
        old_driver.stop(list(self.fmt.epoch_lines), render_final=False)
        self._standard_after_quit = True
        self._screen = InlineRegion(old_screen.stream, caps=old_screen.caps, theme=self.theme)
        self._driver = LiveDriver(self._screen, self._render_frame, fps=10.0,
                                  threaded=self.threaded)
        self._driver.start()
        if self._capture is not None:
            self._capture.sink = self._driver.print

    # -- rendering -----------------------------------------------------------------
    def draw(self, state: TrainingTracker, width: int, height: int) -> list[Line]:
        if self._standard_after_quit:
            if state.status in ("done", "failed"):
                return []
            return [self.fmt.live_line(state, width - 1, mem=self.mem_text())]
        kind = self.screen_type
        if kind == "plain":
            return []
        if kind == "inline":
            return compact_frame(state, width - 1, height=max(1, height - 1),
                                 charts=self.charts or None)
        return self.frame(state, width, height)

    def frame(self, state: TrainingTracker, width: int, height: int) -> list[Line]:
        """Full fullscreen frame of exactly ``height`` rows."""
        if width < 60 or height < 12:
            frac = state.fraction
            return [line(("topologies", "title"), "  ", _status_badge(state)).clip(width),
                    line(progress_bar(frac, max(4, width - 8)), fmt_percent(frac)).clip(width),
                    styled("enlarge terminal for the full dashboard", "muted").clip(width)]
        names = {p.name for p in self.panels}
        header = next((p for p in self.panels if p.name == "header"), None)
        footer = next((p for p in self.panels if p.name == "footer"), None)
        blocks: list[tuple[Rect, Sequence[Line]]] = []
        top = 0
        if header is not None:
            body = header_panel(state, width - 4, 3) if "header" in names else []
            hdr = box(body, _title(state), width, len(body) + 2, title_right=_status_badge(state))
            blocks.append((Rect(0, 0, width, len(hdr)), hdr))
            top = len(hdr)
        bottom = height - (1 if footer is not None else 0)
        area = Rect(0, top, width, max(0, bottom - top))
        if self.help:
            help_box = box(_help_lines(self.controls), "keys", min(width, 70),
                           min(area.h, len(_help_lines(self.controls)) + 2))
            blocks.append((Rect(0, area.y, min(width, 70), len(help_box)), help_box))
        else:
            for panel_obj, rect in grid_layout(self._body_panels(state), area):
                blocks.append((rect, panel_obj.render(state, rect.w, rect.h)))
        if footer is not None:
            hints = "q quit · tab view · ? help" + (" · S stop" if self.controls else "")
            label = f"[{self.view_name}] "
            blocks.append((Rect(0, height - 1, width - 1, 1),
                           [_footer(state, width - 1, label + hints, self.message)]))
        return compose(blocks, width, height)

    def _body_panels(self, state: TrainingTracker) -> list[Panel]:
        if self.view_name == "metrics":
            families = _families(state)
            return [_family_panel(n, k, log) for n, k, log in families] or \
                [p for p in self.panels if p.name == "metrics"]
        if self.view_name == "log":
            return [Panel("log", log_panel, title="log", min_height=3, priority=100,
                          preferred_height=200)]
        if self.view_name == "system":
            return [Panel("system", system_panel, title="system", min_height=4, priority=100),
                    Panel("run", _run_info_panel, title="run info", min_height=4, priority=90,
                          preferred_height=20)]
        return [p for p in self.panels if p.name not in ("header", "progress", "footer")]

    def final(self, state: TrainingTracker) -> list[Line] | None:
        if state.status == "idle":
            return None
        if self.plain:
            if self._snapshots and state.status == "done":
                return [self.fmt.done_line(state, plain=False)]
            out = [Line()] + compact_frame(state, self.width or 100,
                                           charts=self.charts or None)
            return [*out, self.fmt.done_line(state, plain=False)]
        if self._standard_after_quit:
            return [self.fmt.done_line(state, plain=False)]
        return [*self.fmt.epoch_lines, self.fmt.done_line(state, plain=False)]

    def snapshot(self, width: int = 100) -> list[str]:
        """Current compact frame as plain text lines (also handy in notebooks/tests)."""
        from topologies.visualization.terminal.text import to_plain

        with self.state.lock:
            return [to_plain(ln) for ln in compact_frame(self.state, width,
                                                         charts=self.charts or None)]


def _run_info_panel(state: TrainingTracker, width: int, height: int) -> list[Line]:
    run = getattr(state, "run", None)
    if run is None:
        return [styled("no run info", "muted")]
    rows = []
    for key, value in vars(run).items():
        if value in (None, "", 0) and key not in ("seed",):
            continue
        rows.append([key, fmt_lr(value) if key == "lr" else str(value)])
    return table(["field", "value"], rows[: max(1, height - 2)], width=width, indent=0)


# ---------------------------------------------------------------- resolver


def _monitor_options(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    try:
        return {k: v for k, v in cfg.model_dump().items() if v is not None}
    except AttributeError:
        return {k: v for k, v in vars(cfg).items() if v is not None}


def view_options(mode: str, options: dict[str, Any]) -> dict[str, Any]:
    """Filter producer options to what the selected tier accepts."""
    common = {"columns", "show_eta", "show_mem", "show_lr", "precision", "log_every_steps",
              "use_ema", "stream", "theme", "capture_output", "tty", "width", "color",
              "refresh_rate", "threaded"}
    dashboard = {"charts", "fullscreen", "panels", "view", "interactive", "controls",
                 "smoothing"}
    keep = common | (dashboard if mode == "dashboard" else set())
    out = {k: v for k, v in options.items() if k in keep and v is not None}
    if out.get("color") == "auto":
        out.pop("color")
    return out


def make_training_view(mode: str, options: dict[str, Any] | None = None) -> _TrainingView:
    """Construct the view for a ``monitor=`` mode (``plain``/``compact``/``standard``/...)."""
    mode = (mode or "standard").strip().lower()
    opts = view_options("dashboard" if mode == "dashboard" else "standard", dict(options or {}))
    if mode in ("plain", "compact"):
        return PlainTrainingVisualizer(**opts)
    if mode == "dashboard":
        return TrainingDashboard(**opts)
    return StandardTrainingVisualizer(**opts)


def resolve_training_visualizers(monitor: Any, monitoring_cfg: Any, selection: Any = None, *,
                                 request_stop: Any = None, log_every_steps: int = 10,
                                 selection_given: bool = False) -> tuple[Any, Any]:
    """``(monitor facade, VisualizerGroup)`` for a Trainer's display arguments.

    ``selection`` (``visualizer=``/``visualizers=``) wins when ``selection_given``;
    otherwise the ``monitor``/``monitoring`` tier is used. ``request_stop`` reaches
    the views through :class:`SessionInfo` at prepare time.
    """
    from topologies.training.monitoring import BaseMonitor, build_monitor
    from topologies.visualization.core import LegacyMonitorVisualizer, create_visualizer

    del request_stop
    options = _monitor_options(monitoring_cfg)
    mode = monitor if isinstance(monitor, str) else options.pop("mode", None) or "standard"
    options.pop("mode", None)
    options.setdefault("log_every_steps", log_every_steps)
    if isinstance(monitor, BaseMonitor) and not hasattr(monitor, "visualizer"):
        facade = monitor
    else:
        facade = monitor if isinstance(monitor, BaseMonitor) else build_monitor(mode, options)
    if selection_given:
        return facade, create_visualizer(False if selection is None else selection)
    visualizer = getattr(facade, "visualizer", None)
    if visualizer is not None:
        return facade, create_visualizer([visualizer])
    return facade, create_visualizer([LegacyMonitorVisualizer(facade)])


__all__ = [
    "DEFAULT_PANELS",
    "PlainTrainingVisualizer",
    "StandardTrainingVisualizer",
    "TrainingDashboard",
    "TrainingFormatter",
    "compact_frame",
    "compose",
    "make_training_view",
    "resolve_training_visualizers",
    "view_options",
]

