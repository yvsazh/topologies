"""Public testing helpers for visualizers.

* :class:`VirtualTerminal` — a small VT emulator for the sequences this library emits,
  including resize with optional reflow of soft-wrapped lines.
* :class:`RecordingVisualizer` — stores every event and rendered result.
* :func:`simulate_training` / :func:`simulate_experiment` — drive any visualizer through
  the real event path with realistic synthetic data (no Torch needed).
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from topologies.visualization.core import Visualizer
from topologies.visualization.terminal.text import char_width

_CSI = re.compile(r"\x1b\[([?>=]?)([0-9;]*)([ -/]*)([@-~])")
_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


class VirtualTerminal:
    """Emulate a terminal of ``cols`` × ``rows`` fed with the bytes a program writes.

    Supports printable text (wide-char aware), CR, LF (``onlcr``: LF also returns the
    carriage, like a tty's output processing), BS, CUP/HVP, CUU/CUD/CUF/CUB, CNL/CPL,
    CHA, EL, ED, SGR (tracked, not rendered), DECAWM (``?7``), DECTCEM (``?25``),
    alternate screen (``?1049``), synchronized updates (``?2026``), DECSC/DECRC,
    autowrap with soft-wrap tracking and scrolling into a scrollback buffer.
    :meth:`resize` re-wraps soft-wrapped lines when ``reflow`` is true.
    """

    def __init__(self, cols: int = 80, rows: int = 24, *, reflow: bool = True,
                 onlcr: bool = True) -> None:
        self.cols, self.rows = int(cols), int(rows)
        self.reflow = reflow
        self.onlcr = onlcr
        self.autowrap = True
        self.cursor_visible = True
        self.alt = False
        self.sync_depth = 0
        self.sync_frames = 0
        self.sgr: list[str] = []
        self.scrollback: list[tuple[list[str], bool]] = []
        self._main = self._blank_grid()
        self._main_soft = [False] * self.rows
        self._alt = self._blank_grid()
        self._alt_soft = [False] * self.rows
        self.row = 0
        self.col = 0
        self._saved = (0, 0)
        self._alt_saved = (0, 0)
        self._pending = ""
        self.writes = 0

    # -- buffers ---------------------------------------------------------------
    def _blank_row(self, cols: int | None = None) -> list[str]:
        return [" "] * (self.cols if cols is None else cols)

    def _blank_grid(self) -> list[list[str]]:
        return [self._blank_row() for _ in range(self.rows)]

    @property
    def grid(self) -> list[list[str]]:
        return self._alt if self.alt else self._main

    @property
    def soft(self) -> list[bool]:
        return self._alt_soft if self.alt else self._main_soft

    @property
    def cursor(self) -> tuple[int, int]:
        return self.row, min(self.col, self.cols - 1)

    # -- feeding -----------------------------------------------------------------
    def feed(self, data: str | bytes) -> None:
        if isinstance(data, bytes):
            data = data.decode("utf-8", "replace")
        self.writes += 1
        data = self._pending + data
        self._pending = ""
        i = 0
        n = len(data)
        while i < n:
            ch = data[i]
            if ch == "\x1b":
                if i + 1 >= n:
                    self._pending = data[i:]
                    return
                nxt = data[i + 1]
                if nxt == "[":
                    m = _CSI.match(data, i)
                    if m is None:
                        if not re.search(r"[@-~]", data[i + 2:]):
                            self._pending = data[i:]
                            return
                        i += 2
                        continue
                    self._csi(m.group(1), m.group(2), m.group(4))
                    i = m.end()
                    continue
                if nxt == "]":
                    m = _OSC.match(data, i)
                    if m is None:
                        self._pending = data[i:]
                        return
                    i = m.end()
                    continue
                if nxt == "7":
                    self._saved = (self.row, self.col)
                elif nxt == "8":
                    self.row, self.col = self._saved
                i += 2
                continue
            if ch == "\r":
                self.col = 0
            elif ch == "\n":
                self._linefeed()
                if self.onlcr:
                    self.col = 0
            elif ch == "\b":
                self.col = max(0, min(self.col, self.cols - 1) - 1)
            elif ch == "\t":
                self.col = min(self.cols - 1, (self.col // 8 + 1) * 8)
            elif ch >= " " or ord(ch) > 0x7F:
                self._put(ch)
            i += 1

    def _linefeed(self) -> None:
        if self.row >= self.rows - 1:
            top = self.grid.pop(0)
            soft = self.soft.pop(0)
            if not self.alt:
                self.scrollback.append((top, soft))
            self.grid.append(self._blank_row())
            self.soft.append(False)
            self.row = self.rows - 1
        else:
            self.row += 1

    def _wrap(self) -> None:
        self.soft[self.row] = True
        self._linefeed()
        self.col = 0

    def _put(self, ch: str) -> None:
        w = char_width(ch)
        if w == 0:
            c = max(0, min(self.col, self.cols) - 1)
            self.grid[self.row][c] += ch
            return
        if self.col >= self.cols:
            if self.autowrap:
                self._wrap()
            else:
                self.col = self.cols - 1
        if w == 2 and self.col == self.cols - 1:
            if self.autowrap:
                self.grid[self.row][self.col] = " "
                self._wrap()
            else:
                return
        row = self.grid[self.row]
        row[self.col] = ch
        if w == 2:
            row[self.col + 1] = ""
        self.col += w
        if self.col >= self.cols and not self.autowrap:
            self.col = self.cols - 1

    def _clear_cells(self, r: int, start: int, end: int) -> None:
        row = self.grid[r]
        for c in range(max(0, start), min(self.cols, end)):
            row[c] = " "

    def _csi(self, private: str, params: str, final: str) -> None:
        nums = [int(p) if p else 0 for p in params.split(";")] if params else []

        def arg(k: int, default: int = 1) -> int:
            return nums[k] if len(nums) > k and nums[k] else default

        if private == "?":
            if final in "hl":
                on = final == "h"
                for code in nums:
                    self._mode(code, on)
            return
        col = min(self.col, self.cols - 1)
        if final == "m":
            self.sgr.append(params)
        elif final == "A":
            self.row = max(0, self.row - arg(0))
            self.col = col
        elif final == "B":
            self.row = min(self.rows - 1, self.row + arg(0))
            self.col = col
        elif final == "C":
            self.col = min(self.cols - 1, col + arg(0))
        elif final == "D":
            self.col = max(0, col - arg(0))
        elif final == "E":
            self.row = min(self.rows - 1, self.row + arg(0))
            self.col = 0
        elif final == "F":
            self.row = max(0, self.row - arg(0))
            self.col = 0
        elif final == "G":
            self.col = min(self.cols - 1, arg(0) - 1)
        elif final in "Hf":
            self.row = min(self.rows - 1, max(0, arg(0) - 1))
            self.col = min(self.cols - 1, max(0, arg(1) - 1))
        elif final == "J":
            mode = nums[0] if nums else 0
            if mode == 0:
                self._clear_cells(self.row, col, self.cols)
                self.soft[self.row] = False
                for r in range(self.row + 1, self.rows):
                    self.grid[r] = self._blank_row()
                    self.soft[r] = False
            elif mode == 1:
                for r in range(self.row):
                    self.grid[r] = self._blank_row()
                    self.soft[r] = False
                self._clear_cells(self.row, 0, col + 1)
            elif mode == 2:
                for r in range(self.rows):
                    self.grid[r] = self._blank_row()
                    self.soft[r] = False
            elif mode == 3:
                self.scrollback.clear()
        elif final == "K":
            mode = nums[0] if nums else 0
            if mode == 0:
                self._clear_cells(self.row, col, self.cols)
                self.soft[self.row] = False
            elif mode == 1:
                self._clear_cells(self.row, 0, col + 1)
            elif mode == 2:
                self._clear_cells(self.row, 0, self.cols)
                self.soft[self.row] = False

    def _mode(self, code: int, on: bool) -> None:
        if code == 7:
            self.autowrap = on
        elif code == 25:
            self.cursor_visible = on
        elif code == 2026:
            if on:
                self.sync_depth += 1
            else:
                if self.sync_depth:
                    self.sync_frames += 1
                self.sync_depth = 0
        elif code in (1049, 1047, 47):
            if on and not self.alt:
                self._alt_saved = (self.row, self.col)
                self.alt = True
                self._alt = self._blank_grid()
                self._alt_soft = [False] * self.rows
            elif not on and self.alt:
                self.alt = False
                self.row, self.col = self._alt_saved

    # -- resize ------------------------------------------------------------------
    def resize(self, cols: int, rows: int) -> None:
        """Change the size; reflow soft-wrapped lines of the main screen when enabled."""
        cols, rows = max(1, int(cols)), max(1, int(rows))
        alt_active = self.alt
        if alt_active:
            self._alt = [(r + [" "] * cols)[:cols] for r in self._alt[:rows]]
            self._alt += [[" "] * cols for _ in range(rows - len(self._alt))]
            self._alt_soft = [False] * rows
            alt_cursor = (min(self.row, rows - 1), min(self.col, cols - 1))
            self.row, self.col = self._alt_saved
        if self.reflow:
            self._reflow_main(cols, rows)
        else:
            self._crop_main(cols, rows)
        self.cols, self.rows = cols, rows
        if alt_active:
            self._alt_saved = (self.row, self.col)
            self.row, self.col = alt_cursor

    def _crop_main(self, cols: int, rows: int) -> None:
        grid = [(r + [" "] * cols)[:cols] for r in self._main]
        soft = [False] * len(grid)
        if rows < len(grid):
            spill = max(0, self.row - rows + 1)
            for r in range(spill):
                self.scrollback.append((grid[r], False))
            grid, soft = grid[spill:spill + rows], soft[spill:spill + rows]
            self.row -= spill
        while len(grid) < rows:
            grid.append([" "] * cols)
            soft.append(False)
        self._main, self._main_soft = grid, soft
        self.col = min(self.col, cols)

    def _reflow_main(self, cols: int, rows: int) -> None:
        physical = [*self.scrollback, *zip(self._main, self._main_soft)]
        cursor_g = len(self.scrollback) + self.row
        last_content = max((i for i, (cells, _s) in enumerate(physical)
                            if "".join(cells).strip()), default=-1)
        keep_to = max(cursor_g, last_content)
        physical = physical[: keep_to + 1]
        logical: list[list[str]] = []
        cursor_logical = (0, 0)
        current: list[str] = []
        for g, (cells, soft) in enumerate(physical):
            if g == cursor_g:
                cursor_logical = (len(logical), len(current) + min(self.col, len(cells)))
            current = current + list(cells)
            if not soft:
                logical.append(current)
                current = []
        if current:
            logical.append(current)
        new_rows: list[tuple[list[str], bool]] = []
        cursor_new = 0
        cursor_col = 0
        for li, cells in enumerate(logical):
            trimmed = list(cells)
            while trimmed and trimmed[-1] == " ":
                trimmed.pop()
            chunks: list[list[str]] = []
            cur: list[str] = []
            for cell in trimmed:
                if cell == "":
                    continue
                w = char_width(cell[0]) if cell else 1
                if len(cur) + w > cols and cur:
                    chunks.append(cur)
                    cur = []
                cur.append(cell)
                if w == 2:
                    cur.append("")
            chunks.append(cur)
            start = len(new_rows)
            for k, chunk in enumerate(chunks):
                new_rows.append(((chunk + [" "] * cols)[:cols], k < len(chunks) - 1))
            if li == cursor_logical[0]:
                offset = cursor_logical[1]
                r = min(offset // cols, len(chunks) - 1) if cols else 0
                cursor_new = start + r
                cursor_col = min(offset - r * cols, cols)
        if not logical:
            new_rows = [([" "] * cols, False)]
        end = max(cursor_new, len(new_rows) - 1)
        top = max(0, end - rows + 1)
        self.scrollback = new_rows[:top]
        screen = new_rows[top: top + rows]
        while len(screen) < rows:
            screen.append(([" "] * cols, False))
        self._main = [r for r, _ in screen]
        self._main_soft = [s for _, s in screen]
        self.row = cursor_new - top
        self.col = cursor_col

    # -- wiring ------------------------------------------------------------------
    def stream(self) -> VTStream:
        """A TTY-like text stream whose writes are fed into this terminal."""
        return VTStream(self)

    def caps(self, *, color: int = 0, unicode: bool = True, fullscreen: bool = True) -> Any:
        """Capabilities whose live ``size()`` follows :meth:`resize`."""
        from topologies.visualization.terminal.caps import TerminalCaps

        return TerminalCaps(is_tty=True, color=color, unicode=unicode, columns=self.cols,
                            rows=self.rows, fullscreen_ok=fullscreen,
                            size_source=lambda: (self.cols, self.rows))

    # -- inspection ---------------------------------------------------------------
    @staticmethod
    def _text(cells: Sequence[str]) -> str:
        return "".join(cells).rstrip()

    def screen_lines(self) -> list[str]:
        return [self._text(r) for r in self.grid]

    def screen_text(self) -> str:
        return "\n".join(self.screen_lines()).rstrip("\n")

    def scrollback_lines(self) -> list[str]:
        return [self._text(r) for r, _ in self.scrollback]

    def scrollback_text(self) -> str:
        return "\n".join(self.scrollback_lines())

    def all_lines(self) -> list[str]:
        """Scrollback followed by the main screen, trailing blank rows dropped."""
        lines = self.scrollback_lines() + [self._text(r) for r in self._main]
        while lines and not lines[-1]:
            lines.pop()
        return lines


class VTStream:
    """File-like adapter feeding a :class:`VirtualTerminal` (``isatty()`` is true)."""

    encoding = "utf-8"

    def __init__(self, vt: VirtualTerminal) -> None:
        self.vt = vt
        self.data: list[str] = []

    def write(self, text: str) -> int:
        self.data.append(text)
        self.vt.feed(text)
        return len(text)

    def flush(self) -> None:
        return

    def isatty(self) -> bool:
        return True

    @property
    def raw(self) -> str:
        return "".join(self.data)


# ---------------------------------------------------------------- pseudo-terminal runs


@dataclass
class PtyRun:
    """Result of :func:`run_in_pty`: raw output, resize offsets, exit status."""

    output: bytes
    resizes: list[tuple[int, int, int]]
    returncode: int
    cols: int
    rows: int
    timed_out: bool = False

    @property
    def text(self) -> str:
        return self.output.decode("utf-8", "replace")

    def replay(self, *, reflow: bool = True) -> VirtualTerminal:
        """Feed the captured bytes into a VirtualTerminal resized at the same offsets."""
        vt = VirtualTerminal(self.cols, self.rows, reflow=reflow, onlcr=False)
        pos = 0
        for offset, cols, rows in self.resizes:
            vt.feed(self.output[pos:offset])
            vt.resize(cols, rows)
            pos = offset
        vt.feed(self.output[pos:])
        return vt


def run_in_pty(argv: Sequence[str], *, cols: int = 100, rows: int = 30,
               resizes: Sequence[tuple[float, int, int]] = (), timeout: float = 120.0,
               env: dict[str, str] | None = None, cwd: str | None = None,
               keys: Sequence[tuple[float, bytes]] = ()) -> PtyRun:
    """Run ``argv`` under a pseudo-terminal (POSIX), resizing it at ``(seconds, cols, rows)``.

    ``keys`` are ``(seconds, bytes)`` written to the child's terminal input.
    """
    import fcntl
    import os
    import pty
    import select
    import signal
    import struct
    import subprocess
    import termios
    import time

    def set_size(fd: int, c: int, r: int) -> None:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", r, c, 0, 0))

    master, slave = pty.openpty()
    set_size(master, cols, rows)
    proc = subprocess.Popen(list(argv), stdin=slave, stdout=slave, stderr=slave,
                            env={**os.environ, **(env or {})}, cwd=cwd,
                            start_new_session=True, close_fds=True)
    os.close(slave)
    fd = master
    out = bytearray()
    applied: list[tuple[int, int, int]] = []
    pending = sorted(resizes)
    pending_keys = sorted(keys)
    start = time.monotonic()
    timed_out = False

    def drain(wait: float) -> bool:
        """Read everything available; False once the child closed the terminal."""
        try:
            while select.select([fd], [], [], wait)[0]:
                chunk = os.read(fd, 65536)
                if not chunk:
                    return False
                out.extend(chunk)
                wait = 0.0
        except OSError:
            return False
        return True

    while True:
        now = time.monotonic() - start
        while pending and pending[0][0] <= now:
            _t, c, r = pending.pop(0)
            # A terminal emulator renders bytes already written before it resizes.
            drain(0.02)
            set_size(fd, c, r)
            applied.append((len(out), c, r))
            with contextlib.suppress(ProcessLookupError):
                os.kill(proc.pid, signal.SIGWINCH)
        while pending_keys and pending_keys[0][0] <= now:
            _t, data = pending_keys.pop(0)
            with contextlib.suppress(OSError):
                os.write(fd, data)
        if now > timeout:
            timed_out = True
            proc.kill()
        alive = drain(0.05)
        if not alive or proc.poll() is not None:
            drain(0.05)
            break
    code = proc.wait()
    with contextlib.suppress(OSError):
        os.close(fd)
    return PtyRun(bytes(out), applied, code, cols, rows, timed_out)


# ---------------------------------------------------------------- recording


class RecordingVisualizer(Visualizer):
    """Stores prepared sessions, every event and every rendered result."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.results: list[Any] = []
        self.sessions: list[tuple[Any, Any]] = []
        self.prepared = 0
        self.closed = 0

    def prepare(self, source: Any = None, context: Any = None) -> None:
        self.prepared += 1
        self.sessions.append((source, context))

    def handle(self, event: Any) -> None:
        self.events.append(event)

    def render(self, result: Any) -> Any:
        self.results.append(result)
        return result

    def close(self) -> None:
        self.closed += 1

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]


# ---------------------------------------------------------------- simulation


def _events_module() -> Any:
    """``training.reporting.events`` without importing the Torch-based training package."""
    import importlib.util
    import sys
    from pathlib import Path

    loaded = sys.modules.get("topologies.training.reporting.events")
    if loaded is not None:
        return loaded
    cached = sys.modules.get("_topologies_simulated_events")
    if cached is not None:
        return cached
    path = Path(__file__).resolve().parents[1] / "training" / "reporting" / "events.py"
    spec = importlib.util.spec_from_file_location("_topologies_simulated_events", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["_topologies_simulated_events"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _curve(name: str, t: float, rng: Any, *, val: bool = False) -> float:
    import math

    lname = name.lower()
    noise = rng.gauss(0, 1)
    if "lr" == lname or "learning_rate" in lname:
        return 1e-3 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    if "acc" in lname or "f1" in lname:
        base = 0.95 - 0.55 * math.exp(-3.0 * t) - (0.03 if val else 0.0)
        return max(0.0, min(1.0, base + 0.015 * noise))
    if "grad" in lname:
        return max(0.01, 1.5 * math.exp(-1.5 * t) + 0.25 + 0.08 * noise)
    base = 2.3 * math.exp(-3.0 * t) + 0.12 + (0.05 if val else 0.0)
    return max(1e-4, base + (0.01 if val else 0.06) * noise)


def _session(visualizer: Any) -> Any:
    from topologies.visualization.core import VisualizerGroup, create_visualizer

    if isinstance(visualizer, VisualizerGroup):
        return visualizer
    if hasattr(visualizer, "handle") and not isinstance(visualizer, (str, list, tuple, dict)):
        return VisualizerGroup([visualizer])
    return create_visualizer(visualizer)


def simulate_training(visualizer: Any, *, epochs: int = 3, steps_per_epoch: int = 50,
                      metrics: Sequence[str] = ("loss", "acc"), validation: bool = True,
                      checkpoints: bool = True, step_delay: float = 0.0, seed: int = 0,
                      run_info: Any = None) -> list[Any]:
    """Drive ``visualizer`` through a synthetic training run on the real event path.

    Events are the Trainer's own dataclasses normalized with ``normalize_event``;
    curves are realistic (exponential loss decay + noise, cosine LR, rising accuracy).
    The session is prepared, rendered with the final history and closed like a real
    ``Trainer.fit()``. Returns the emitted :class:`VisualizationEvent` list.
    """
    import random
    import time

    from topologies.visualization.core import SessionInfo, VisualizationResult, normalize_event

    ev = _events_module()
    rng = random.Random(seed)
    total = int(epochs) * int(steps_per_epoch)
    info = run_info if run_info is not None else ev.RunInfo(
        model_name="SimNet", recipe_name="simulated", param_total=354, device="cpu",
        precision="fp32", optimizer_name="AdamW", lr=1e-3, max_epochs=int(epochs),
        steps_per_epoch=int(steps_per_epoch), batch_size=16, seed=seed,
        eval_every=1 if validation else None, checkpoint_dir="ckpt" if checkpoints else None)
    group = _session(visualizer)
    emitted: list[Any] = []
    t0 = time.monotonic()
    simulated = 0.0

    def emit(raw: Any, **kw: Any) -> None:
        event = normalize_event(raw, source="training", **kw)
        emitted.append(event)
        group.handle_safe(event)

    def elapsed() -> float:
        return time.monotonic() - t0 if step_delay else simulated

    group.prepare(source="training", context=SessionInfo(
        source="training", title=getattr(info, "model_name", None),
        run_info={k: getattr(info, k) for k in vars(info)} if hasattr(info, "__dict__")
        else None))
    history: list[dict[str, Any]] = []
    best = None
    best_epoch = None
    step = 0
    try:
        emit(ev.TrainBeginEvent(info, total_steps=total))
        for epoch in range(int(epochs)):
            epoch_t = elapsed()
            emit(ev.EpochBeginEvent(epoch, int(epochs)))
            sums: dict[str, float] = {}
            for es in range(1, int(steps_per_epoch) + 1):
                step += 1
                t = step / max(1, total)
                values = {m: _curve(m, t, rng) for m in metrics}
                values["lr"] = _curve("lr", t, rng)
                for k, v in values.items():
                    sums[k] = sums.get(k, 0.0) + v
                if step_delay:
                    time.sleep(step_delay)
                else:
                    simulated += 0.01
                emit(ev.StepEvent(step, epoch, es, int(steps_per_epoch), values, elapsed()))
            train = {f"train_{k}": v / steps_per_epoch for k, v in sums.items() if k != "lr"}
            val: dict[str, float] = {}
            if validation:
                emit(ev.ValidationBeginEvent(epoch))
                t = step / max(1, total)
                val = {f"val_{m}": _curve(m, t, rng, val=True) for m in metrics}
                emit({**vars(ev.ValidationEvent(epoch, val, 0.05)), "global_step": step,
                      "kind": "training.validation_completed"})
            key = "val_loss" if "val_loss" in val else next(iter(val), None)
            ckpt = None
            if key is not None:
                value = val[key]
                maximize = "acc" in key
                if best is None or (value > best if maximize else value < best):
                    best, best_epoch = value, epoch
                    if checkpoints:
                        ckpt = "ckpt/best.pt"
                        emit(ev.CheckpointEvent(ckpt, epoch, step, True))
            emit(ev.EpochEndEvent(epoch, train, val, best, elapsed() - epoch_t, ckpt))
            history.append({"epoch": epoch, **train, **val})
        emit(ev.TrainEndEvent(int(epochs), step, elapsed(), best, best_epoch, history))
        group.render_safe(VisualizationResult("history", tuple(history)))
    finally:
        group.close()
    return emitted


def simulate_experiment(visualizer: Any, *, axes: Any = None, seeds: int = 2,
                        epochs: int = 5, failures: int = 1, step_delay: float = 0.0,
                        seed: int = 0, name: str = "demo",
                        goal: tuple[str, str] = ("val_loss", "min"),
                        steps_per_epoch: int = 4) -> list[Any]:
    """Drive ``visualizer`` through a synthetic experiment (events + final ResultSet).

    ``axes`` defaults to ``{"lr": [0.1, 1.0, 3.0], "width": [8, 32]}``; the last
    ``failures`` trials fail with an error so failure displays can be checked.
    """
    import itertools
    import math
    import random
    import time
    import zlib

    from topologies.visualization.adapters import adapt_result
    from topologies.visualization.core import SessionInfo, normalize_event

    axes = dict(axes or {"lr": [0.1, 1.0, 3.0], "width": [8, 32]})
    rng = random.Random(seed)
    combos = list(itertools.product(*axes.values()))
    plan = []
    for s in range(int(seeds)):
        for combo in combos:
            labels = {k: v for k, v in zip(axes, combo)}
            digest = zlib.crc32(repr((combo, s)).encode()) % 10**6
            plan.append({"trial_id": f"t{len(plan):03d}{digest:06d}",
                         "labels": {**labels, "seed": s}, "seed": s,
                         "index": len(plan) + 1})
    metric, mode = goal
    group = _session(visualizer)
    emitted: list[Any] = []
    run_id = f"simulated:{name}"

    def emit(raw: dict[str, Any], trial_id: str | None = None) -> None:
        event = normalize_event(raw, source="experiment", run_id=run_id, trial_id=trial_id)
        emitted.append(event)
        group.handle_safe(event)

    group.prepare(source="experiment", context=SessionInfo(
        source="experiment", run_id=run_id, title=name, goal=goal,
        axes={k: tuple(v) for k, v in axes.items()}, total_trials=len(plan)))
    results = []
    best = None
    try:
        emit({"type": "started", "name": name, "goal": list(goal), "current": 0,
              "total": len(plan), "axes": {k: list(v) for k, v in axes.items()},
              "varying": {k: list(v) for k, v in axes.items() if len(v) > 1},
              "pending_ids": [p["trial_id"] for p in plan], "cached": [],
              "trials": plan, "workers": 1, "durations": {}, "strategy": "grid",
              "seeds": int(seeds)})
        for order, trial in enumerate(plan, start=1):
            tid = trial["trial_id"]
            labels = trial["labels"]
            emit({"type": "trial_started", "trial_id": tid, "index": trial["index"],
                  "labels": labels, "config": {k: v for k, v in labels.items() if k != "seed"},
                  "seed": trial["seed"], "max_epochs": epochs,
                  "total_steps": epochs * steps_per_epoch, "current": order,
                  "total": len(plan)}, tid)
            started = time.monotonic()
            quality = 1.0 / (1.0 + sum(abs(math.log10(float(v) + 1e-9)) if isinstance(v, (int,
                                       float)) else 0.0 for k, v in labels.items() if k != "seed"))
            failing = order > len(plan) - int(failures)
            history = []
            gstep = 0
            for epoch in range(int(epochs)):
                for _ in range(steps_per_epoch):
                    gstep += 1
                    if step_delay:
                        time.sleep(step_delay)
                    loss = 2.0 * math.exp(-gstep * quality / 4) + 0.05 + rng.random() * 0.02
                    emit({"type": "progress", "trial_id": tid, "global_step": gstep,
                          "epoch": epoch, "total_steps": epochs * steps_per_epoch,
                          "metrics": {"loss": loss}}, tid)
                if failing and epoch == 1:
                    break
                value = 1.5 * math.exp(-(epoch + 1) * quality) + 0.02 + rng.random() * 0.02
                history.append({"epoch": epoch, metric: value})
                emit({"type": "epoch", "trial_id": tid, "epoch": epoch,
                      "metrics": {metric: value}}, tid)
            wall = time.monotonic() - started if step_delay else 0.3
            if failing:
                result = {"trial_id": tid, "status": "failed", "labels": labels,
                          "goal": {"metric": metric, "value": None, "mode": mode},
                          "cost": {"wall_s": wall}, "history": history,
                          "error": {"type": "RuntimeError",
                                    "message": "shape mismatch (32 vs 64)"}}
            else:
                result = {"trial_id": tid, "status": "completed", "labels": labels,
                          "goal": {"metric": metric, "value": history[-1][metric],
                                   "mode": mode, "select": "last"},
                          "cost": {"wall_s": wall}, "history": history,
                          "final": {metric: history[-1][metric]}}
            results.append(result)
            status = result["status"]
            metrics = dict(result.get("final") or {})
            emit({"type": "trial_completed" if status == "completed" else "trial_failed",
                  "trial_id": tid, "status": status, "config": labels, "result": result,
                  "metrics": metrics, "goal_metric": metric, "mode": mode}, tid)
            emit({"type": "leaderboard_changed", "trial_id": tid, "status": status}, tid)
            value = result["goal"]["value"]
            if value is not None and (best is None or (value < best if mode == "min"
                                                       else value > best)):
                best = value
                emit({"type": "best_configuration_changed", "trial_id": tid,
                      "config": labels, "metric": metric, "value": value, "mode": mode}, tid)
        rs = _result_set(results, goal, list(axes), name)
        if rs is not None:
            group.render_safe(adapt_result(rs))
        emit({"type": "completed", "result": rs})
    finally:
        group.close()
    return emitted


def _result_set(results: list[dict[str, Any]], goal: tuple[str, str], axes: list[str],
                name: str) -> Any:
    try:
        from topologies.experiments.results import ResultSet, TrialRecord
    except Exception:  # noqa: BLE001
        return None
    records = [TrialRecord.from_dict(r) for r in results]
    return ResultSet(records, goal=goal, axes=axes, name=name)


__all__ = ["PtyRun", "RecordingVisualizer", "VTStream", "VirtualTerminal", "run_in_pty",
           "simulate_experiment", "simulate_training"]
