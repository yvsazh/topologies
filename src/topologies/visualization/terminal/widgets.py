"""Pure terminal widgets: progress bars, sparklines, braille charts, tables, boxes.

Every widget takes explicit sizes and returns a :class:`Line` (single-row widgets) or a
``list[Line]``; nothing is encoded or written here. Widgets never raise: a problem
renders one ``muted`` error line instead (``TOPOLOGIES_VIZ_DEBUG=1`` re-raises).
"""

from __future__ import annotations

import itertools
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from topologies.visualization.format import MISSING, fmt_column, fmt_number
from topologies.visualization.terminal.glyphs import UNICODE
from topologies.visualization.terminal.text import (
    Line,
    Span,
    as_line,
    cell_width,
    line,
    styled,
)

G = UNICODE
BLOCKS = G.blocks
MARKERS = G.markers
_BRAILLE = 0x2800
# Dot bit for (column 0..1, row 0..3) inside one braille cell.
_DOTS = ((0x01, 0x02, 0x04, 0x40), (0x08, 0x10, 0x20, 0x80))


def debug_enabled() -> bool:
    return os.environ.get("TOPOLOGIES_VIZ_DEBUG", "").strip().lower() not in ("", "0", "false")


def _error_line(name: str, exc: BaseException, width: int | None = None) -> Line:
    if debug_enabled():
        raise exc
    ln = styled(f"<{name} failed: {type(exc).__name__}: {exc}>", "muted")
    return ln.clip(width) if width else ln


def safe(name: str) -> Callable:
    """Decorator: turn widget exceptions into one muted error line (list or Line)."""
    def wrap(fn: Callable) -> Callable:
        def inner(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                width = kwargs.get("width")
                if width is None and len(args) > 1 and isinstance(args[1], int):
                    width = args[1]
                err = _error_line(name, exc, width)
                return err if getattr(fn, "_single", False) else [err]
        inner.__name__ = fn.__name__
        inner.__doc__ = fn.__doc__
        inner.__wrapped__ = fn  # type: ignore[attr-defined]
        return inner
    return wrap


def _single(fn: Callable) -> Callable:
    fn._single = True  # type: ignore[attr-defined]
    return fn


def _finite(values: Any) -> list[float]:
    try:
        isfinite = math.isfinite
        return [f for f in map(float, values or ()) if isfinite(f)]
    except (TypeError, ValueError):
        pass
    out = []
    for v in values or ():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


# ---------------------------------------------------------------- progress


@safe("progress_bar")
@_single
def progress_bar(fraction: float | None, width: int, *, style: str = "line") -> Line:
    """Thin line bar ``━━━━━╸─────`` with half-cell resolution; ``None`` = static todo bar."""
    width = max(1, int(width))
    if fraction is None:
        return styled(G.bar_todo * width, "muted")
    try:
        frac = float(fraction)
    except (TypeError, ValueError):
        return styled(G.bar_todo * width, "muted")
    frac = 0.0 if not math.isfinite(frac) else min(1.0, max(0.0, frac))
    halves = int(frac * width * 2 + 1e-9)
    full, half = divmod(halves, 2)
    full = min(full, width)
    half = half if full < width else 0
    todo = width - full - half
    return line((G.bar_done * full + G.bar_half * half, "progress.done"),
                (G.bar_todo * todo, "progress.todo"))


def fmt_percent(fraction: float | None) -> str:
    """Right-aligned 4-cell percentage (``" 62%"``); unknown → ``"  ?%"``."""
    if fraction is None:
        return "  ?%"
    try:
        f = min(1.0, max(0.0, float(fraction)))
    except (TypeError, ValueError):
        return "  ?%"
    pct = int(f * 100) if f < 1 else 100
    return f"{pct:3d}%"


# ---------------------------------------------------------------- sparkline


def ema(values: Sequence[float], weight: float) -> list[float]:
    """Exponential moving average with smoothing ``weight`` in [0, 1) (0 = raw)."""
    w = min(0.999, max(0.0, float(weight)))
    out: list[float] = []
    acc = None
    for v in values:
        acc = v if acc is None else w * acc + (1 - w) * v
        out.append(acc)
    return out


def bucket(values: Sequence[float], n: int, agg: str = "mean") -> list[float]:
    """Aggregate ``values`` into ``n`` contiguous buckets (``mean|last|min|max``)."""
    if len(values) <= n:
        return list(values)
    out = []
    size = len(values) / n
    for i in range(n):
        chunk = values[int(i * size): max(int(i * size) + 1, int((i + 1) * size))]
        if agg == "last":
            out.append(chunk[-1])
        elif agg == "min":
            out.append(min(chunk))
        elif agg == "max":
            out.append(max(chunk))
        else:
            out.append(sum(chunk) / len(chunk))
    return out


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    k = (len(sorted_values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_values[int(k)]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def robust_range(values: Sequence[float]) -> tuple[float, float]:
    """2nd–98th percentile for ≥ 20 points (one spike cannot flatten the rest)."""
    s = sorted(values)
    if len(s) >= 20:
        lo, hi = percentile(s, 0.02), percentile(s, 0.98)
        if hi > lo:
            return lo, hi
    return s[0], s[-1]


@safe("sparkline")
@_single
def sparkline(values: Any, width: int, *, smoothing: float | None = None,
              lo: float | None = None, hi: float | None = None, agg: str = "mean",
              style: Any = None) -> Line:
    """Block sparkline aggregating into ``width`` buckets (never sampling)."""
    vals = _finite(values)
    if not vals:
        return Line()
    width = max(1, int(width))
    if smoothing:
        vals = ema(vals, smoothing)
    rlo, rhi = robust_range(vals)
    lo = rlo if lo is None else float(lo)
    hi = rhi if hi is None else float(hi)
    cells = bucket(vals, width, agg)
    if hi <= lo:
        return styled(BLOCKS[3] * len(cells), style)
    top = len(BLOCKS) - 1
    chars = []
    for v in cells:
        frac = (v - lo) / (hi - lo)
        chars.append(BLOCKS[max(0, min(top, round(frac * top)))])
    return styled("".join(chars), style)


# ---------------------------------------------------------------- ticks


def nice_number(x: float, round_: bool) -> float:
    if x <= 0:
        return 1.0
    exp = math.floor(math.log10(x))
    f = x / 10**exp
    if round_:
        nf = 1 if f < 1.5 else 2 if f < 2.25 else 2.5 if f < 3.5 else 5 if f < 7.5 else 10
    else:
        nf = 1 if f <= 1 else 2 if f <= 2 else 2.5 if f <= 2.5 else 5 if f <= 5 else 10
    return nf * 10**exp


def nice_ticks(lo: float, hi: float, max_ticks: int = 5) -> list[float]:
    """Ticks at 1, 2, 2.5, 5 × 10^k covering ``[lo, hi]`` (inclusive, extended)."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return []
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        pad = abs(lo) * 0.1 or 1.0
        lo, hi = lo - pad, hi + pad
    max_ticks = max(2, max_ticks)
    span = nice_number(hi - lo, False)
    step = nice_number(span / (max_ticks - 1), True)
    while math.floor(lo / step) * step + step * (max_ticks - 1) < hi - 1e-12 * step:
        step = nice_number(step * 1.5, True)
    start = math.floor(lo / step + 1e-9) * step
    ticks: list[float] = []
    k = 0
    while len(ticks) < 50:
        value = round((start + k * step) / step) * step
        ticks.append(value)
        if value >= hi - 1e-12 * max(1.0, abs(hi)):
            break
        k += 1
    return ticks


def fmt_ticks(values: Sequence[float], *, integer: bool = False) -> list[str]:
    """Shared-decimal tick labels; integers with thousands separators."""
    if integer:
        return [f"{round(v):,}" for v in values]
    clean = [0.0 if abs(v) < 1e-12 * max(1.0, max(abs(x) for x in values)) else v
             for v in values]
    labels = fmt_column(clean, sig=3)
    # Trim shared trailing zeros so ticks read 0.5 / 1.0 / 1.5 rather than 0.500.
    while labels and all("." in s and s.endswith("0") and "e" not in s for s in labels):
        labels = [s[:-1].rstrip(".") if s[:-1].endswith(".") else s[:-1] for s in labels]
    return labels


# ---------------------------------------------------------------- line chart


def _series_points(seq: Any) -> list[tuple[float, float, float | None]]:
    if hasattr(seq, "x") and hasattr(seq, "y") and not isinstance(seq, (list, tuple)):
        seq = list(zip(seq.x, seq.y))
    try:  # fast path: well-formed numeric tuples
        fast = [(float(p[0]), float(p[1]),
                 float(p[2]) if len(p) > 2 and p[2] is not None else None) for p in seq or ()]
        isfinite = math.isfinite
        out = [(x, y, s if s is not None and isfinite(s) else None) for x, y, s in fast
               if isfinite(x) and isfinite(y)]
        if all(out[i][0] <= out[i + 1][0] for i in range(len(out) - 1)):
            return out
        return sorted(out, key=lambda t: t[0])
    except (TypeError, ValueError, IndexError):
        pass
    clean = []
    for p in seq or ():
        try:
            x, y = float(p[0]), float(p[1])
            s = float(p[2]) if len(p) > 2 and p[2] is not None else None
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            clean.append((x, y, s if s is not None and math.isfinite(s) else None))
    return sorted(clean, key=lambda t: t[0])


@dataclass
class _Canvas:
    w: int
    h: int

    def __post_init__(self) -> None:
        self.bits = [[0] * self.w for _ in range(self.h)]
        self.owner: list[list[int | None]] = [[None] * self.w for _ in range(self.h)]
        self.band = [[0] * self.w for _ in range(self.h)]
        self.band_owner: list[list[int | None]] = [[None] * self.w for _ in range(self.h)]
        self.marks: dict[tuple[int, int], tuple[str, int]] = {}

    def dot(self, gx: int, gy: int, idx: int, band: bool = False) -> None:
        if not (0 <= gx < 2 * self.w and 0 <= gy < 4 * self.h):
            return
        cx, cy = gx // 2, gy // 4
        if band:
            self.band[cy][cx] |= _DOTS[gx % 2][gy % 4]
            self.band_owner[cy][cx] = idx
        else:
            self.bits[cy][cx] |= _DOTS[gx % 2][gy % 4]
            self.owner[cy][cx] = idx

    def segment(self, a: tuple[int, int], b: tuple[int, int], idx: int) -> None:
        (ax, ay), (bx, by) = a, b
        dx, dy = abs(bx - ax), -abs(by - ay)
        sx, sy = (1 if ax < bx else -1), (1 if ay < by else -1)
        err = dx + dy
        while True:
            self.dot(ax, ay, idx)
            if ax == bx and ay == by:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                ax += sx
            if e2 <= dx:
                err += dx
                ay += sy


def line_chart(series: Mapping[str, Any], width: int, height: int, *,
               x_label: str | None = None, y_label: str | None = None, log_y: bool = False,
               band: bool | None = None, markers: bool | None = None, color: bool = True,
               legend: bool = True, highlight: str | None = None,
               direct_labels: bool = True, styles: Mapping[str, Any] | None = None,
               slots: Mapping[str, int] | None = None,
               x_integer: bool | None = None, background: Sequence[str] = ()) -> list[Line]:
    """Multi-series braille chart; points are ``(x, y)`` or ``(x, mean, std)``.

    ``height`` is the number of plot rows; axis, tick labels, axis titles and the
    legend add rows below/above. Without color every series also gets a marker glyph
    (``●▲■◆○△□◇``) at its data points, and the legend uses the same glyph.
    ``background`` names series drawn first and left out of the legend and direct
    labels (e.g. a faint raw curve behind its smoothed version).
    """
    try:
        return _line_chart(series, int(width), int(height), x_label, y_label, log_y, band,
                           markers, color, legend, highlight, direct_labels, styles, slots,
                           x_integer, tuple(background))
    except Exception as exc:  # noqa: BLE001
        return [_error_line("chart", exc, int(width) if isinstance(width, int) else None)]


def _line_chart(series: Mapping[str, Any], width: int, height: int, x_label: str | None,
                y_label: str | None, log_y: bool, band: bool | None, markers: bool | None,
                color: bool, legend: bool, highlight: str | None, direct_labels: bool,
                styles: Mapping[str, Any] | None, slots: Mapping[str, int] | None,
                x_integer: bool | None, background: tuple[str, ...] = ()) -> list[Line]:
    if not isinstance(series, Mapping):
        raise TypeError("series must be a mapping of name -> points")
    pts: dict[str, list[tuple[float, float, float | None]]] = {}
    dropped = 0
    for name, seq in series.items():
        clean = _series_points(seq)
        if log_y:
            kept = []
            for x, y, s in clean:
                if y <= 0:
                    dropped += 1
                    continue
                lo_s = None if s is None else s / (y * math.log(10))
                kept.append((x, math.log10(y), lo_s))
            clean = kept
        if clean:
            pts[str(name)] = clean
    if not pts:
        return [styled("(no data)", "muted")]
    names = list(pts)
    other: list[str] = []
    if len(names) > 8:
        names, other = names[:7], names[7:]
    show_band = (color if band is None else bool(band) and color) and any(
        p[2] is not None for n in names for p in pts[n])
    band_hidden = (band is not False) and not color and any(
        p[2] is not None for n in names for p in pts[n])
    use_markers = (not color) if markers is None else bool(markers)
    xs = [p[0] for n in names + other for p in pts[n]]
    ys = [p[1] for n in names + other for p in pts[n]]
    if show_band:
        ys += [p[1] + sgn * p[2] for n in names for p in pts[n] if p[2] is not None
               for sgn in (-1, 1)]
    x0, x1 = min(xs), max(xs)
    if x1 == x0:
        x0, x1 = x0 - 1, x1 + 1
    h = max(2, height)
    yt = nice_ticks(min(ys), max(ys), max_ticks=3 if h < 6 else (4 if h < 10 else 5))
    y0, y1 = yt[0], yt[-1]
    if y1 == y0:
        y0, y1 = y0 - 1, y1 + 1
    if log_y:
        ylabels = [f"{10 ** t:.0e}" if abs(t) >= 3 or t != int(t) and abs(t) >= 2
                   else fmt_number(10 ** t, sig=2) for t in yt]
        if any("e" in lab for lab in ylabels):
            ylabels = [f"{10 ** t:.0e}" for t in yt]
    else:
        ylabels = fmt_ticks(yt)
    lw = max(cell_width(t) for t in ylabels)
    last_vals = {n: pts[n][-1][1] for n in names if n not in background}
    dl_texts = {n: fmt_number(10 ** v if log_y else v, sig=3) for n, v in last_vals.items()}
    dlw = (max(cell_width(t) for t in dl_texts.values()) + 1) if dl_texts else 0
    w = max(4, width - lw - 2)
    use_direct = direct_labels and len(last_vals) <= 4 and w - dlw >= 16
    if use_direct:
        w -= dlw
    canvas = _Canvas(w, h)
    pw, ph = 2 * w, 4 * h

    def px(x: float) -> int:
        return max(0, min(pw - 1, round((x - x0) / (x1 - x0) * (pw - 1))))

    def py(y: float) -> int:
        return max(0, min(ph - 1, round((y1 - y) / (y1 - y0) * (ph - 1))))

    for n in list(pts):
        if len(pts[n]) > 4 * pw:
            pts[n] = _decimate(pts[n], px)
    order = [n for n in names if n in background] + \
        [n for n in names if n != highlight and n not in background] + \
        ([highlight] if highlight in names else [])
    index = {n: i for i, n in enumerate(names)}
    other_idx = len(names)
    if show_band:
        for n in order:
            seq = [p for p in pts[n] if p[2] is not None]
            for (xa, ya, sa), (xb, yb, sb) in itertools.pairwise(seq):
                ga, gb = px(xa), px(xb)
                for gx in range(ga, gb + 1):
                    t = 0.0 if gb == ga else (gx - ga) / (gb - ga)
                    m = ya + (yb - ya) * t
                    s = (sa or 0) + ((sb or 0) - (sa or 0)) * t
                    top, bot = py(m + s), py(m - s)
                    for gy in range(min(top, bot), max(top, bot) + 1):
                        canvas.dot(gx, gy, index[n], band=True)
            if len(seq) == 1:
                x, m, s = seq[0]
                for gy in range(py(m + (s or 0)), py(m - (s or 0)) + 1):
                    canvas.dot(px(x), gy, index[n], band=True)
    for n in other:
        prev = None
        for x, y, _s in pts[n]:
            cur = (px(x), py(y))
            if prev is None:
                canvas.dot(*cur, other_idx)
            else:
                canvas.segment(prev, cur, other_idx)
            prev = cur
    for n in order:
        prev = None
        for x, y, _s in pts[n]:
            cur = (px(x), py(y))
            if prev is None:
                canvas.dot(*cur, index[n])
            else:
                canvas.segment(prev, cur, index[n])
            prev = cur
    if use_markers:
        for n in [k for k in order if k not in background]:
            i = index[n]
            glyph = MARKERS[i % len(MARKERS)]
            seq = pts[n]
            stride = max(1, math.ceil(len(seq) / max(1, w // 4)))
            chosen = seq[::stride]
            if seq[-1] not in chosen:
                chosen.append(seq[-1])
            for x, y, _s in chosen:
                canvas.marks[(py(y) // 4, px(x) // 2)] = (glyph, i)
    zero_row = None
    if not log_y and y0 < 0 < y1:
        zero_row = py(0.0) // 4

    def series_style(i: int) -> Any:
        if i == other_idx:
            return "muted"
        name = names[i]
        if styles and name in styles:
            st = styles[name]
        elif color:
            slot = slots.get(name, i) if slots else i
            st = f"series.{slot}"
        else:
            st = None
        if highlight is not None and name == highlight:
            st = f"{st}+bold" if isinstance(st, str) else "bold"
        return st

    tick_rows: dict[int, str] = {}
    for t, lab in zip(yt, ylabels):
        r = py(t) // 4
        if r not in tick_rows:
            tick_rows[r] = lab
    rows: list[Line] = []
    if y_label:
        rows.append(styled(y_label, "muted").clip(width))
    label_at: dict[int, list[str]] = {}
    if use_direct:
        taken: set[int] = set()
        for n in sorted(last_vals, key=lambda k: py(last_vals[k])):
            r = min(h - 1, py(last_vals[n]) // 4)
            while r in taken and r < h - 1:
                r += 1
            while r in taken and r > 0:
                r -= 1
            if r in taken:
                continue
            taken.add(r)
            label_at.setdefault(r, []).append(n)
    for r in range(h):
        parts: list[Any] = []
        lab = tick_rows.get(r, "")
        parts.append(" " * (lw - cell_width(lab)) + lab)
        parts.append((" ┤" if lab else " │", "muted"))
        cells: list[tuple[str, Any]] = []
        for c in range(w):
            mark = canvas.marks.get((r, c))
            if mark is not None:
                ch, st = mark[0], series_style(mark[1])
            elif canvas.bits[r][c]:
                ch = chr(_BRAILLE + (canvas.bits[r][c] | canvas.band[r][c]))
                o = canvas.owner[r][c]
                st = series_style(o) if o is not None else None
            elif canvas.band[r][c]:
                ch = chr(_BRAILLE + canvas.band[r][c])
                o = canvas.band_owner[r][c]
                base = series_style(o) if o is not None else None
                st = f"{base}+dim" if isinstance(base, str) else "dim"
            elif zero_row == r:
                ch, st = "┈", "muted"
            else:
                ch, st = " ", None
            cells.append((ch, st))
        parts.extend(_runs(cells))
        if use_direct and r in label_at:
            n = label_at[r][0]
            parts.append(" " + dl_texts[n])
        rows.append(line(*parts))
    x_int = x_integer if x_integer is not None else all(
        float(p[0]).is_integer() for n in names + other for p in pts[n])
    xt = [t for t in nice_ticks(x0, x1, max_ticks=max(3, min(6, w // 8)))
          if x0 - 1e-9 <= t <= x1 + 1e-9]
    if x_int:
        xt = [t for t in xt if float(t).is_integer()]
    xlabels = fmt_ticks(xt, integer=x_int) if xt else []
    axis = [" "] * w
    label_row = [" "] * (w + 2)
    last_end = -2
    for t, lab in zip(xt, xlabels):
        c = round((t - x0) / (x1 - x0) * (w - 1))
        c = max(0, min(w - 1, c))
        axis[c] = "┴"
        start = max(0, min(c - cell_width(lab) // 2, w - cell_width(lab)))
        if start <= last_end + 1:
            continue
        for k, ch in enumerate(lab):
            if start + k < len(label_row):
                label_row[start + k] = ch
        last_end = start + cell_width(lab) - 1
    axis_line = "".join("─" if a == " " else a for a in axis)
    rows.append(line(" " * lw, (" └" + axis_line, "muted")))
    tick_text = "".join(label_row).rstrip()
    if x_label:
        room = w - cell_width(tick_text) - 2
        if room >= cell_width(x_label):
            tick_text = tick_text + " " * (w - cell_width(tick_text) - cell_width(x_label)) \
                + x_label
            rows.append(line(" " * (lw + 2), (tick_text, "muted")).clip(width))
        else:
            rows.append(line(" " * (lw + 2), (tick_text, "muted")).clip(width))
            rows.append(line(" " * max(0, lw + 2 + w - cell_width(x_label)),
                             (x_label, "muted")).clip(width))
    elif tick_text:
        rows.append(line(" " * (lw + 2), (tick_text, "muted")).clip(width))
    if legend:
        items: list[Line] = []
        for n in names:
            if n in background:
                continue
            i = index[n]
            glyph = MARKERS[i % len(MARKERS)] if use_markers else "●"
            st = series_style(i) if (color or use_markers) else None
            items.append(line((glyph, st), " ", n))
        if other:
            items.append(line(("─", "muted"), (f" other ({len(other)} more)", "muted")))
        notes = []
        if band_hidden:
            notes.append("mean (band hidden: no color)")
        elif show_band:
            notes.append("band = ±1 std")
        if dropped:
            notes.append(f"{dropped} non-positive value{'s' if dropped != 1 else ''} dropped")
        rows.extend(_flow(items, width - lw - 2, indent=lw + 2, gap=3))
        for note in notes:
            rows.append(line(" " * (lw + 2), (note, "muted")).clip(width))
    return [r.clip(width) for r in rows]


def _decimate(seq: list[tuple[float, float, float | None]], px: Callable[[float], int]
              ) -> list[tuple[float, float, float | None]]:
    """Keep first/min/max/last point per pixel column (extremes survive)."""
    out: list[tuple[float, float, float | None]] = []
    bucket: list[tuple[float, float, float | None]] = []
    current = None
    for p in seq:
        col = px(p[0])
        if col != current and bucket:
            out.extend(_bucket_points(bucket))
            bucket = []
        current = col
        bucket.append(p)
    if bucket:
        out.extend(_bucket_points(bucket))
    return out


def _bucket_points(bucket: list[tuple[float, float, float | None]]
                   ) -> list[tuple[float, float, float | None]]:
    if len(bucket) <= 4:
        return bucket
    lo = min(bucket, key=lambda p: p[1])
    hi = max(bucket, key=lambda p: p[1])
    keep = {id(bucket[0]): bucket[0], id(lo): lo, id(hi): hi, id(bucket[-1]): bucket[-1]}
    return sorted(keep.values(), key=lambda p: p[0])


def _runs(cells: Sequence[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Merge consecutive same-style characters into ``(text, style)`` runs."""
    out: list[tuple[str, Any]] = []
    for ch, st in cells:
        if out and out[-1][1] == st:
            out[-1] = (out[-1][0] + ch, st)
        else:
            out.append((ch, st))
    return out


def _flow(items: Sequence[Line], width: int, *, indent: int = 0, gap: int = 2) -> list[Line]:
    """Wrap items into as many rows as needed, each ≤ ``width`` (+indent)."""
    rows: list[Line] = []
    cur: list[Any] = []
    used = 0
    for item in items:
        w = item.width
        need = w + (gap if cur else 0)
        if cur and used + need > width:
            rows.append(line(" " * indent, *cur))
            cur, used = [], 0
            need = w
        if cur:
            cur.append(" " * gap)
        cur.append(item)
        used += need
    if cur:
        rows.append(line(" " * indent, *cur))
    return rows


# ---------------------------------------------------------------- table


@dataclass(frozen=True)
class Column:
    """Table column: alignment (``<``/``>``/auto), drop priority, numeric formatting."""

    name: str
    align: str | None = None
    priority: int = 50
    numeric: bool | None = None
    sig: int = 4
    min_width: int = 3
    style: Any = None


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _cell_line(v: Any) -> Line:
    if isinstance(v, Line):
        return v
    if isinstance(v, Span):
        return Line((v,))
    if v is None:
        return styled(MISSING, "muted")
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str):
        return line(v)
    return as_line(str(v))


def _empty_cell(v: Any) -> bool:
    if v is None:
        return True
    text = v.plain if isinstance(v, Line) else (v.text if isinstance(v, Span) else str(v))
    return text.strip() in ("", MISSING, "-", "n/a")


def table(columns: Sequence[str | Column], rows: Sequence[Sequence[Any]], *, width: int,
          align: Mapping[str, str] | None = None, heat: Mapping[str, str] | None = None,
          max_rows: int | None = None, zebra: bool = False, hide_empty: bool = True,
          color: bool = True, indent: int = 1, gap: int = 2, title: str | None = None,
          header: bool = True) -> list[Line]:
    """Aligned table: numeric columns right-aligned with shared decimals.

    Width negotiation: shrink the widest text columns first, then drop columns by
    ascending ``priority``, finally clip text cells with ``…``. ``heat`` maps column
    names to ``"min"``/``"max"``: the best value is marked ``★`` (``best`` token) and,
    without color, cells get ``░▒▓█`` shading.
    """
    try:
        return _table(columns, rows, width, align, heat, max_rows, zebra, hide_empty, color,
                      indent, gap, title, header)
    except Exception as exc:  # noqa: BLE001
        return [_error_line("table", exc, width)]


def _table(columns: Sequence[str | Column], rows: Sequence[Sequence[Any]], width: int,
           align: Mapping[str, str] | None, heat: Mapping[str, str] | None,
           max_rows: int | None, zebra: bool, hide_empty: bool, color: bool, indent: int,
           gap: int, title: str | None, header: bool) -> list[Line]:
    cols = [c if isinstance(c, Column) else Column(str(c)) for c in columns]
    data = [list(r) + [None] * (len(cols) - len(r)) for r in rows]
    extra = 0
    if max_rows is not None and len(data) > max_rows:
        extra = len(data) - max_rows
        data = data[:max_rows]
    keep = list(range(len(cols)))
    if hide_empty and data:
        keep = [j for j in keep if not all(_empty_cell(r[j]) for r in data)]
    numeric = {}
    for j in keep:
        c = cols[j]
        vals = [r[j] for r in data if r[j] is not None]
        numeric[j] = c.numeric if c.numeric is not None else (
            bool(vals) and all(_is_num(v) for v in vals))
    cells: list[list[Line]] = [[Line()] * len(cols) for _ in data]
    for j in keep:
        if numeric[j]:
            texts = fmt_column([r[j] if _is_num(r[j]) else None for r in data], sig=cols[j].sig)
            for i, r in enumerate(data):
                cells[i][j] = styled(texts[i], cols[j].style) if _is_num(r[j]) else \
                    _cell_line(r[j])
        else:
            for i, r in enumerate(data):
                cells[i][j] = _cell_line(r[j])
    if heat:
        for j in keep:
            mode = heat.get(cols[j].name)
            if not mode:
                continue
            vals = [(i, float(r[j])) for i, r in enumerate(data) if _is_num(r[j])
                    and math.isfinite(float(r[j]))]
            if not vals:
                continue
            lo = min(v for _, v in vals)
            hi = max(v for _, v in vals)
            best_v = lo if mode == "min" else hi
            for i, v in vals:
                is_best = v == best_v
                base = cells[i][j]
                if color:
                    st = "best" if is_best else None
                    cells[i][j] = line(Line(Span(s.text, st or s.style) for s in base),
                                       (G.best if is_best else " ", "best" if is_best else None))
                else:
                    good = 1.0 if hi == lo else ((hi - v) / (hi - lo) if mode == "min"
                                                 else (v - lo) / (hi - lo))
                    shade = "░▒▓█"[max(0, min(3, round(good * 3)))]
                    cells[i][j] = line(shade + " ", base, G.best if is_best else " ")
    aligns = {}
    for j in keep:
        a = (align or {}).get(cols[j].name) or cols[j].align
        aligns[j] = a or (">" if numeric[j] else "<")
    natural = {j: max([cell_width(cols[j].name)] + [cells[i][j].width for i in range(len(data))])
               for j in keep}
    widths = dict(natural)

    def total() -> int:
        return indent + sum(widths[j] for j in keep) + gap * max(0, len(keep) - 1)

    floor = {j: (natural[j] if numeric[j] else max(cols[j].min_width,
                                                   min(natural[j], cell_width(cols[j].name), 8)))
             for j in keep}
    while total() > width:
        text_cols = [j for j in keep if not numeric[j] and widths[j] > floor[j]]
        if text_cols:
            j = max(text_cols, key=lambda k: (widths[k], k))
            widths[j] -= 1
            continue
        if len(keep) > 1:
            drop = min(keep, key=lambda k: (cols[k].priority, -k))
            keep.remove(drop)
            continue
        break
    out: list[Line] = []
    if title:
        out.append(styled(title, "title").clip(width))

    def row_line(values: list[Line], style: Any = None) -> Line:
        parts: list[Any] = [" " * indent]
        for k, j in enumerate(keep):
            cell = values[j]
            if style is not None:
                cell = Line(Span(s.text, style if s.style is None else s.style) for s in cell)
            parts.append(cell.pad(widths[j], aligns[j]))
            if k < len(keep) - 1:
                parts.append(" " * gap)
        return line(*parts).clip(width)

    if header:
        out.append(row_line([styled(c.name, "label") for c in cols]))
        rule_w = min(width, total()) - indent
        out.append(line(" " * indent, ("─" * max(0, rule_w), "border")))
    for i, row in enumerate(cells):
        out.append(row_line(row, "muted" if zebra and i % 2 else None))
    if extra:
        out.append(line(" " * indent, (f"… {extra} more", "muted")))
    return out


# ---------------------------------------------------------------- small widgets


@safe("kv_line")
@_single
def kv_line(pairs: Sequence[tuple[str, Any]], width: int, *, sep: str = "  ·  ") -> Line:
    """``label value  ·  label value``; pairs with ``None`` skipped, tail dropped to fit."""
    items: list[Line] = []
    for key, value in pairs:
        if value is None:
            continue
        val = value if isinstance(value, Line) else styled(str(value), "value")
        items.append(line((str(key), "muted"), " ", val) if key else val)
    out = Line()
    for item in items:
        candidate = item if not out else line(out, (sep, "muted"), item)
        if candidate.width > width:
            break
        out = candidate
    return out


@safe("rule")
@_single
def rule(width: int, title: str | None = None, *, style: Any = "border",
         title_style: Any = "title") -> Line:
    """``── title ─────`` across ``width`` cells."""
    width = max(0, int(width))
    if not title:
        return styled(G.rule * width, style)
    head = line((G.rule * 2 + " ", style), (title, title_style), (" ", style))
    if head.width >= width:
        return head.clip(width)
    return line(head, (G.rule * (width - head.width), style))


@safe("badge")
@_single
def badge(text: str, token: str = "accent", glyph: str | None = None) -> Line:
    """Status badge such as ``● training`` / ``✓ done`` / ``✗ failed`` (glyph + word)."""
    if glyph is None:
        glyph = {"good": G.completed, "bad": G.failed, "warn": "▲"}.get(token, "●")
    return line((glyph, token), " ", (text, token))


@safe("meter")
@_single
def meter(label: str, fraction: float | None, width: int, *, label_width: int = 0) -> Line:
    """``label ━━━━╸───  45%`` (unknown fraction renders ``n/a``)."""
    lab = label.ljust(label_width) if label_width else label
    bar_w = max(4, width - cell_width(lab) - 6)
    if fraction is None:
        return line((lab, "muted"), " ", progress_bar(None, bar_w), ("  n/a", "muted"))
    return line((lab, "muted"), " ", progress_bar(fraction, bar_w), " ", fmt_percent(fraction))


def box(lines: Sequence[Any], title: str | None = None, width: int = 40,
        height: int | None = None, footer: str | Line | None = None,
        title_right: Any = None, *, border: Any = "border") -> list[Line]:
    """Rounded box of exactly ``width`` cells (and ``height`` rows when given)."""
    try:
        width = max(4, int(width))
        inner = width - 4
        top_parts: list[Any] = [(G.box_tl + G.box_h, border)]
        used = 2
        if title:
            t = styled(f" {title} ", "title").clip(max(0, inner - 1))
            top_parts.append(t)
            used += t.width
        right = as_line(title_right) if title_right is not None else Line()
        if right and used + right.width + 3 <= width:
            fill = width - used - right.width - 3
            top_parts += [(G.box_h * fill + " ", border), right, (" " + G.box_tr, border)]
        else:
            top_parts.append((G.box_h * (width - used - 1) + G.box_tr, border))
        out = [line(*top_parts).crop(width)]
        body = [as_line(x) for x in lines]
        if height is not None:
            room = max(0, int(height) - 2)
            body = body[:room] + [Line()] * max(0, room - len(body))
        for b in body:
            out.append(line((G.box_v + " ", border), b.pad(inner, "<"),
                            (" " + G.box_v, border)))
        if footer is not None:
            f = as_line(footer).clip(max(0, inner - 1))
            out.append(line((G.box_bl + G.box_h, border), " ", f, " ",
                            (G.box_h * max(0, width - f.width - 5) + G.box_br, border)).crop(width))
        else:
            out.append(styled(G.box_bl + G.box_h * (width - 2) + G.box_br, border))
        return out
    except Exception as exc:  # noqa: BLE001
        return [_error_line("box", exc, width if isinstance(width, int) else None)]


def hstack(blocks: Sequence[Sequence[Line]], gap: int = 1,
           widths: Sequence[int] | None = None) -> list[Line]:
    """Place blocks side by side; each block is padded to its widest line (or ``widths``)."""
    if not blocks:
        return []
    ws = list(widths) if widths is not None else [
        max((as_line(x).width for x in b), default=0) for b in blocks]
    height = max(len(b) for b in blocks)
    out = []
    for r in range(height):
        parts: list[Any] = []
        for k, b in enumerate(blocks):
            cell = as_line(b[r]) if r < len(b) else Line()
            parts.append(cell.pad(ws[k], "<"))
            if k < len(blocks) - 1 and gap:
                parts.append(" " * gap)
        out.append(line(*parts))
    return out


def vstack(blocks: Sequence[Sequence[Line]], gap: int = 0) -> list[Line]:
    """Concatenate blocks vertically with ``gap`` blank lines between them."""
    out: list[Line] = []
    for k, b in enumerate(blocks):
        if k and gap:
            out.extend([Line()] * gap)
        out.extend(as_line(x) for x in b)
    return out


__all__ = [
    "Column",
    "badge",
    "box",
    "bucket",
    "ema",
    "fmt_percent",
    "fmt_ticks",
    "hstack",
    "kv_line",
    "line_chart",
    "meter",
    "nice_ticks",
    "progress_bar",
    "robust_range",
    "rule",
    "sparkline",
    "table",
    "vstack",
]
