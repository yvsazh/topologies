"""Single formatting authority for numbers, durations, sizes and metric values.

Every display surface (terminal views, Matplotlib tick formatters, HTML) formats
through these functions so the same value always looks the same. Pure, stdlib-only,
never raises: unformattable input degrades to ``str(value)``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence
from typing import Any

MISSING = "—"
_LR_NAMES = frozenset({"lr", "learning_rate"})


def _unicode_default() -> bool:
    return os.environ.get("TOPOLOGIES_ASCII", "") in ("", "0")


def _as_float(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _decimals(f: float, sig: int) -> int:
    """Decimals needed to show ``sig`` significant digits of a mid-range ``f``."""
    if f == 0:
        return max(0, sig - 1)
    exp = math.floor(math.log10(abs(f)))
    dec = max(0, sig - 1 - exp)
    # 9.9996 with 3 decimals rounds to 10.000: one digit more on the left.
    if abs(round(f, dec)) >= 10 ** (exp + 1):
        dec = max(0, dec - 1)
    return min(dec, 12)


def _is_sci(f: float) -> bool:
    a = abs(f)
    return a != 0 and (a < 1e-3 or a >= 1e6)


def _special(f: float, unicode: bool) -> str | None:
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        inf = "∞" if unicode else "inf"
        return inf if f > 0 else f"-{inf}"
    return None


def fmt_number(x: Any, *, sig: int = 4, unicode: bool | None = None) -> str:
    """``None`` → ``n/a``; scientific outside [1e-3, 1e6); else ``sig`` significant digits.

    Python ints are exact counts and stay integers (``354``, ``1021440``); floats keep
    trailing zeros (``0.6100``) so a column aligns with :func:`fmt_column`.
    """
    unicode = _unicode_default() if unicode is None else unicode
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    f = _as_float(x)
    if f is None:
        return str(x)
    special = _special(f, unicode)
    if special is not None:
        return special
    if f == 0:
        return "0"
    if _is_sci(f):
        return f"{f:.1e}"
    return f"{f:.{_decimals(f, max(1, int(sig)))}f}"


def fmt_column(values: Sequence[Any] | Iterable[Any], *, sig: int = 4,
               missing: str = MISSING, unicode: bool | None = None) -> list[str]:
    """Format a numeric column with one shared number of decimals (``0.6660``, ``0.6100``).

    Values outside the mid range use scientific notation, integers stay integers when
    the whole column is integral, missing values render as ``missing``.
    """
    unicode = _unicode_default() if unicode is None else unicode
    vals = list(values)
    floats = [_as_float(v) for v in vals]
    mid = [f for v, f in zip(vals, floats)
           if f is not None and math.isfinite(f) and not _is_sci(f) and f != 0
           and not isinstance(v, int)]
    all_int = all(isinstance(v, int) and not isinstance(v, bool) for v in vals if v is not None)
    dec = 0 if all_int else max((_decimals(f, sig) for f in mid), default=0)
    dec = min(dec, 6)
    out = []
    for v, f in zip(vals, floats):
        if v is None:
            out.append(missing)
        elif f is None:
            out.append(str(v))
        elif (special := _special(f, unicode)) is not None:
            out.append(special)
        elif isinstance(v, int):
            out.append(str(v))
        elif _is_sci(f):
            out.append(f"{f:.1e}")
        elif all_int or (f == 0 and dec == 0):
            out.append(str(int(f)) if f == int(f) else f"{f:.{dec}f}")
        else:
            out.append(f"{f:.{dec}f}")
    return out


def fmt_lr(x: Any) -> str:
    """Learning rates: always scientific with two significant digits, never ``0``."""
    f = _as_float(x)
    if f is None:
        return "n/a" if x is None else str(x)
    if math.isnan(f) or math.isinf(f):
        return fmt_number(f)
    return f"{f:.1e}"


def _metric_info(name: str) -> Any:
    try:
        from topologies.experiments.metrics import metric_info

        return metric_info(name)
    except Exception:  # noqa: BLE001
        return None


def fmt_metric(name: str, value: Any, *, sig: int = 4) -> str:
    """Display a metric value: ``lr`` scientific, registered metrics via their format."""
    key = str(name or "")
    base = key.split(":")[-1]
    if base in _LR_NAMES or base.endswith("_lr"):
        return fmt_lr(value)
    if value is None:
        return "n/a"
    info = _metric_info(base)
    if info is not None and not getattr(info, "inferred", True):
        try:
            return str(info.format(value))
        except Exception:  # noqa: BLE001, S110
            pass
    return fmt_number(value, sig=sig)


def fmt_duration(seconds: Any) -> str:
    """Wall durations: ``0.3s``, ``42s``, ``3:07``, ``1:02:03``, ``2d 04h``; None → ``--:--``."""
    f = _as_float(seconds)
    if f is None or not math.isfinite(f):
        return "--:--"
    f = max(0.0, f)
    if f < 10:
        return f"{f:.1f}s"
    if f < 60:
        return f"{round(f)}s"
    s = round(f)
    if s < 3600:
        return f"{s // 60}:{s % 60:02d}"
    if s < 86400:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 86400}d {(s % 86400) // 3600:02d}h"


def fmt_clock(seconds: Any) -> str:
    """Elapsed/remaining clock time: ``0:04``, ``3:07``, ``1:02:03``; None → ``--:--``."""
    f = _as_float(seconds)
    if f is None or not math.isfinite(f):
        return "--:--"
    s = int(max(0.0, f))
    if s < 3600:
        return f"{s // 60}:{s % 60:02d}"
    if s < 86400:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 86400}d {(s % 86400) // 3600:02d}h"


def fmt_eta(seconds: Any, *, unicode: bool | None = None) -> str:
    """Approximate remaining time: ``≈42s``, ``≈3m``, ``≈1h12m``; None → ``?``."""
    unicode = _unicode_default() if unicode is None else unicode
    f = _as_float(seconds)
    if f is None or not math.isfinite(f):
        return "?"
    s = int(max(0.0, f))
    approx = "≈" if unicode else "~"
    if s >= 3600:
        return f"{approx}{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{approx}{s // 60}m"
    return f"{approx}{s}s"


def fmt_count(n: Any) -> str:
    """Counts: ``354``, ``12.3K``, ``1.10M``, ``7.0B``; None → ``n/a``."""
    f = _as_float(n)
    if f is None or not math.isfinite(f):
        return "n/a" if n is None else str(n)
    a = abs(f)
    if a < 1000:
        return str(int(f)) if f == int(f) else f"{f:.3g}"
    if a < 1e6:
        return f"{f / 1e3:.1f}K"
    if a < 1e9:
        return f"{f / 1e6:.2f}M"
    if a < 1e12:
        return f"{f / 1e9:.1f}B"
    return f"{f / 1e12:.1f}T"


def fmt_bytes(n: Any, *, binary: bool = True) -> str:
    """Sizes: ``512 B``, ``1.5 KiB``, ``0.29 GiB`` (≥ 256 MiB shows in GiB)."""
    f = _as_float(n)
    if f is None or not math.isfinite(f):
        return "n/a" if n is None else str(n)
    base = 1024.0 if binary else 1000.0
    units = (["B", "KiB", "MiB", "GiB", "TiB", "PiB"] if binary
             else ["B", "kB", "MB", "GB", "TB", "PB"])
    a = abs(f)
    if a < base:
        return f"{int(f)} B" if f == int(f) else f"{f:.1f} B"
    i = 0
    value = f
    while i < len(units) - 1 and (abs(value) >= base or (i >= 2 and abs(value) >= base / 4)):
        value /= base
        i += 1
    a = abs(value)
    digits = 2 if a < 1 else (1 if a < 100 else 0)
    return f"{value:.{digits}f} {units[i]}"


def fmt_rate(x: Any, unit: str = "it") -> str:
    """Rates: ``142 it/s``, ``1.2K rec/s``; below 1/s inverted: ``0.8 s/it``."""
    f = _as_float(x)
    if f is None or not math.isfinite(f) or f <= 0:
        return f"? {unit}/s"
    if f < 1:
        inv = 1.0 / f
        return f"{inv:.1f} s/{unit}" if inv < 100 else f"{inv:.0f} s/{unit}"
    if f >= 1000:
        return f"{fmt_count(f)} {unit}/s"
    if f >= 100:
        return f"{f:.0f} {unit}/s"
    if f >= 10:
        return f"{f:.1f} {unit}/s"
    return f"{f:.2f} {unit}/s"


def fmt_delta(value: Any, reference: Any, mode: str = "relative", *,
              unicode: bool | None = None) -> str:
    """Signed change: ``+4.2 %``, ``−0.013`` (tiny reference), ``×24`` beyond ±999 %."""
    unicode = _unicode_default() if unicode is None else unicode
    v, r = _as_float(value), _as_float(reference)
    if v is None or r is None or not (math.isfinite(v) and math.isfinite(r)):
        return MISSING if unicode else "-"
    minus = "−" if unicode else "-"
    times = "×" if unicode else "x"

    def signed(text: str, neg: bool) -> str:
        return (minus if neg else "+") + text

    diff = v - r
    if mode == "absolute" or abs(r) < 1e-12:
        if diff == 0:
            return "0"
        return signed(fmt_number(abs(diff), sig=2, unicode=unicode), diff < 0)
    pct = diff / abs(r) * 100.0
    if abs(pct) > 999:
        ratio = v / r if r else float("inf")
        if not math.isfinite(ratio) or ratio <= 0:
            return signed(fmt_number(abs(diff), sig=2, unicode=unicode), diff < 0)
        shown = f"{ratio:.0f}" if ratio >= 10 else f"{ratio:.1f}"
        return f"{times}{shown}"
    if pct == 0:
        return "0 %"
    return signed(f"{abs(pct):.1f} %", pct < 0)


def direction(name: str) -> str:
    """``"min"`` or ``"max"`` for a metric name (registry, else name heuristic)."""
    info = _metric_info(str(name).split(":")[-1])
    if info is not None:
        return "max" if info.mode == "max" else "min"
    import re

    return "max" if re.search(r"(acc|f1|auc|bleu|rouge|_per_s|throughput|score)",
                              str(name), re.IGNORECASE) else "min"


def is_better(name: str, a: Any, b: Any) -> bool:
    """True when ``a`` is strictly better than ``b`` for metric ``name``."""
    fa, fb = _as_float(a), _as_float(b)
    if fa is None or not math.isfinite(fa):
        return False
    if fb is None or not math.isfinite(fb):
        return True
    return fa > fb if direction(name) == "max" else fa < fb


__all__ = [
    "MISSING",
    "direction",
    "fmt_bytes",
    "fmt_clock",
    "fmt_column",
    "fmt_count",
    "fmt_delta",
    "fmt_duration",
    "fmt_eta",
    "fmt_lr",
    "fmt_metric",
    "fmt_number",
    "fmt_rate",
    "is_better",
]
