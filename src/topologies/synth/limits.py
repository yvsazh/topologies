"""Run limits: wall-clock time (``max_time``) and raw records (``max_raw``).

    parse_time("30m")      # 1800.0 seconds
    parse_count("20k")     # 20000.0
    clock = RunClock(max_time=60); clock.check()   # None, or the reason the run must stop

Both limits stop a run cleanly: finished chunks are kept and the partial
version is valid, so ``spec.with_(max_time="2h").run()`` resumes it.
"""

from __future__ import annotations

import re
import time
from typing import Any

from topologies.synth.errors import SynthError

_SI = {"": 1, "k": 1e3, "m": 1e6, "b": 1e9}


def parse_count(v: Any) -> float:
    """``20_000`` / ``"20k"`` / ``"1.5M records"`` -> a number."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([kKmMbB]?)\s*[a-zA-Z]*\s*$", str(v))
    if not m:
        raise SynthError(f"Invalid count {v!r}.", stage="spec",
                         hint="Use a number such as 20_000, '20k' or '1.5M'.")
    return float(m.group(1)) * _SI[m.group(2).lower()]


def parse_time(v: Any) -> float | None:
    """``None`` / seconds / ``"90s"`` / ``"30m"`` / ``"2h"`` -> seconds."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if v <= 0:
            raise SynthError("max_time must be > 0.", stage="spec")
        return float(v)
    from topologies.experiments.routing import parse_duration

    try:
        out = float(parse_duration(str(v)))
    except Exception:  # noqa: BLE001
        raise SynthError(f"Invalid max_time {v!r}.", stage="spec",
                         hint="Use seconds (90) or a duration such as '30m' or '2h'.") from None
    if out <= 0:
        raise SynthError("max_time must be > 0.", stage="spec")
    return out


def fmt_seconds(s: float | None) -> str:
    if s is None:
        return "—"
    from topologies.experiments.table import fmt_seconds as _fmt

    return _fmt(s)


class RunClock:
    """Elapsed time of one run, checked against ``max_time``."""

    def __init__(self, max_time: float | None = None, *, clock: Any = None) -> None:
        self.max_time = max_time
        self.clock = clock or time.monotonic
        self.t0 = self.clock()

    @property
    def elapsed(self) -> float:
        return self.clock() - self.t0

    def check(self) -> str | None:
        """The reason to stop (``max_time`` reached) or ``None``."""
        if self.max_time is not None and self.elapsed >= self.max_time:
            return f"max_time {self.max_time:g}s reached"
        return None

    def as_dict(self) -> dict[str, Any]:
        return {"elapsed_s": round(self.elapsed, 3), "max_time": self.max_time}


__all__ = ["RunClock", "fmt_seconds", "parse_count", "parse_time"]
