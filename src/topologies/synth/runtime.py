"""Per-run services shared by generators and ops: the store, the clock, events.

One ``Runtime`` exists per ``spec.run()`` / ``spec.plan()``. It owns the run
clock (``max_time``) and the event log; ``setup(runtime)`` hooks of nodes and
ops receive it (``runtime.dir`` is the dataset directory, ``runtime.spec`` the
spec being run).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from topologies.synth.limits import RunClock


class Runtime:
    """Services for one run (see module docstring)."""

    def __init__(self, spec: Any, store: Any = None, *, clock: Any = None) -> None:
        self.spec = spec
        self.store = store
        self.dir: Path | None = store.dir if store is not None else None
        self.clock = RunClock(getattr(spec, "max_time", None), clock=clock)
        self.events: list[dict[str, Any]] = []

    def event(self, kind: str, **data: Any) -> None:
        self.events.append({"event": kind, **data})
        if self.store is not None:
            self.store.log_event(kind, **data)


__all__ = ["Runtime"]
