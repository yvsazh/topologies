"""Structured synth errors: message + stage + the fix (``hint``), like ``ExperimentError``.

``SynthError`` subclasses ``ExperimentError`` so synth problems raised inside
an experiment trial are classified and reported the same way. Unknown names
(node parameters, paths, attributes, types, library factories) come with a
``Did you mean ...?`` suggestion:

    raise unknown("node parameter", "seconsd", ["seconds", "sr"], stage="node")
    # Unknown node parameter 'seconsd'. ... Hint: Did you mean 'seconds'?
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable

from topologies.experiments.errors import ExperimentError


class SynthError(ExperimentError):
    """Invalid spec / node / op use, with the fix spelled out in ``hint``.

    ``stage`` names where it happened (``"spec"``, ``"node"``, ``"attributes"``,
    ``"generator"``, ``"pipeline"``, ``"store"``, ``"types"``, ...).
    """


class RunStopped(SynthError):
    """A condition that stops the whole run cleanly (finished chunks are kept).

    ``reason`` becomes the version's ``stop_reason`` (``"max_time"``, ``"stopped"``).
    """

    reason = "stopped"


class TimeLimit(RunStopped):
    """``max_time`` was reached; the partial dataset is valid and the run resumes."""

    reason = "max_time"


class ParseError(SynthError):
    """A value could not be parsed into the schema."""

    def __init__(self, message: str, *, raw: str | None = None, hint: str | None = None) -> None:
        super().__init__(message, stage="parse", hint=hint)
        self.raw = raw


def suggest(name: str, options: Iterable[str], *, cutoff: float = 0.6) -> str | None:
    """Closest option to ``name`` (difflib) or ``None``."""
    sug = difflib.get_close_matches(str(name), sorted({str(o) for o in options}), n=1,
                                    cutoff=cutoff)
    return sug[0] if sug else None


def unknown(what: str, name: str, options: Iterable[str], *, stage: str,
            extra: str = "") -> SynthError:
    """``SynthError`` for an unknown name with a ``Did you mean`` hint."""
    opts = sorted({str(o) for o in options})
    sug = suggest(name, opts)
    hint = (f"Did you mean {sug!r}? " if sug else "") + extra
    return SynthError(f"Unknown {what} {name!r}.", stage=stage, key=str(name),
                      hint=hint.strip() or None, available=opts)


__all__ = ["ParseError", "RunStopped", "SynthError", "TimeLimit", "suggest", "unknown"]
