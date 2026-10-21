"""``Verdict``: the shared result type of every filter, verifier and consistency check.

Evidence kinds, strongest first: ``construction`` > ``verifier`` >
``consistency`` > ``heuristic``. A record's lineage lists the
kinds it passed; reports show the share of the dataset backed by each.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EVIDENCE_KINDS = ("construction", "verifier", "consistency", "heuristic")


@dataclass
class Verdict:
    """``ok`` + optional ``score`` / ``reason`` / ``evidence`` details."""

    ok: bool
    score: float | None = None
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.ok)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": bool(self.ok)}
        if self.score is not None:
            out["score"] = self.score
        if self.reason:
            out["reason"] = self.reason
        if self.evidence:
            from topologies.experiments.store import jsonable

            out["evidence"] = jsonable(self.evidence)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Verdict:
        return cls(ok=bool(d.get("ok")), score=d.get("score"), reason=d.get("reason", ""),
                   evidence=dict(d.get("evidence") or {}))


def as_verdict(value: Any, *, reason: str = "") -> Verdict:
    """``Verdict`` from a bool, a number (> 0 passes), a ``(ok, reason)`` tuple, a dict or a
    ``Verdict``."""
    if isinstance(value, Verdict):
        return value
    if value is None:
        return Verdict(True, reason=reason)
    if isinstance(value, bool):
        return Verdict(value, reason=reason if not value else "")
    if isinstance(value, (int, float)):
        return Verdict(value > 0, score=float(value), reason=reason)
    if isinstance(value, tuple) and value:
        ok = bool(value[0])
        why = str(value[1]) if len(value) > 1 else reason
        return Verdict(ok, reason=why)
    if isinstance(value, dict) and "ok" in value:
        return Verdict.from_dict(value)
    return Verdict(bool(value), reason=reason)


def strongest(kinds: list[str]) -> str | None:
    """Strongest evidence kind in ``kinds`` (``None`` if empty)."""
    for k in EVIDENCE_KINDS:
        if k in kinds:
            return k
    return None


__all__ = ["EVIDENCE_KINDS", "Verdict", "as_verdict", "strongest"]
