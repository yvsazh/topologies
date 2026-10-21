"""Centralized quality gate: the ONLY place that may declare a candidate verified."""

from __future__ import annotations

from dataclasses import dataclass, field

from topologies.enums import GateDecisionType
from topologies.metadata.schema import now_iso
from topologies.validation.framework import CheckResult


@dataclass
class GateDecision:
    decision: GateDecisionType
    reasons: list[str] = field(default_factory=list)
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    evaluated_at: str = ""
    evaluator: str = "quality-gate"

    def __post_init__(self) -> None:
        if not self.evaluated_at:
            self.evaluated_at = now_iso()

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "reasons": self.reasons,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "evaluated_at": self.evaluated_at,
            "evaluator": self.evaluator,
        }


class QualityGate:
    """Deterministic gate over check results + verification evidence.

    Rules (conservative by design):
    - any failed ERROR check -> quarantine (or reject if license/paper missing)
    - only warnings -> experimental
    - all passed + reviewed + source_match ok -> accept
    - clean-room implementations additionally require explicit source-review
      and independent-math checks; they are not reported as official-code matches
    - otherwise needs_review
    """

    REJECT_REASONS = {"license incompatible", "paper unavailable"}

    @classmethod
    def evaluate(
        cls,
        results: list[CheckResult],
        *,
        reviewed: bool = False,
        source_match: str = "unknown",
        rejection_reason: str = "",
    ) -> GateDecision:
        failed = [r for r in results if not r.passed and r.severity == "error"]
        warnings = [r for r in results if not r.passed and r.severity == "warning"]
        passed = [r.name for r in results if r.passed]
        failed_names = [r.name for r in failed]
        reasons = [r.message for r in failed + warnings]

        if rejection_reason.strip().lower() in cls.REJECT_REASONS:
            return GateDecision(GateDecisionType.REJECT, [f"hard rejection: {rejection_reason}", *reasons], passed, failed_names)
        if failed:
            return GateDecision(GateDecisionType.QUARANTINE, ["error checks failed", *reasons], passed, failed_names)
        clean_room_evidence = {"source_review", "independent_math"}.issubset(passed)
        source_is_verified = source_match == "matches-official" or (
            source_match == "clean-room-verified" and clean_room_evidence
        )
        if not reviewed or not source_is_verified:
            reason = "requires human review and verified source/equation evidence"
            if source_match == "clean-room-verified" and not clean_room_evidence:
                reason = "clean-room acceptance requires source_review and independent_math checks"
            return GateDecision(
                GateDecisionType.NEEDS_REVIEW,
                [reason, *reasons],
                passed, failed_names,
            )
        if warnings:
            return GateDecision(GateDecisionType.EXPERIMENTAL, ["warnings only", *reasons], passed, failed_names)
        accepted_reason = ("all checks passed, reviewed, official source matched"
                           if source_match == "matches-official" else
                           "all checks passed, reviewed, clean-room equations independently verified")
        return GateDecision(GateDecisionType.ACCEPT, [accepted_reason], passed, failed_names)
