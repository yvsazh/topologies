"""Structured experiment errors (mirrors ``TrainingError``: message + context + hint)."""

from __future__ import annotations


class ExperimentError(Exception):
    """Invalid experiment definition or a trial-level problem, with the fix spelled out.

    ``stage`` names where it happened (``"routing"``, ``"space"``, ``"model"``,
    ``"tokenizer"``, ``"data"``, ``"plan"``, ...); ``hint`` says what to do.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        hint: str | None = None,
        key: str | None = None,
        trial: str | None = None,
        available: list[str] | None = None,
    ) -> None:
        parts = [message]
        if stage:
            parts.append(f"Stage: {stage}")
        if key:
            parts.append(f"Key: {key}")
        if trial:
            parts.append(f"Trial: {trial}")
        if available:
            shown = list(available)[:40]
            more = len(available) - len(shown)
            parts.append("Available:\n" + "\n".join(f"- {v}" for v in shown)
                         + (f"\n- ... ({more} more)" if more > 0 else ""))
        if hint:
            parts.append(f"Hint: {hint}")
        super().__init__("\n".join(parts))
        self.message = message
        self.stage = stage
        self.hint = hint
        self.key = key
        self.trial = trial
        self.available = list(available or [])


class Pruned(Exception):
    """Raised by ``ctx.report`` (or the pruning callback) when a trial should stop early."""

    def __init__(self, reason: str = "pruned") -> None:
        super().__init__(reason)
        self.reason = reason


class TrialTimeout(Exception):
    """Raised by ``ctx.report`` when ``trial_timeout`` is exceeded in a custom loop."""


__all__ = ["ExperimentError", "Pruned", "TrialTimeout"]
