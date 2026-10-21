"""Structured training errors with actionable context."""

from __future__ import annotations


class TrainingError(Exception):
    """Structured error raised at meaningful training stage boundaries.

    Carries machine-readable context (recipe, stage, expected/received, hints)
    while preserving the original traceback via ``raise ... from``.
    """

    def __init__(
        self,
        message: str,
        *,
        recipe: str | None = None,
        stage: str | None = None,
        expected: str | None = None,
        received: str | None = None,
        device: str | None = None,
        precision: str | None = None,
        optimizer_name: str | None = None,
        component: str | None = None,
        config_key: str | None = None,
        hint: str | None = None,
        available: list[str] | None = None,
    ) -> None:
        parts = [message]
        if recipe:
            parts.append(f"Recipe: {recipe}")
        if stage:
            parts.append(f"Stage: {stage}")
        if expected:
            parts.append(f"Expected:\n{expected}")
        if received:
            parts.append(f"Received:\n{received}")
        if device or precision:
            parts.append(f"Device: {device or '?'}  Precision: {precision or '?'}")
        if optimizer_name:
            parts.append(f"Optimizer: {optimizer_name}")
        if component:
            parts.append(f"Component: {component}")
        if config_key:
            parts.append(f"Config key: {config_key}")
        if available:
            parts.append("Available values:\n" + "\n".join(f"- {v}" for v in available))
        if hint:
            parts.append(f"Hint: {hint}")
        super().__init__("\n".join(parts))
        self.recipe = recipe
        self.stage = stage
        self.expected = expected
        self.received = received
        self.device = device
        self.precision = precision
        self.optimizer_name = optimizer_name
        self.component = component
        self.config_key = config_key
        self.hint = hint
        self.available = list(available or [])
