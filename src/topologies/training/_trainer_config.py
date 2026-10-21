"""Trainer configuration merge: one path for TrainerConfig, dict and keyword inputs."""

from __future__ import annotations

from typing import Any

from topologies.training.configs import (
    LoggingConfig,
    MonitoringConfig,
    TrainerConfig,
    validate_with_hints,
)
from topologies.training.errors import TrainingError


def config_input(config: TrainerConfig | dict | None) -> dict[str, Any]:
    """The user's config input as a dict (used to tell explicit settings from defaults)."""
    if isinstance(config, TrainerConfig):
        return config.model_dump(mode="python")
    return dict(config or {})


def merge_trainer_config(config: TrainerConfig | dict | None, trainer_kwargs: dict[str, Any],
                         *, allow_ambiguous_recipe: bool = False,
                         monitoring: Any = None, logging_cfg: Any = None) -> TrainerConfig:
    """``base = config``, then keyword overrides, then the direct monitoring/logging configs.

    Precedence: explicit kwargs > ``monitoring=`` / ``logging_cfg=`` > ``config``.
    Validation errors carry typo hints (``validate_with_hints``).
    """
    merged = {**config_input(config), **trainer_kwargs}
    if allow_ambiguous_recipe:
        merged["allow_ambiguous_recipe"] = True
    try:
        cfg = TrainerConfig(**merged)
    except (TypeError, ValueError):  # pydantic ValidationError is a ValueError
        cfg = validate_with_hints(TrainerConfig, merged)  # type: ignore[assignment]
    if monitoring is not None:
        cfg.monitoring = _direct(monitoring, MonitoringConfig, "monitoring config")
    if logging_cfg is not None:
        cfg.logging_cfg = _direct(logging_cfg, LoggingConfig, "logging_cfg")
    return cfg


def _direct(value: Any, model: type, what: str) -> Any:
    if isinstance(value, model):
        return value
    if isinstance(value, dict):
        return validate_with_hints(model, dict(value))
    raise TrainingError(f"Invalid {what}.", stage="config", received=type(value).__name__)
