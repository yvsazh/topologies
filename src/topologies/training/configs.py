"""Strict Trainer / optimizer configuration models (Pydantic v2)."""

from __future__ import annotations

import difflib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from topologies._util.suggest import did_you_mean
from topologies.training.errors import TrainingError

Mode = Literal["min", "max"]
Strategy = Literal["auto", "single", "ddp"]
_MODES = ("min", "max")
_STRATEGIES = ("auto", "single", "ddp")
_PRECISIONS = ("auto", "fp32", "fp16", "bf16")


def __getattr__(name: str) -> Any:
    # Stale literals: the registries in ``topologies.training.registries`` are the
    # source of truth (``list_optimizers()`` / ``list_schedulers()``).
    if name in ("OptimizerName", "SchedulerName"):
        import warnings

        warnings.warn(f"topologies.training.configs.{name} is deprecated; use "
                      "topologies.training.list_optimizers()/list_schedulers().",
                      DeprecationWarning, stacklevel=2)
        return str
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


PinMemory = bool | Literal["auto"] | None


def _suggest(name: str, valid: list[str]) -> str:
    hint = did_you_mean(str(name), valid).strip()
    return f"{hint + ' ' if hint else ''}Available: {', '.join(valid)}."


def _choice(value: Any, valid: tuple[str, ...], what: str) -> str:
    """Case-insensitive choice with a did-you-mean hint (typos never pass silently)."""
    lv = str(value).strip().lower()
    if lv not in valid:
        raise ValueError(f"Unknown {what} {value!r}. {_suggest(lv, list(valid))}")
    return lv


class OptimizerConfig(BaseModel):
    """Name-based optimizer (+ scheduler) configuration for one named optimizer.

    ``name`` / ``scheduler`` are validated against the registries
    (``register_optimizer`` / ``register_scheduler``). ``kwargs`` holds
    optimizer-specific extras. ``warmup_steps`` (or ``warmup_ratio`` of the
    total steps) adds a linear warmup in front of any scheduler; 0 = off.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.0
    # "matrices" decays only rank-2+ weights that are not embeddings.
    decay_policy: Literal["all", "matrices"] = "all"
    momentum: float = 0.9
    eps: float = 1e-8
    betas: tuple[float, float] | None = None
    scheduler: str | None = None
    scheduler_kwargs: dict[str, Any] = Field(default_factory=dict)
    clip_grad_norm: float | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)
    warmup_steps: int = Field(default=0, ge=0)
    warmup_ratio: float = Field(default=0.0, ge=0.0, lt=1.0)

    @field_validator("name")
    @classmethod
    def _name_valid(cls, v: str) -> str:
        from topologies.training.registries import list_optimizers

        lv = str(v).strip().lower()
        valid = list_optimizers()
        if lv not in valid:
            hint = _suggest(lv, valid)
            raise ValueError(f"Unknown optimizer {v!r}. {hint}")
        return lv

    @field_validator("scheduler")
    @classmethod
    def _sched_valid(cls, v: str | None) -> str | None:
        from topologies.training.registries import canonical_scheduler, list_schedulers

        if v is None:
            return None
        lv = str(v).strip().lower()
        if lv in ("null", "none", ""):
            return None
        valid = list_schedulers()
        if canonical_scheduler(lv) not in valid:
            hint = _suggest(lv, valid)
            raise ValueError(f"Unknown scheduler {v!r}. {hint}")
        return lv

    @field_validator("lr")
    @classmethod
    def _lr_valid(cls, v: float) -> float:
        if not (v > 0):
            raise ValueError("lr must be > 0")
        return float(v)


class CheckpointPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    every_epochs: int | None = None
    every_steps: int | None = None
    save_best: bool = False
    monitor: str = "val_loss"
    monitor_mode: Mode = "min"
    save_last: bool = True

    @field_validator("monitor_mode", mode="before")
    @classmethod
    def _mode_valid(cls, v: Any) -> str:
        return _choice(v, _MODES, "monitor_mode")


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: str = "jsonl"
    directory: str | None = None
    log_every_steps: int = 10
    filename: str | None = None

    @field_validator("log_every_steps")
    @classmethod
    def _every_valid(cls, v: int) -> int:
        if v < 1:
            raise ValueError("log_every_steps must be >= 1")
        return int(v)


class MonitoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "standard"
    refresh_rate: float = 10.0
    columns: list[str] | None = None
    charts: list[str] | None = None
    show_eta: bool = True
    show_mem: bool = True
    show_lr: bool = True
    precision: int = 3
    # Dashboard / live-view options (None = automatic).
    fullscreen: bool | None = None
    panels: list[str] | None = None
    view: str | None = None
    interactive: bool | None = None
    controls: bool | None = None
    capture_output: bool | None = None
    theme: str | None = None
    smoothing: float | None = None
    color: Literal["auto", "always", "never"] = "auto"


class TrainerConfig(BaseModel):
    """Generic execution/runtime configuration. No task-specific knowledge."""

    model_config = ConfigDict(extra="forbid")

    max_epochs: int = 1
    max_steps: int | None = None
    # Shortcuts over the recipe's optimizer policy (None = keep the policy value).
    lr: float | None = None
    weight_decay: float | None = None
    # Used only when datasets (not DataLoaders) are passed to Trainer.
    batch_size: int = 32
    num_workers: int = 0
    # Pinned host memory + non-blocking copies for loaders the Trainer builds;
    # "auto" (or None) pins only when the resolved device is CUDA.
    pin_memory: PinMemory = "auto"
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    device: str = "auto"
    devices: Any = "auto"
    strategy: Strategy = "auto"
    precision: str = "auto"
    accumulation_steps: int = 1
    clip_grad_norm: float | None = None
    clip_grad_value: float | None = None
    compile: bool = False
    compile_mode: str | None = None
    seed: int | None = 0
    eval_every_epochs: int | None = 1
    checkpoint: CheckpointPolicy = Field(default_factory=CheckpointPolicy)
    checkpoint_dir: str | None = None
    early_stopping_patience: int | None = None
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: Mode = "min"
    early_stopping_min_delta: float = 0.0
    log_every_steps: int = 10
    logger: str = "console"
    log_dir: str | None = None
    resume: Any = False
    trust_checkpoint: bool = False
    strict_batch: bool = True
    allow_ambiguous_recipe: bool = False
    monitor: str = "standard"
    monitoring: MonitoringConfig | None = None
    logging_cfg: LoggingConfig | None = None

    @field_validator("early_stopping_mode", mode="before")
    @classmethod
    def _es_mode_valid(cls, v: Any) -> str:
        return _choice(v, _MODES, "early_stopping_mode")

    @field_validator("strategy", mode="before")
    @classmethod
    def _strategy_valid(cls, v: Any) -> str:
        return _choice(v, _STRATEGIES, "strategy")

    @field_validator("precision", mode="before")
    @classmethod
    def _precision_valid(cls, v: Any) -> str:
        return _choice(v, _PRECISIONS, "precision")

    @field_validator("max_epochs")
    @classmethod
    def _epochs_valid(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_epochs must be >= 1")
        return int(v)

    @field_validator("lr")
    @classmethod
    def _lr_shortcut_valid(cls, v: float | None) -> float | None:
        if v is not None and not v > 0:
            raise ValueError("lr must be > 0")
        return v

    @field_validator("batch_size")
    @classmethod
    def _batch_valid(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size must be >= 1")
        return int(v)

    @field_validator("pin_memory")
    @classmethod
    def _pin_valid(cls, v: bool | str) -> bool | str:
        if isinstance(v, str) and v != "auto":
            raise ValueError("pin_memory must be True, False, or 'auto'")
        return v

    @field_validator("prefetch_factor")
    @classmethod
    def _prefetch_valid(cls, v: int | None) -> int | None:
        if v is None:
            return None
        if int(v) < 1:
            raise ValueError("prefetch_factor must be >= 1")
        return int(v)

    @field_validator("accumulation_steps")
    @classmethod
    def _accum_valid(cls, v: int) -> int:
        if v < 1:
            raise ValueError("accumulation_steps must be >= 1")
        return int(v)


def validate_with_hints(model_cls: type[BaseModel], values: dict) -> BaseModel:
    """Validate ``values`` against ``model_cls`` with typo hints on failure."""
    try:
        return model_cls(**values)
    except Exception as e:
        # Pydantic ValidationError: enrich extra-forbidden errors with suggestions.
        try:
            from pydantic import ValidationError as _VE

            if isinstance(e, _VE):
                fields = list(model_cls.model_fields.keys())
                notes: list[str] = []
                for err in e.errors():
                    loc = ".".join(str(x) for x in err.get("loc", ()))
                    if err.get("type") == "extra_forbidden":
                        sug = difflib.get_close_matches(loc, fields, n=3, cutoff=0.6)
                        if sug:
                            notes.append(f"Unknown parameter {loc!r}. Did you mean {sug[0]!r}?")
                        else:
                            notes.append(f"Unknown parameter {loc!r}.")
                        notes.append(
                            "Available parameters:\n"
                            + "\n".join(
                                f"- {k}: {model_cls.model_fields[k].annotation}"
                                for k in fields
                            )
                        )
                extra = "\n".join(notes)
                raise TrainingError(
                    f"Invalid configuration for {model_cls.__name__}: {e}\n{extra}".rstrip(),
                    stage="config",
                    hint=extra or str(e),
                ) from e
        except TrainingError:
            raise
        except Exception:  # noqa: BLE001, S110 - hint enrichment is best-effort; the plain error follows
            pass
        raise TrainingError(f"Invalid configuration: {e}", stage="config") from e
