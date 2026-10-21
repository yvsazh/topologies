"""Logger factory (disk loggers only; terminal is a monitor concern)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from topologies.training.errors import TrainingError
from topologies.training.logging.base import BaseLogger, NullLogger
from topologies.training.logging.jsonl import JsonlLogger
from topologies.training.logging.tabular import CSVLogger, TensorBoardLogger
from topologies.training.logging.text import TextLogger

_AVAILABLE = ["none", "jsonl", "csv", "tensorboard", "txt", "file", "null", "off"]


def build_logger(
    kind: Any = None,
    log_dir: str | None = None,
    rank: int = 0,
    log_every_steps: int = 10,
    filename: str | None = None,
) -> BaseLogger:
    """Build a disk logger. Never renders terminal progress.

    Accepts None / "none" / "null" / "off" / "" -> NullLogger,
    "jsonl" -> JsonlLogger, "csv" -> CSVLogger, "tensorboard" / "tb" ->
    TensorBoardLogger (needs the ``tensorboard`` extra), "txt" / "file" -> TextLogger,
    BaseLogger instance -> returned as-is.
    Legacy "console" maps to no disk logger (monitor concern).
    """
    if kind is None:
        return NullLogger()
    if isinstance(kind, BaseLogger):
        try:
            kind.rank = rank
        except Exception:  # noqa: BLE001, S110
            pass
        return kind
    if not isinstance(kind, str):
        raise TrainingError(
            f"Unknown logger {kind!r}.",
            stage="config",
            available=_AVAILABLE + ["console", "BaseLogger instance"],
            hint="Use logger='none'|'jsonl'|'csv'|'tensorboard'|'txt'|BaseLogger(...).",
        )
    k = kind.strip().lower()
    if k in ("none", "null", "off", ""):
        return NullLogger()
    if k == "console":
        # Legacy console means terminal monitoring, not disk logging.
        return NullLogger()
    if k == "jsonl":
        base = Path(log_dir or "runs") / (filename or "metrics.jsonl")
        return JsonlLogger(base, rank=rank)
    if k == "csv":
        return CSVLogger(Path(log_dir or "runs") / (filename or "metrics.csv"), rank=rank)
    if k in ("tensorboard", "tb"):
        return TensorBoardLogger(Path(log_dir or "runs") / (filename or "tensorboard"),
                                 rank=rank)
    if k in ("file", "txt"):
        base = Path(log_dir or "runs") / (filename or "training.log")
        return TextLogger(base, rank=rank)
    raise TrainingError(
        f"Unknown logger {kind!r}.",
        stage="config",
        available=_AVAILABLE + ["console"],
        hint="Use logger='none'|'jsonl'|'csv'|'tensorboard'|'txt'|BaseLogger(...).",
    )
