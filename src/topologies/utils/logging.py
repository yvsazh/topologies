"""Structured logging for humans and autonomous runs. Never log secrets."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_SENSITIVE = ("token", "secret", "password", "api_key", "apikey")


def _scrub(d: dict) -> dict:
    return {k: ("***" if any(s in k.lower() for s in _SENSITIVE) else v) for k, v in d.items()}


def get_logger(name: str = "topologies", level: int = logging.INFO) -> logging.Logger:
    """A logger with ``level``; output handlers are the application's choice.

    The library never attaches handlers (``topologies`` has a ``NullHandler``);
    command-line tools call ``add_stream_handler`` themselves.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def add_stream_handler(logger: logging.Logger, stream=None) -> logging.Logger:
    """Attach one stdout handler (for CLI entry points, not library code)."""
    if not any(getattr(h, "_topologies_cli", False) for h in logger.handlers):
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        handler._topologies_cli = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, stage: str, candidate: str = "", status: str = "", **fields) -> None:
    payload = {"ts": datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z", "stage": stage, "candidate": candidate, "status": status, **_scrub(fields)}
    logger.info(json.dumps(payload))
