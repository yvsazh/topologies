"""Deprecated compatibility shim. Use topologies.training.logging instead."""

from __future__ import annotations

# Re-export new implementation; keep old names working.
from topologies.training.logging.base import (
    BaseLogger,
    ConsoleLogger,
    NullLogger,
)
from topologies.training.logging.factory import build_logger
from topologies.training.logging.jsonl import JsonlLogger
from topologies.training.logging.text import FileLogger, TextLogger

__all__ = [
    "BaseLogger",
    "ConsoleLogger",
    "FileLogger",
    "JsonlLogger",
    "NullLogger",
    "TextLogger",
    "build_logger",
]
