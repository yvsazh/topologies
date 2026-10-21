"""Compatibility exports for pack-backed language pools."""
from __future__ import annotations

from typing import Any

from topologies import synth


def personas(n: int = 200, **kwargs: Any):
    """Legacy persona attribute pool backed by ``synth.text`` pack data."""
    return synth.text.persona_pool(n, **kwargs)


def topics(n: int = 50, **kwargs: Any):
    """Legacy topic attribute pool backed by ``synth.text`` pack data."""
    return synth.text.topic_pool(n, **kwargs)


__all__ = ["personas", "topics"]
