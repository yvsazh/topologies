"""Blueprints: ready-made, editable specs for common task types (§14 of docs/synth.md).

    spec = synth.blueprints.tabular("fraud", size=20_000)

Each returns a normal ``Spec`` (inspect it, ``with_(...)`` it, ``to_yaml()`` it) with
recommended attributes, generators, verifiers, dedup, balancing and splits.
``names()`` lists them; ``get(name)`` returns one; ``example(name)`` gives the
options ``topologies synth init`` scaffolds. Every blueprint has a page in
``docs/synth/blueprints/<name>.md``.
"""

from __future__ import annotations

import importlib
from typing import Any

from topologies.synth.errors import unknown

_MODULES = {
    "tabular": "structured", "time_series": "structured",
    "algorithmic_probes": "reasoning",
    "image_classification": "vision", "detection": "vision", "ocr_lines": "vision",
    "text_classification": "algorithmic", "ner": "algorithmic", "span_extraction": "algorithmic",
    "structured_extraction": "algorithmic", "seq2seq_transform": "algorithmic",
    "math_word_problems": "algorithmic", "lm_corpus": "algorithmic",
    "audio_classification": "algorithmic", "pitch_estimation": "algorithmic",
    "graph_tasks": "algorithmic",
}


def names() -> list[str]:
    """All blueprint names (aliases included)."""
    return sorted(_MODULES)


def get(name: str) -> Any:
    """The blueprint function ``name`` (``fn(spec_name, **options) -> Spec``)."""
    key = str(name).replace("-", "_")
    if key not in _MODULES:
        raise unknown("blueprint", str(name), names(), stage="blueprint")
    importlib.import_module(f"topologies.synth.blueprints.{_MODULES[key]}")
    from topologies.synth.blueprints._base import BLUEPRINTS

    return BLUEPRINTS[key]


def example(name: str) -> dict[str, Any]:
    """Example options of blueprint ``name`` (what ``synth init`` writes)."""
    return dict(get(name).example)


def __getattr__(name: str) -> Any:
    if name in _MODULES:
        return get(name)
    raise AttributeError(f"module 'topologies.synth.blueprints' has no attribute {name!r}")


def __dir__() -> list[str]:
    return [*names(), "example", "get", "names"]


__all__ = ["algorithmic_probes", "audio_classification", "detection", "example", "get",
           "graph_tasks", "image_classification", "lm_corpus", "math_word_problems", "names",
           "ner", "ocr_lines", "pitch_estimation", "seq2seq_transform", "span_extraction",
           "structured_extraction", "tabular", "text_classification", "time_series"]
