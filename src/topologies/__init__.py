"""topologies: a library of modern neural-network architectures on PyTorch.

The most-used entry points are importable from the top level (loaded lazily):

    from topologies import Trainer, build, data, generate, load_pretrained

Subpackages: ``data`` (datasets/loaders), ``tokenizers``, ``nn`` (components
and stacks), ``build`` (presets, parameter budgets, validation),
``training`` (Trainer + recipes), ``models``/``layers`` (catalog),
``generation``, ``inference``, ``metrics``, ``pretrained``, ``experiments``
(``Experiment`` / ``experiment`` / ``compare``: comparisons over many trials),
``synth`` (synthetic data generation, measured by training on it),
``introspection`` (``introspect(model)``: what happens inside a network).
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

__version__ = "0.0.5"

# A library never configures output: applications attach handlers themselves.
logging.getLogger("topologies").addHandler(logging.NullHandler())

_LAZY: dict[str, tuple[str, str | None]] = {
    "Registry": ("topologies.registry.api", "Registry"),
    "Trainer": ("topologies.training", "Trainer"),
    "analyze": ("topologies.analysis", "analyze"),
    "analyze_training": ("topologies.analysis", "analyze_training"),
    "analyze_inference": ("topologies.analysis", "analyze_inference"),
    "analyze_hardware": ("topologies.analysis", "analyze_hardware"),
    "TrainingError": ("topologies.training", "TrainingError"),
    "build": ("topologies.build", None),  # callable package: build(...)
    "ConfigError": ("topologies.build", "ConfigError"),
    "generate": ("topologies.generation", "generate"),
    "stream_generate": ("topologies.generation", "stream_generate"),
    "load_pretrained": ("topologies.pretrained", "load_pretrained"),
    "save_pretrained": ("topologies.pretrained", "save_pretrained"),
    "train_tokenizer": ("topologies.tokenizers", "train_tokenizer"),
    "load_tokenizer": ("topologies.tokenizers", "load_tokenizer"),
    "predict": ("topologies.inference", "predict"),
    "data": ("topologies.data", None),
    "datasets": ("topologies.datasets", None),
    "list_datasets": ("topologies.datasets", "list_datasets"),
    "search_datasets": ("topologies.datasets", "search_datasets"),
    "get_dataset": ("topologies.datasets", "get_dataset"),
    "tokenizers": ("topologies.tokenizers", None),
    "nn": ("topologies.nn", None),
    "metrics": ("topologies.metrics", None),
    "models": ("topologies.models", None),
    "layers": ("topologies.layers", None),
    "training": ("topologies.training", None),
    "generation": ("topologies.generation", None),
    "inference": ("topologies.inference", None),
    "pretrained": ("topologies.pretrained", None),
    "Experiment": ("topologies.experiments", "Experiment"),
    "experiment": ("topologies.experiments", "experiment"),
    "compare": ("topologies.experiments", "compare"),
    "experiments": ("topologies.experiments", None),
    "synth": ("topologies.synth", None),
    "optimization": ("topologies.optimization", None),
    "evaluation": ("topologies.evaluation", None),
    "introspect": ("topologies.introspection.api", "introspect"),
    "introspection": ("topologies.introspection", None),
}

__all__ = ["__version__", *_LAZY]  # noqa: PLE0604

# Catalog packages are plain PyTorch; the framework adds list_*/get_* and lazy
# class attributes to them from the outside.
from topologies import _catalog

_catalog.install()


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'topologies' has no attribute {name!r}")
    module = importlib.import_module(target[0])
    value = module if target[1] is None else getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
