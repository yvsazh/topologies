"""Registers the built-in adapters on first adapter resolution."""
from __future__ import annotations

from topologies.introspection.adapters import DEFAULT_ADAPTERS
from topologies.introspection.adapters.huggingface import HuggingFaceAdapter
from topologies.introspection.adapters.topologies_catalog import (
    CatalogTransformerAdapter,
    CatalogVisionAdapter,
)

DEFAULT_ADAPTERS.register(CatalogTransformerAdapter())
DEFAULT_ADAPTERS.register(CatalogVisionAdapter())
DEFAULT_ADAPTERS.register(HuggingFaceAdapter())

try:  # the diffusion adapter arrives with process trajectories
    from topologies.introspection.adapters.diffusion import DiffusionAdapter

    DEFAULT_ADAPTERS.register(DiffusionAdapter())
except ImportError:  # pragma: no cover
    pass
