"""Semantics that follow from ``torch.nn`` layer types alone (mixed into every adapter).

Axis inference for the layer types lives in ``semantics.infer_axes``. This adapter
contributes attention sites: ``nn.MultiheadAttention`` (weights captured by
forcing ``need_weights``) and modules whose ``component_id`` says they call SDPA.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from topologies.introspection.adapters import Adapter, AttentionSite

SDPA_COMPONENTS = ("multi-head-attention",)


class TorchModulesAdapter(Adapter):
    name = "torch-modules"

    def matches(self, model: nn.Module) -> float:
        return 0.0

    def attention_sites(self, index: Any) -> list[AttentionSite]:
        sites = []
        for path in index.paths():
            module = index.module(path)
            if isinstance(module, nn.MultiheadAttention):
                sites.append(AttentionSite(path, "mha", module.num_heads,
                                           "weights via need_weights=True (math path)"))
            elif getattr(type(module), "component_id", None) in SDPA_COMPONENTS:
                sites.append(AttentionSite(path, "sdpa", getattr(module, "heads", None),
                                           "weights recomputed from scaled_dot_product_attention"))
        return sites
