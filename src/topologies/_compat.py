"""Deprecated base classes kept for external code written against older releases.

Catalog classes (``topologies.models`` / ``topologies.layers``) subclass
``torch.nn.Module`` directly and never import ``topologies``; nothing in the
framework requires these bases. Reach them as ``topologies.models.TopologiesModel``
/ ``topologies.layers.TopologiesLayer`` (emits ``DeprecationWarning``).
"""

from __future__ import annotations

import torch.nn as nn

from topologies.build.config import Configurable


class TopologiesModel(Configurable, nn.Module):
    """Deprecated: subclass ``nn.Module`` (plus ``Configurable`` if wanted)."""

    architecture_id: str = ""
    is_demo: bool = False


class TopologiesLayer(nn.Module):
    """Deprecated: subclass ``nn.Module``."""

    component_id: str = ""
    is_demo: bool = False
