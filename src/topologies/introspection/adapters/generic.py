"""The fallback adapter: no semantics beyond ``torch.nn`` module-type inference."""
from __future__ import annotations

import torch.nn as nn

from topologies.introspection.adapters.torch_modules import TorchModulesAdapter


class GenericAdapter(TorchModulesAdapter):
    name = "generic"

    def matches(self, model: nn.Module) -> float:
        return 0.0

    def notes(self) -> list[str]:
        return [("no adapter: only torch.nn module-type axes are inferred; semantic roles are "
                 "unavailable until declared with insp.declare(...)")]
