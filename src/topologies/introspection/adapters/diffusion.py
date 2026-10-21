"""Sampler-aware adapter for denoisers used with ``inference_tasks.diffusion``.

The whole model is the ``denoiser``. Sampling steps are its occurrences, and the
samplers' ``callback=`` provides x_t and the predicted x0. Register it by name with
``introspect(unet, adapter="diffusion")``. It also matches the catalog
``DiffusionUNet`` automatically.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from topologies.introspection.adapters import ProcessSpec
from topologies.introspection.adapters.torch_modules import TorchModulesAdapter
from topologies.introspection.semantics import Axes


class DiffusionAdapter(TorchModulesAdapter):
    name = "diffusion"
    priority = 5

    def matches(self, model: nn.Module) -> float:
        return 1.0 if type(model).__name__ == "DiffusionUNet" else 0.0

    def roles(self, index: Any) -> dict[str, list[str]]:
        roles = {"denoiser": ["<root>"]}
        for name in ("down_blocks", "middle1", "middle2", "up_blocks"):
            if name in index:
                roles.setdefault("block", []).append(name if name.startswith("middle")
                                                     else f"{name}.*")
        return roles

    def axes(self, index: Any) -> dict[str, Axes]:
        return {"<root>": Axes.parse("b c h w")}

    def process(self) -> ProcessSpec:
        return ProcessSpec(step_from="occurrence", step_module="", sampler_callback=True)

    def notes(self) -> list[str]:
        return ["diffusion adapter: occurrence k of the denoiser = sampling step k"]
