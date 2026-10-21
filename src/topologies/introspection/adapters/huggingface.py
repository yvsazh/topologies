"""Optional Hugging Face adapter (duck-typed; ``transformers`` is never imported here).

It declares only what it finds: the layer list, input/output embeddings, the final
norm under its known attribute names, and per-layer attention modules. A role it
cannot find stays missing and is never invented. Attention weights come from the
model itself (``output_attentions=True``) when the model supports that.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from topologies.introspection.adapters import AttentionSite
from topologies.introspection.adapters.torch_modules import TorchModulesAdapter
from topologies.introspection.semantics import Axes

LAYER_LISTS = ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers",
               "encoder.layer", "bert.encoder.layer", "roberta.encoder.layer", "h", "layers")
FINAL_NORMS = ("model.norm", "transformer.ln_f", "model.final_layernorm", "gpt_neox.final_layer_norm",
               "model.decoder.final_layer_norm", "ln_f", "norm")
ATTENTION_NAMES = ("self_attn", "attn", "attention", "self_attention")


def is_huggingface(model: nn.Module) -> bool:
    return hasattr(model, "config") and callable(getattr(model, "get_output_embeddings", None)) \
        and callable(getattr(model, "get_input_embeddings", None))


class HuggingFaceAdapter(TorchModulesAdapter):
    name = "huggingface"
    priority = 20

    def matches(self, model: nn.Module) -> float:
        return 1.0 if is_huggingface(model) else 0.0

    def _layers(self, index: Any) -> str | None:
        for name in LAYER_LISTS:
            if name in index and isinstance(index.module(name), nn.ModuleList):
                return name
        return None

    def roles(self, index: Any) -> dict[str, list[str]]:
        model = index.model
        roles: dict[str, list[str]] = {}
        layers = self._layers(index)
        if layers:
            roles["block"] = [f"{layers}.*"]
            roles["residual"] = [f"{layers}.*"]
            first = f"{layers}.0"
            for name in ATTENTION_NAMES:
                if f"{first}.{name}" in index:
                    roles["attention"] = [f"{layers}.*.{name}"]
                    break
            for name in ("mlp", "feed_forward", "intermediate"):
                if f"{first}.{name}" in index:
                    roles["mlp"] = [f"{layers}.*.{name}"]
                    break
        embed = index.path_of(model.get_input_embeddings()) if model.get_input_embeddings() else None
        if embed is not None:
            roles["embed"] = [embed]
        out = model.get_output_embeddings()
        if out is not None and index.path_of(out) is not None:
            roles["unembed"] = [index.path_of(out)]
        for name in FINAL_NORMS:
            if name in index:
                roles["final_norm"] = [name]
                break
        return roles

    def axes(self, index: Any) -> dict[str, Axes]:
        roles = self.roles(index)
        btd = Axes.parse("b t d")
        return {s: btd for role in ("block", "embed", "final_norm") for s in roles.get(role, [])}

    def decoders(self, model: nn.Module) -> dict[str, Callable]:
        from topologies.introspection.sites import SiteIndex

        index = SiteIndex.build(model)
        roles = self.roles(index)
        if "unembed" not in roles:
            return {}
        head = index.module(roles["unembed"][0])
        if "final_norm" in roles:
            norm = index.module(roles["final_norm"][0])
            return {"unembed": lambda h: head(norm(h)), "unembed_raw": head}
        return {"unembed_raw": head}

    def attention_sites(self, index: Any) -> list[AttentionSite]:
        return [AttentionSite(p, "native", None, "model-native output_attentions")
                for p in index.resolve_modules("role:attention", allow_empty=True)]

    def notes(self) -> list[str]:
        return [("huggingface adapter: layer outputs are tuples; the first tensor (hidden state) "
                "is the residual stream")]
