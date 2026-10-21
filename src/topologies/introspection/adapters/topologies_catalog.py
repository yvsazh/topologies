"""Adapters for topologies catalog models, matched by structure and ``component_id``.

They declare only what the code guarantees: the text encoder core
(``embedding`` + ``blocks`` + ``norm``) shared by the transformer LM, classifier and
tagger (directly or wrapped as ``core.encoder`` by ``topologies.nn``), blocks built
from ``layers.TransformerBlock``, the ViT layout, and CNN stage containers.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from topologies.introspection.adapters import AttentionSite
from topologies.introspection.adapters.torch_modules import SDPA_COMPONENTS, TorchModulesAdapter
from topologies.introspection.semantics import Axes

BTD = Axes.parse("b t d")
TD = Axes.parse("t d")
BCHW = Axes.parse("b c h w")


def _find(model: nn.Module, predicate: Callable[[nn.Module], bool]) -> str | None:
    for name, module in model.named_modules():
        if predicate(module):
            return name
    return None


def _is_text_encoder(module: nn.Module) -> bool:
    return isinstance(getattr(module, "embedding", None), nn.Embedding) and \
        isinstance(getattr(module, "blocks", None), nn.ModuleList) and \
        isinstance(getattr(module, "norm", None), nn.Module)


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class CatalogTransformerAdapter(TorchModulesAdapter):
    """Catalog text transformers (LM / classifier / tagger) and ``TransformerBlock`` stacks."""

    name = "catalog-transformer"
    priority = 10

    def matches(self, model: nn.Module) -> float:
        if _find(model, _is_text_encoder) is not None:
            return 1.0
        if any(getattr(type(m), "component_id", None) == "transformer-block" for m in model.modules()):
            return 0.6
        return 0.0

    def _layout(self, model: nn.Module) -> dict[str, Any] | None:
        encoder = _find(model, _is_text_encoder)
        if encoder is None:
            return None
        parent = encoder.rsplit(".", 1)[0] if "." in encoder else ""
        owner = model.get_submodule(parent) if parent else model
        enc = model.get_submodule(encoder)
        layout: dict[str, Any] = {"encoder": encoder, "parent": parent}
        lm_head = getattr(owner, "lm_head", None)
        head = getattr(owner, "head", None)
        if isinstance(lm_head, nn.Linear):
            layout["unembed"] = _join(parent, "lm_head")
        elif isinstance(head, nn.Linear):
            layout["head"] = _join(parent, "head")
        layout["position"] = _join(encoder, "pos") if isinstance(getattr(enc, "pos", None),
                                                                 nn.Module) else None
        first = enc.blocks[0] if len(enc.blocks) else None
        ff = getattr(first, "ff", None)
        layout["mlp_hidden"] = isinstance(ff, nn.Sequential) and len(ff) > 1
        return layout

    def roles(self, index: Any) -> dict[str, list[str]]:
        model = index.model
        layout = self._layout(model)
        if layout is None:
            blocks = index.resolve_modules("component:transformer-block", allow_empty=True)
            if not blocks:
                return {}
            return {"block": ["component:transformer-block"],
                    "residual": ["component:transformer-block"],
                    "attention": [f"{b}.attn" for b in blocks if f"{b}.attn" in index],
                    "mlp": [f"{b}.ff" for b in blocks if f"{b}.ff" in index]}
        enc = layout["encoder"]
        roles = {"embed": [_join(enc, "embedding")], "block": [_join(enc, "blocks.*")],
                 "residual": [_join(enc, "blocks.*")], "attention": [_join(enc, "blocks.*.attn")],
                 "mlp": [_join(enc, "blocks.*.ff")], "norm": [_join(enc, "blocks.*.n1"),
                                                             _join(enc, "blocks.*.n2")],
                 "final_norm": [_join(enc, "norm")], "encoder": [enc]}
        if layout["position"]:
            roles["position"] = [layout["position"]]
        if layout["mlp_hidden"]:
            roles["mlp_hidden"] = [_join(enc, "blocks.*.ff.1")]
        if "unembed" in layout:
            roles["unembed"] = [layout["unembed"]]
        if "head" in layout:
            roles["head"] = [layout["head"]]
        return roles

    def axes(self, index: Any) -> dict[str, Axes]:
        roles = self.roles(index)
        out = {}
        for role in ("embed", "position", "block", "attention", "mlp", "mlp_hidden", "norm"):
            for selector in roles.get(role, []):
                out[selector] = BTD
        # A learned position table is called with arange(T): its output has no batch axis.
        for selector in roles.get("position", []):
            try:
                module = index.model.get_submodule(selector)
            except AttributeError:
                continue
            if isinstance(module, nn.Embedding):
                out[selector] = TD
        return out

    def decoders(self, model: nn.Module) -> dict[str, Callable]:
        layout = self._layout(model)
        if layout is None or "unembed" not in layout:
            return {}
        norm = model.get_submodule(_join(layout["encoder"], "norm"))
        head = model.get_submodule(layout["unembed"])
        return {"unembed": lambda h: head(norm(h)), "unembed_raw": head}

    def attention_sites(self, index: Any) -> list[AttentionSite]:
        sites = []
        for path in index.resolve_modules("role:attention", allow_empty=True):
            module = index.module(path)
            sdpa = [p for p in index.paths() if (p == path or p.startswith(path + ".")) and
                    getattr(type(index.module(p)), "component_id", None) in SDPA_COMPONENTS]
            if sdpa:
                sites.append(AttentionSite(sdpa[0], "sdpa", getattr(index.module(sdpa[0]), "heads", None)))
            elif isinstance(module, nn.MultiheadAttention):
                sites.append(AttentionSite(path, "mha", module.num_heads))
            elif hasattr(module, "qkv") and hasattr(module, "heads"):
                sites.append(AttentionSite(path, "softmax", module.heads,
                                           "manual softmax inside forward, captured in op mode"))
            elif _uses_sdpa(module):
                sites.append(AttentionSite(path, "sdpa", getattr(module, "heads", None)))
        return sites or super().attention_sites(index)

    def notes(self) -> list[str]:
        return [("catalog transformer adapter: residual = block outputs (the stream after each "
                "block); padded positions are zeroed by the catalog encoder")]


def _uses_sdpa(module: nn.Module) -> bool:
    import inspect

    try:
        return "scaled_dot_product_attention" in inspect.getsource(type(module).forward)
    except (OSError, TypeError):
        return False


class CatalogVisionAdapter(TorchModulesAdapter):
    """ViT (patch/block/head roles, CLS at position 0) and CNN stage containers."""

    name = "catalog-vision"
    priority = 10

    @staticmethod
    def _vit(model: nn.Module) -> str | None:
        return _find(model, lambda m: isinstance(getattr(m, "patch_embed", None), nn.Module) and
                     isinstance(getattr(m, "blocks", None), nn.Sequential | nn.ModuleList) and
                     hasattr(m, "cls_token"))

    @staticmethod
    def _cnn(model: nn.Module) -> tuple[str, str] | None:
        for name, module in model.named_modules():
            for container in ("stages", "features"):
                child = getattr(module, container, None)
                if isinstance(child, nn.Sequential | nn.ModuleList) and any(
                        isinstance(m, nn.modules.conv._ConvNd) for m in child.modules()):
                    return name, container
        return None

    def matches(self, model: nn.Module) -> float:
        if self._vit(model) is not None:
            return 1.0
        return 0.8 if self._cnn(model) is not None else 0.0

    def roles(self, index: Any) -> dict[str, list[str]]:
        model = index.model
        vit = self._vit(model)
        if vit is not None:
            roles = {"patch": [_join(vit, "patch_embed")], "block": [_join(vit, "blocks.*")],
                     "residual": [_join(vit, "blocks.*")], "attention": [_join(vit, "blocks.*.attn")],
                     "mlp": [_join(vit, "blocks.*.mlp")], "final_norm": [_join(vit, "norm")]}
            if _join(vit, "head") in index:
                roles["head"] = [_join(vit, "head")]
            return roles
        cnn = self._cnn(model)
        if cnn is None:
            return {}
        owner, container = cnn
        roles = {"stage": [_join(owner, f"{container}.*")], "block": [_join(owner, f"{container}.*")]}
        for head in ("classifier", "head", "fc"):
            if _join(owner, head) in index:
                roles["head"] = [_join(owner, head)]
                break
        return roles

    def axes(self, index: Any) -> dict[str, Axes]:
        roles = self.roles(index)
        if "patch" in roles:
            return {s: BTD for role in ("block", "attention", "mlp") for s in roles.get(role, [])}
        return {s: BCHW for s in roles.get("stage", [])}

    def notes(self) -> list[str]:
        return ["vision adapter: ViT token 0 is the CLS token; CNN stage outputs are b c h w"]
