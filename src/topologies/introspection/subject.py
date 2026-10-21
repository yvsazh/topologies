"""Subject: the model plus everything that tells us how to feed and interpret it."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from topologies.introspection.adapters import Adapter, resolve_adapter
from topologies.introspection.errors import UnsupportedModel
from topologies.introspection.semantics import Axes
from topologies.introspection.sites import SiteIndex


@dataclass
class Declaration:
    selector: Any
    axes: Axes | None = None
    role: str | None = None


@dataclass
class Subject:
    """Model + tokenizer + recipe + adapter + device, resolved once per inspector."""

    model: nn.Module
    source_model: nn.Module | None = None
    tokenizer: Any = None
    recipe: Any = None
    recipe_name: str | None = None
    recipe_source: str | None = None
    recipe_config: dict = field(default_factory=dict)
    adapter: Adapter | None = None
    adapter_provenance: dict = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    declarations: list[Declaration] = field(default_factory=list)
    batch_size: int | None = None
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    inspector: Any = None
    _index: SiteIndex | None = None
    _axes_cache: dict = field(default_factory=dict)

    # -- resolution -------------------------------------------------------------------

    @classmethod
    def resolve(cls, source: Any, *, tokenizer: Any = None, recipe: Any = None,
                adapter: Any = "auto", device: Any = None) -> Subject:
        from topologies.build.contract import unwrap

        meta: dict[str, Any] = {}
        recipe_name, recipe_source, recipe_config = None, None, {}
        if isinstance(source, (str, Path)):
            from topologies.pretrained import load_pretrained

            loaded = load_pretrained(source, device=device or "cpu")
            meta["path"] = str(source)
            source = loaded
        if type(source).__name__ == "Pretrained" and hasattr(source, "model"):
            model = source.model
            tokenizer = tokenizer if tokenizer is not None else source.tokenizer
            if source.recipe:
                recipe_name, recipe_source = source.recipe, "bundle"
                recipe_config = dict(source.recipe_config or {})
            meta.update({"metadata": dict(source.metadata or {}),
                         "path": str(source.path) if source.path else meta.get("path")})
        elif not isinstance(source, nn.Module) and hasattr(source, "model") and hasattr(source, "fit"):
            model = source.model
            tokenizer = tokenizer if tokenizer is not None else getattr(source, "tokenizer", None)
            trainer_recipe = getattr(source, "recipe", None)
            if trainer_recipe is not None and recipe is None:
                recipe = trainer_recipe
                recipe_source = "trainer"
            cfg = getattr(source, "recipe_config", None)
            if cfg is not None:
                try:
                    recipe_config = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)
                except Exception:  # noqa: BLE001
                    recipe_config = {}
            meta["trainer"] = type(source).__name__
        elif isinstance(source, nn.Module):
            model = source
        else:
            raise UnsupportedModel(
                f"cannot introspect {type(source).__name__}; pass an nn.Module, a Pretrained, "
                "a Trainer or a path to a saved bundle")
        if not isinstance(model, nn.Module):
            raise UnsupportedModel(f"{type(model).__name__} is not an nn.Module")
        inner = unwrap(model)
        warn: list[str] = []
        if inner is not model:
            warn.append(f"introspecting the unwrapped {type(inner).__name__} inside "
                        f"{type(model).__name__} (DDP/compile wrappers run eagerly here)")
        if isinstance(inner, torch.jit.ScriptModule):
            warn.append("TorchScript module: internals are opaque; the call is treated as a leaf")
        if _is_huggingface(inner):
            meta["huggingface"] = True
        if recipe is not None and not isinstance(recipe, str):
            recipe_obj = recipe
            recipe_name = getattr(recipe, "name", type(recipe).__name__)
            recipe_source = recipe_source or "explicit"
        else:
            if isinstance(recipe, str):
                recipe_name, recipe_source = recipe, "explicit"
            elif recipe_name is None:
                recipe_name = infer_recipe_name(inner)
                recipe_source = "inferred" if recipe_name else None
            recipe_obj = _load_recipe(recipe_name, warn) if recipe_name else None
        resolved_adapter, provenance = resolve_adapter(inner, adapter)
        dev = torch.device(device) if device is not None else _model_device(inner)
        subject = cls(model=inner, source_model=model, tokenizer=tokenizer, recipe=recipe_obj,
                      recipe_name=recipe_name, recipe_source=recipe_source,
                      recipe_config=recipe_config, adapter=resolved_adapter,
                      adapter_provenance=provenance, device=dev, metadata=meta, warnings=warn)
        for message in warn:
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        return subject

    # -- index and semantics ------------------------------------------------------------

    @property
    def index(self) -> SiteIndex:
        if self._index is None:
            self._index = SiteIndex.build(self.model)
            self._install_roles()
        return self._index

    def _install_roles(self) -> None:
        index = self._index
        assert index is not None
        roles: dict[str, list[str]] = {}
        if self.adapter is not None:
            for role, selectors in self.adapter.roles(index).items():
                roles.setdefault(role, []).extend(selectors)
        for decl in self.declarations:
            if decl.role:
                roles.setdefault(decl.role, []).append(decl.selector)
        index.set_roles(roles)
        self._axes_cache.clear()

    def declare(self, selector: Any, *, axes: Any = None, role: str | None = None) -> None:
        parsed = Axes.parse(axes) if axes is not None else None
        if parsed is None and role is None:
            raise ValueError("declare(...) needs axes=... and/or role=...")
        self.index.resolve_modules(selector)  # fail early on empty selections
        self.declarations.append(Declaration(selector, parsed, role))
        self._install_roles()

    def declared_axes(self, path: str, port: str = "output") -> Axes | None:
        """User declarations win over adapter axes; ``None`` means fall back to inference."""
        key = (path, port)
        if key in self._axes_cache:
            return self._axes_cache[key]
        found = None
        if port in ("output", "output.0", "grad_output"):
            for decl in reversed(self.declarations):
                if decl.axes is not None and path in self.index.resolve_modules(
                        decl.selector, allow_empty=True):
                    found = decl.axes
                    break
            if found is None and self.adapter is not None:
                for selector, axes in self.adapter.axes(self.index).items():
                    if path in self.index.resolve_modules(selector, allow_empty=True):
                        found = axes
        self._axes_cache[key] = found
        return found

    def describe(self) -> dict[str, Any]:
        return {"model": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "adapter": self.adapter_provenance, "device": str(self.device),
                "recipe": self.recipe_name, "recipe_source": self.recipe_source,
                "tokenizer": type(self.tokenizer).__name__ if self.tokenizer is not None else None,
                **{k: v for k, v in self.metadata.items() if k in ("path", "huggingface", "trainer")}}

    def recipe_cfg(self) -> Any:
        if self.recipe is None:
            return None
        schema = getattr(type(self.recipe), "ConfigSchema", None)
        if schema is None:
            return None
        try:
            return schema(**self.recipe_config)
        except Exception:  # noqa: BLE001 - best effort; base predict_step ignores cfg
            try:
                return schema()
            except Exception:  # noqa: BLE001
                return None


def _model_device(model: nn.Module) -> torch.device:
    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return torch.device("cpu")


def _is_huggingface(model: nn.Module) -> bool:
    return hasattr(model, "config") and callable(getattr(model, "get_output_embeddings", None))


_RECIPE_TABLE: dict[str, str] | None = None


def infer_recipe_name(model: nn.Module) -> str | None:
    """Recipe from ``__topologies_spec__`` or the catalog recipe table (recorded as inferred)."""
    spec = getattr(model, "__topologies_spec__", None)
    if isinstance(spec, dict) and isinstance(spec.get("recipe"), str):
        return spec["recipe"]
    global _RECIPE_TABLE
    if _RECIPE_TABLE is None:
        try:
            raw = json.loads(files("topologies.training").joinpath("model_recipes.json").read_text())
            _RECIPE_TABLE = {f"{v['module']}.{v['class']}": v["recipe"]
                             for v in raw.get("models", {}).values()}
        except Exception:  # noqa: BLE001
            _RECIPE_TABLE = {}
    cls = type(model)
    return _RECIPE_TABLE.get(f"{cls.__module__}.{cls.__name__}")


def _load_recipe(name: str, warn: list[str]) -> Any:
    try:
        import topologies.training  # noqa: F401 - registers built-in recipes
        from topologies.training.recipe_base import get_recipe

        return get_recipe(name)
    except Exception as exc:  # noqa: BLE001
        warn.append(f"recipe {name!r} could not be loaded ({exc}); recipe features are off")
        return None
