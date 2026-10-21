from __future__ import annotations

from typing import Any

import torch
from pydantic import Field

from topologies.synth.torch_views import Converter, register_converter, to_float_tensor
from topologies.training.losses import get_loss
from topologies.training.recipe_base import Recipe, RecipeConfig, register_recipe


class StructuredObjectiveConfig(RecipeConfig):
    loss: str
    loss_kwargs: dict[str, Any] = Field(default_factory=dict)
    bindings: dict[str, dict[str, str]]


@register_recipe("structured-objective", help="Bind structured model and batch fields to a named objective.")
class StructuredObjectiveRecipe(Recipe):
    level = "loss"
    ConfigSchema = StructuredObjectiveConfig

    def expected_batch_spec(self) -> dict[str, str]:
        return {"inputs": "FloatTensor[B,...]"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        return {"inputs": torch.zeros(2, 1, device=device), "targets": torch.zeros(2, 1, device=device)}

    def target_keys(self) -> set[str]:
        return {"targets"}

    def forward(self, model, batch):
        return model(batch["inputs"])

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        outputs = self.forward(model, batch)
        if not isinstance(outputs, dict):
            raise TypeError("structured-objective models must return a dict of named tensors")
        bound = {}
        for name, source in cfg.bindings.items():
            if set(source) == {"output"}:
                bound[name] = outputs[source["output"]]
            elif set(source) == {"batch"}:
                bound[name] = batch[source["batch"]]
            else:
                raise ValueError("each binding must specify exactly one of output or batch")
        loss = get_loss(cfg.loss)(**bound, **cfg.loss_kwargs)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError("configured objective must return a scalar tensor")
        return loss, {"loss": float(loss.detach())}


@register_converter("structured-objective")
class StructuredObjectiveConverter(Converter):
    def convert(self, record):
        from topologies.synth.valuetypes import Type

        row = {key: Type.unwrap(value) for key, value in record.items() if not str(key).startswith("_")}
        if "inputs" not in row:
            raise ValueError("structured-objective records need an inputs field")
        converted = {"inputs": to_float_tensor(row.pop("inputs")).float()}
        converted.update({key: to_float_tensor(value).float() for key, value in row.items()})
        return converted
