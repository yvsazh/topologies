from __future__ import annotations

import torch
from pydantic import Field

from topologies.evaluation import EvaluationRecord
from topologies.training.recipe_base import (
    EvaluationStepResult,
    Recipe,
    RecipeConfig,
    register_recipe,
)


class RouterProbeConfig(RecipeConfig):
    aux_loss_coef: float = Field(default=0.02, ge=0.0)


@register_recipe("router-probe", help="Probe weighted top-k routing against expert-output targets.")
class RouterProbeRecipe(Recipe):
    level = "loss"
    ConfigSchema = RouterProbeConfig
    ComponentsSchema = None

    def expected_batch_spec(self):
        return {"features": "FloatTensor[B,T,D]", "expert_outputs": "FloatTensor[B,T,E,C]",
                "targets": "FloatTensor[B,T,C]", "mask": "BoolTensor[B,T] (optional)"}

    def optional_batch_keys(self):
        return {"mask"}

    def _forward(self, model, batch):
        indices, weights, aux = model(batch["features"], batch.get("mask"))
        experts = batch["expert_outputs"]
        if experts.ndim != 4 or experts.shape[:3] != (*batch["features"].shape[:2], model.num_experts):
            raise ValueError("expert_outputs must have shape [B,T,num_experts,C]")
        chosen = experts.gather(2, indices.unsqueeze(-1).expand(*indices.shape, experts.shape[-1]))
        predictions = (chosen * weights.unsqueeze(-1)).sum(2)
        mask = batch.get("mask")
        valid = torch.ones(predictions.shape[:2], dtype=torch.bool, device=predictions.device) if mask is None else mask.bool()
        denom = valid.sum().clamp_min(1)
        mse = ((predictions - batch["targets"]).square().mean(-1) * valid).sum() / denom
        return predictions, mse, aux

    @staticmethod
    def _metrics(mse, aux, total):
        return {"loss": float(total.detach()), "mse": float(mse.detach()),
                "router_aux_loss": float(aux.detach()), "total_loss": float(total.detach())}

    def batch_to_loss(self, model, batch, cfg: RouterProbeConfig, components, adapter=None):
        _, mse, aux = self._forward(model, batch)
        total = mse + cfg.aux_loss_coef * aux
        return total, self._metrics(mse, aux, total)

    def validation_step(self, model, batch, cfg, components, adapter=None):
        _, mse, aux = self._forward(model, batch)
        return self._metrics(mse, aux, mse + cfg.aux_loss_coef * aux)

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        predictions, mse, aux = self._forward(model, batch)
        total = mse + cfg.aux_loss_coef * aux
        metrics = self._metrics(mse, aux, total)
        return EvaluationStepResult(metrics, EvaluationRecord(predictions, batch["targets"], mask=batch.get("mask"), metadata=batch))
