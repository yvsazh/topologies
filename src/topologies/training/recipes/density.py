"""Maximum-likelihood recipe for continuous density models such as RealNVP."""

from __future__ import annotations

import torch
from pydantic import Field

from topologies.training.losses import LossOverrideConfig, get_loss
from topologies.training.recipe_base import Recipe, register_recipe


class DensityEstimationConfig(LossOverrideConfig):
    features: int = Field(default=2, gt=0)


@register_recipe("density-estimation", help="Maximum likelihood from model log probabilities.")
class DensityEstimationRecipe(Recipe):
    """Train models whose forward returns one log probability per example."""

    level = "loss"
    ConfigSchema = DensityEstimationConfig

    def expected_batch_spec(self) -> dict[str, str]:
        return {"x": "FloatTensor[B, features]"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        return {"x": torch.zeros(8, self.ConfigSchema().features, device=device)}

    def preflight(self, model, batch, cfg, components, adapter=None) -> None:
        self.validate_batch(batch)
        if batch["x"].ndim != 2 or batch["x"].shape[-1] != cfg.features:
            raise ValueError(f"density-estimation expects x [B, {cfg.features}]")
        log_prob = model.log_prob(batch["x"]) if hasattr(model, "log_prob") else model(batch["x"])
        if not isinstance(log_prob, torch.Tensor) or log_prob.shape not in (batch["x"].shape[:1], torch.Size([])):
            raise ValueError("density model must return one log probability per example")

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        self.validate_batch(batch)
        x = batch["x"]
        if x.ndim != 2 or x.shape[-1] != cfg.features:
            raise ValueError(f"density-estimation expects x [B, {cfg.features}]")
        log_prob = model.log_prob(x) if hasattr(model, "log_prob") else model(x)
        if not isinstance(log_prob, torch.Tensor) or log_prob.shape not in (x.shape[:1], torch.Size([])):
            raise ValueError("density model must return one log probability per example")
        loss = -log_prob.mean() if cfg.loss is None else get_loss(cfg.loss)(log_prob)
        return loss, {"loss": float(loss.detach()), "mean_log_prob": float(log_prob.detach().mean())}

    def predict_step(self, model, batch, cfg, components, adapter=None):
        x = batch["x"]
        return model.log_prob(x) if hasattr(model, "log_prob") else model(x)
