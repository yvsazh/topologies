"""Diffusion recipe (noise-prediction MSE on synthetic noise)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from topologies.inference_tasks.diffusion import DiffusionSchedule
from topologies.training.recipe_base import (
    Recipe,
    RecipeComponents,
    RecipeConfig,
    register_recipe,
)


class DiffusionComponents(RecipeComponents):
    noise_sampler: object | None = None
    generator: torch.Generator | None = None


class DiffusionConfig(RecipeConfig):
    noise_schedule: str = "linear"
    prediction_type: str = "epsilon"
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02


@register_recipe("diffusion", help="Simple diffusion noise-prediction (MSE).")
class DiffusionRecipe(Recipe):
    level = "loss"
    ConfigSchema = DiffusionConfig
    ComponentsSchema = DiffusionComponents

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 1e-4, "weight_decay": 0.0, "scheduler": "cosine"}

    def expected_batch_spec(self) -> dict[str, str]:
        return {"x": "FloatTensor[B, C, H, W]"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {"x": torch.randn(2, 16, 8, 8, device=dev)}

    def batch_to_loss(self, model, batch, cfg: DiffusionConfig, components, adapter=None):
        self.validate_batch(batch, strict=True)
        x = batch["x"]
        if cfg.num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        if cfg.prediction_type not in {"epsilon", "x0", "v"}:
            raise ValueError("prediction_type must be epsilon, x0, or v")
        schedule = DiffusionSchedule.create(cfg.num_timesteps, cfg.noise_schedule,
                                            cfg.beta_start, cfg.beta_end, x.device)
        generator = getattr(components, "generator", None) if components is not None else None
        t = torch.randint(0, cfg.num_timesteps, (x.shape[0],), device=x.device, generator=generator)
        sampler = getattr(components, "noise_sampler", None) if components is not None else None
        noise = sampler(x.shape, device=x.device, generator=generator) if sampler is not None else torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        if not isinstance(noise, torch.Tensor) or noise.shape != x.shape:
            raise ValueError("noise_sampler must return a tensor with exactly the input shape")
        noise = noise.to(device=x.device, dtype=x.dtype)
        x_t = schedule.add_noise(x, noise, t)
        if adapter is not None and (
            getattr(adapter, "forward_map", None) is not None
            or getattr(adapter, "output_map", None) is not None
        ):
            pred = adapter.call(model, {**batch, "x_t": x_t, "x": x_t, "t": t})
        else:
            try:
                import inspect as _insp

                if "t" not in _insp.signature(model.forward).parameters:
                    raise TypeError("diffusion model forward must accept timestep t")
                pred = model(x_t, t)
            except (ValueError, TypeError) as exc:
                raise TypeError("unable to inspect/call diffusion model with (x_t, t)") from exc
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if cfg.prediction_type == "epsilon":
            target = noise
        elif cfg.prediction_type == "x0":
            target = x
        else:
            a = schedule.extract(schedule.alpha_cumprod, t, x.ndim)
            target = a.sqrt() * noise - (1 - a).sqrt() * x
        loss = F.mse_loss(pred, target)
        return loss, {"loss": float(loss.detach())}
