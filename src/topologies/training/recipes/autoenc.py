"""Autoencoder / VAE / VQ-VAE recipes."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from topologies.training.adapters import ModelAdapter, call_with_adapter
from topologies.training.losses import LossOverrideConfig, compute_loss
from topologies.training.recipe_base import Recipe, RecipeConfig, register_recipe


def _forward_single(model, x: torch.Tensor, batch: dict | None = None, adapter=None):
    if adapter is not None and (
        getattr(adapter, "forward_map", None) is not None
        or getattr(adapter, "output_map", None) is not None
    ):
        return adapter.call(model, batch or {"x": x})
    try:
        out = call_with_adapter(adapter, model, {"x": x}, ["x"])
    except Exception:  # noqa: BLE001
        try:
            out = model(x)
        except Exception:  # noqa: BLE001
            out = ModelAdapter().call(model, {"x": x})
    return out


class AutoencoderConfig(RecipeConfig):
    pass


@register_recipe("autoencoder", help="Dense autoencoder reconstruction (MSE).")
class AutoencoderRecipe(Recipe):
    level = "loss"
    ConfigSchema = AutoencoderConfig
    ComponentsSchema = None

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.0, "scheduler": None}

    def expected_batch_spec(self) -> dict[str, str]:
        return {"x": "FloatTensor[B, D]"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {"x": torch.rand(8, 32, device=dev)}

    def batch_to_loss(self, model, batch, cfg: AutoencoderConfig, components, adapter=None):
        self.validate_batch(batch, strict=True)
        out = _forward_single(model, batch["x"], batch, adapter)
        recon = out[0] if isinstance(out, (tuple, list)) else out
        loss = F.mse_loss(recon, batch["x"])
        return loss, {"loss": float(loss.detach())}


class VAEConfig(LossOverrideConfig):
    kl_weight: float = 1.0


@register_recipe("vae", help="Variational autoencoder ELBO (MSE + KL).")
class VAERecipe(Recipe):
    level = "loss"
    ConfigSchema = VAEConfig
    ComponentsSchema = None

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.0, "scheduler": None}

    def expected_batch_spec(self) -> dict[str, str]:
        return {"x": "FloatTensor[B, D]"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {"x": torch.rand(8, 32, device=dev)}

    def batch_to_loss(self, model, batch, cfg: VAEConfig, components, adapter=None):
        self.validate_batch(batch, strict=True)
        out = _forward_single(model, batch["x"], batch, adapter)
        recon, mu, logvar = out[0], out[1], out[2]
        mse = F.mse_loss(recon, batch["x"])
        kl = ((-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)).mean()
              if cfg.loss is None else compute_loss(cfg, mu, logvar))
        loss = mse + cfg.kl_weight * kl
        return loss, {
            "loss": float(loss.detach()),
            "mse": float(mse.detach()),
            "kl": float(kl.detach()),
        }


class VQVAEConfig(RecipeConfig):
    commitment_cost: float = 1.0


@register_recipe("vqvae", help="VQ-VAE reconstruction + commitment loss.")
class VQVAERecipe(Recipe):
    level = "loss"
    ConfigSchema = VQVAEConfig
    ComponentsSchema = None

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.0, "scheduler": None}

    def expected_batch_spec(self) -> dict[str, str]:
        return {"x": "FloatTensor[B, D]"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {"x": torch.rand(8, 32, device=dev)}

    def batch_to_loss(self, model, batch, cfg: VQVAEConfig, components, adapter=None):
        self.validate_batch(batch, strict=True)
        out = _forward_single(model, batch["x"], batch, adapter)
        recon, _codes, commit = out[0], out[1], out[2]
        mse = F.mse_loss(recon, batch["x"])
        commit_loss = commit.mean() if isinstance(commit, torch.Tensor) else torch.as_tensor(
            float(commit), device=mse.device
        )
        loss = mse + cfg.commitment_cost * commit_loss
        return loss, {"loss": float(loss.detach()), "mse": float(mse.detach())}
