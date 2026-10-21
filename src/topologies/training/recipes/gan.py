"""GAN recipe (Level 2: alternating generator/discriminator updates)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from topologies._execution import primary_output
from topologies.training.losses import LossOverrideConfig, get_loss
from topologies.training.recipe_base import (
    Recipe,
    RecipeComponents,
    StepContext,
    StepResult,
    register_recipe,
)


class GANComponents(RecipeComponents):
    discriminator: nn.Module
    latent_dim: int = 8


class GANConfig(LossOverrideConfig):
    latent_dim: int = 8
    data_dim: int = 32
    adversarial_weight: float = 1.0
    image_channels: int | None = None
    image_size: int = 64


@register_recipe("gan", help="Alternating GAN updates (generator + discriminator).")
class GANRecipe(Recipe):
    level = "step"
    ConfigSchema = GANConfig
    ComponentsSchema = GANComponents

    def default_training_policy(self) -> dict:
        return {
            "optimizers": {
                "generator": {"name": "adam", "lr": 2e-4},
                "discriminator": {"name": "adam", "lr": 2e-4},
            }
        }

    def expected_batch_spec(self) -> dict[str, str]:
        return {"x": "FloatTensor[B, D] (real samples) or [B,C,H,W] image batches"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        return {"x": torch.randn(2, self.ConfigSchema().data_dim, device=dev)}

    def optimizer_params(self, optimizer_name: str, model, components):
        if optimizer_name == "discriminator" and components is not None:
            disc = getattr(components, "discriminator", None)
            if disc is not None:
                return disc.parameters()
        return model.parameters()

    def training_step(self, ctx: StepContext) -> StepResult:
        assert ctx.batch is not None, "GAN requires dict batches with 'x'"
        real = ctx.batch["x"].to(ctx.device)
        cfg: GANConfig = ctx.config
        comp: GANComponents = ctx.components
        disc = comp.discriminator.to(ctx.device)
        gen = ctx.model
        if real.ndim not in (2, 4):
            raise ValueError("GAN real samples must be vectors [B,D] or images [B,C,H,W]")
        if real.shape[0] < 1:
            raise ValueError("GAN batch must be non-empty")
        bsz = real.shape[0]
        z = torch.randn(bsz, cfg.latent_dim, device=ctx.device)
        # -- discriminator update on real + fake (no grad through generator).
        ctx.zero_grad("discriminator")
        gen_mode = gen.training
        gen.eval()
        try:
            with torch.no_grad():
                fake_detached = gen(z)
                fake_detached = _first_tensor(fake_detached).detach()
        finally:
            gen.train(gen_mode)
        d_real = _first_tensor(_call_disc(disc, real))
        d_fake = _first_tensor(_call_disc(disc, fake_detached))
        if cfg.loss is None:
            d_loss = F.binary_cross_entropy_with_logits(
            d_real.reshape(-1), torch.ones_like(d_real.reshape(-1))
            ) + F.binary_cross_entropy_with_logits(
            d_fake.reshape(-1), torch.zeros_like(d_fake.reshape(-1))
            )
        elif cfg.loss == "adversarial-hinge":
            d_loss = get_loss(cfg.loss)(d_real, d_fake, **dict(cfg.loss_kwargs or {}))
        else:
            raise ValueError("GAN loss override must be 'adversarial-hinge'")
        ctx.backward(d_loss, optimizer_name="discriminator")
        ctx.clip_gradients("discriminator")
        ctx.step_optimizer("discriminator")
        ctx.step_scheduler("discriminator")
        # -- generator update (fool the discriminator).
        ctx.zero_grad("generator")
        disc_mode = disc.training
        requires_grad = [p.requires_grad for p in disc.parameters()]
        disc.eval()
        for parameter in disc.parameters():
            parameter.requires_grad_(False)
        try:
            fake = _first_tensor(gen(z))
            d_fake_g = _first_tensor(disc(fake))
        finally:
            for parameter, enabled in zip(disc.parameters(), requires_grad):
                parameter.requires_grad_(enabled)
            disc.train(disc_mode)
        if cfg.loss is None:
            g_loss = F.binary_cross_entropy_with_logits(
            d_fake_g.reshape(-1), torch.ones_like(d_fake_g.reshape(-1))
            )
        else:
            from topologies.training.losses import adversarial_hinge_generator
            g_loss = adversarial_hinge_generator(d_fake_g, **dict(cfg.loss_kwargs or {}))
        g_loss = cfg.adversarial_weight * g_loss
        ctx.backward(g_loss, optimizer_name="generator")
        ctx.clip_gradients("generator")
        ctx.step_optimizer("generator")
        ctx.step_scheduler("generator")
        total = g_loss.detach() + d_loss.detach()
        return StepResult(
            loss=float(total),
            metrics={
                "loss": float(total),
                "g_loss": float(g_loss.detach()),
                "d_loss": float(d_loss.detach()),
            },
        )


def _first_tensor(out):
    return primary_output(out, prefer=None, strict=False)


def _call_disc(disc: nn.Module, x: torch.Tensor):
    return disc(x)
