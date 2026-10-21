"""Distillation recipe (Level 2: student update against frozen teacher)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from topologies.training.recipe_base import (
    Recipe,
    RecipeComponents,
    RecipeConfig,
    StepContext,
    StepResult,
    register_recipe,
)


class DistillationComponents(RecipeComponents):
    teacher_model: nn.Module


class DistillationConfig(RecipeConfig):
    temperature: float = 2.0
    alpha: float = 0.5


@register_recipe("distillation", help="Teacher/student distillation (KL + CE) for any classifier.")
class DistillationRecipe(Recipe):
    level = "step"
    ConfigSchema = DistillationConfig
    ComponentsSchema = DistillationComponents

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.0, "scheduler": None}

    def expected_batch_spec(self) -> dict[str, str]:
        return {
            "tokens": "LongTensor[B, T] (token classifiers/LMs) — or —",
            "pixels": "FloatTensor[B, C, H, W] (image classifiers)",
            "labels": "LongTensor[B] or LongTensor[B, T]",
        }

    def optional_batch_keys(self) -> set[str]:
        return {"tokens", "pixels"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "tokens": torch.randint(0, 64, (4, 8), device=dev),
            "labels": torch.randint(0, 64, (4, 8), device=dev),
        }

    def _losses(self, student_logits, teacher_logits, labels, cfg: DistillationConfig):
        T = cfg.temperature
        s = student_logits.reshape(-1, student_logits.size(-1))
        t = teacher_logits.reshape(-1, teacher_logits.size(-1))
        distill = F.kl_div(
            F.log_softmax(s / T, dim=-1), F.softmax(t / T, dim=-1), reduction="batchmean"
        ) * (T * T)
        hard = F.cross_entropy(s, labels.reshape(-1).long(), ignore_index=-100)
        return cfg.alpha * distill + (1.0 - cfg.alpha) * hard, distill, hard

    def training_step(self, ctx: StepContext) -> StepResult:
        assert ctx.batch is not None
        cfg: DistillationConfig = ctx.config
        comp: DistillationComponents = ctx.components
        teacher = _frozen_teacher(comp.teacher_model, ctx.device)
        batch = {k: (v.to(ctx.device) if isinstance(v, torch.Tensor) else v)
                 for k, v in ctx.batch.items()}
        ctx.zero_grad("default")
        student_logits = _forward(ctx.model, batch, ctx.adapter)
        with torch.no_grad():
            teacher_logits = _forward(teacher, batch, None)
        loss, distill, hard = self._losses(student_logits, teacher_logits, batch["labels"], cfg)
        ctx.backward(loss, optimizer_name="default")
        ctx.clip_gradients("default")
        ctx.step_optimizer("default")
        ctx.step_scheduler("default")
        return StepResult(
            loss=float(loss.detach()),
            metrics={
                "loss": float(loss.detach()),
                "distill": float(distill.detach()),
                "hard": float(hard.detach()),
            },
        )

    def validation_step(self, model, batch, cfg: DistillationConfig, components, adapter=None):
        """Student quality on held-out data: hard-label loss + accuracy."""
        logits = _forward(model, batch, adapter)
        labels = batch["labels"].long()
        flat = logits.reshape(-1, logits.size(-1))
        loss = F.cross_entropy(flat, labels.reshape(-1), ignore_index=-100)
        valid = labels.reshape(-1) != -100
        acc = ((flat.argmax(-1) == labels.reshape(-1)) & valid).float().sum() / valid.sum().clamp(
            min=1)
        return {"loss": float(loss), "acc": float(acc)}

    def predict_step(self, model, batch, cfg, components, adapter=None):
        return _forward(model, batch, adapter)


def _frozen_teacher(teacher: nn.Module, device: torch.device) -> nn.Module:
    # The teacher runs under no_grad; its requires_grad flags are left untouched
    # so the same module stays trainable elsewhere (e.g. a just-trained model).
    teacher = teacher.to(device)
    teacher.eval()
    return teacher


def _forward(model: nn.Module, batch: dict, adapter=None) -> torch.Tensor:
    from topologies.training.adapters import call_with_adapter
    from topologies.training.errors import TrainingError

    key = next((k for k in ("tokens", "pixels", "images", "x") if k in batch), None)
    if key is None:
        raise TrainingError(
            "Distillation batch has no model input.",
            recipe="distillation", stage="batch",
            received=f"batch keys={sorted(batch)}",
            hint="Provide 'tokens' (token models) or 'pixels' (image models) plus 'labels'.",
        )
    out = call_with_adapter(adapter, model, {key: batch[key]}, [key], primary=key)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out
