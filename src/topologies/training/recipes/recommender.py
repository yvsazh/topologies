"""Recommender recipe (binary cross-entropy with logits)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from topologies.training.losses import LossOverrideConfig, compute_loss
from topologies.training.recipe_base import Recipe, RecipeConfig, register_recipe


class RecommenderBCEConfig(RecipeConfig):
    input_mode: str = "user_item"


@register_recipe("recommender-bce", help="Recommender click/BCE loss.")
class RecommenderBCERecipe(Recipe):
    level = "loss"
    ConfigSchema = RecommenderBCEConfig
    ComponentsSchema = None

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.0, "scheduler": None}

    def expected_batch_spec(self) -> dict[str, str]:
        return {
            "user": "LongTensor[B]",
            "item": "LongTensor[B]",
            "label": "FloatTensor[B]",
        }

    def validate_batch(self, batch: dict, *, strict: bool = True) -> None:
        if not strict:
            return
        has_ncf = all(k in batch for k in ("user", "item", "label"))
        has_fm = all(k in batch for k in ("x", "label"))
        if has_ncf or has_fm:
            return
        from topologies.training.errors import TrainingError as _TE

        raise _TE(
            "Training failed during batch validation.",
            recipe=self.name,
            stage="batch",
            expected=(
                "    user: LongTensor[B]  +  item: LongTensor[B]  +  label: FloatTensor[B]"
                "  (NCF-style)\n"
                "    OR  x: LongTensor[B, fields]  +  label: FloatTensor[B]  (DeepFM-style)"
            ),
            received=f"batch keys={sorted(batch.keys())}",
            hint="Provide user/item/label (NCF) or x/label (DeepFM field matrix).",
        )

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "user": torch.randint(0, 32, (16,), device=dev),
            "item": torch.randint(0, 64, (16,), device=dev),
            "label": torch.randint(0, 2, (16,), device=dev).float(),
            # Field matrix for DeepFM-style models: forward(x) with x[B, fields].
            "x": torch.randint(0, 8, (16, 4), device=dev),
        }

    def batch_to_loss(self, model, batch, cfg: RecommenderBCEConfig, components, adapter=None):
        self.validate_batch(batch, strict=True)
        if cfg.input_mode == "user_item":
            if not all(k in batch for k in ("user", "item")):
                raise ValueError("input_mode=user_item requires user and item IDs")
            out = adapter.call(model, batch) if adapter is not None else model(batch["user"], batch["item"])
        elif cfg.input_mode == "fields":
            if "x" not in batch:
                raise ValueError("input_mode=fields requires x categorical fields")
            out = adapter.call(model, batch) if adapter is not None else model(batch["x"])
        else:
            raise ValueError("input_mode must be 'user_item' or 'fields'")
        if isinstance(out, (tuple, list)):
            out = out[0]
        loss = F.binary_cross_entropy_with_logits(out.float().reshape(-1),
                                                  batch["label"].float().reshape(-1))
        return loss, {"loss": float(loss.detach())}


class RecommenderRankingConfig(LossOverrideConfig):
    pass


@register_recipe("recommender-bpr", help="Pairwise BPR ranking; negative sampling is supplied by the data pipeline.")
class RecommenderRankingRecipe(Recipe):
    level = "loss"
    ConfigSchema = RecommenderRankingConfig
    ComponentsSchema = None

    def expected_batch_spec(self) -> dict[str, str]:
        return {"users": "LongTensor[B]", "positive_items": "LongTensor[B]", "negative_items": "LongTensor[B]"}

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        required = ("users", "positive_items", "negative_items")
        if any(key not in batch for key in required):
            raise ValueError("BPR batch requires users, positive_items, and negative_items; sampler is external")
        pos = model(batch["users"], batch["positive_items"])
        neg = model(batch["users"], batch["negative_items"])
        loss = torch.nn.functional.softplus(neg - pos).mean() if cfg.loss is None else compute_loss(cfg, pos, neg)
        return loss, {"loss": float(loss.detach()), "pairwise_margin": float((pos - neg).mean().detach())}
