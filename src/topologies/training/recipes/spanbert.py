from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from pydantic import Field

from topologies.evaluation import EvaluationRecord
from topologies.training.recipe_base import EvaluationStepResult, Recipe, RecipeConfig, register_recipe


class SpanBERTConfig(RecipeConfig):
    vocab_size: int | None = Field(default=None, gt=1)
    sbo_weight: float = Field(default=1.0, gt=0)
    ignore_index: int = -100
    pad_token_id: int = Field(default=0, ge=0)
    cls_token_id: int = Field(default=2, ge=0)
    sep_token_id: int = Field(default=3, ge=0)
    mask_token_id: int = Field(default=4, ge=0)
    special_token_ids: tuple[int, ...] = (0, 1, 2, 3, 4)
    masking_ratio: float = Field(default=0.15, gt=0, le=1)
    mean_span_length: float = Field(default=3.0, gt=0)
    max_span_length: int = Field(default=200, ge=1)
    geometric_p: float = Field(default=0.2, gt=0, le=1)
    span_lower: int = Field(default=1, ge=1)
    span_upper: int = Field(default=10, ge=1)


@register_recipe("spanbert-pretraining", help="Joint SpanBERT masked-language and span-boundary pretraining.")
class SpanBERTPretrainingRecipe(Recipe):
    ConfigSchema = SpanBERTConfig
    ComponentsSchema = None

    def default_training_policy(self):
        return {"optimizer": "adamw", "lr": 1e-4, "weight_decay": 0.01, "scheduler": "linear"}

    def expected_batch_spec(self):
        return {"tokens": "LongTensor[B,T] corrupted inputs", "mask": "BoolTensor[B,T] valid tokens",
                "labels": "LongTensor[B,T] MLM targets (-100 ignored)",
                "span_starts": "LongTensor[B,S] inclusive starts", "span_ends": "LongTensor[B,S] inclusive ends",
                "span_positions": "LongTensor[B,S,L] target positions", "span_labels": "LongTensor[B,S,L] SBO targets",
                "span_mask": "BoolTensor[B,S,L] valid SBO targets"}

    def optional_batch_keys(self):
        return set()

    def dummy_batch(self, device="cpu"):
        tokens = torch.tensor([[2, 5, 4, 6, 3], [2, 7, 4, 8, 3]], device=device)
        labels = torch.full_like(tokens, -100)
        labels[:, 2] = torch.tensor([6, 8], device=device)
        return {"tokens": tokens, "mask": torch.ones_like(tokens, dtype=torch.bool), "labels": labels,
                "span_starts": torch.full((2, 1), 2, dtype=torch.long, device=device),
                "span_ends": torch.full((2, 1), 2, dtype=torch.long, device=device),
                "span_positions": torch.full((2, 1, 1), 2, dtype=torch.long, device=device),
                "span_labels": torch.tensor([[[6]], [[8]]], device=device),
                "span_mask": torch.ones((2, 1, 1), dtype=torch.bool, device=device)}

    def synthetic_batch(self, *, vocab_size, context_length, batch_size, seed, config):
        from topologies.models.spanbert import SpanMaskingCollator

        if vocab_size < 8 or context_length < 3:
            raise ValueError("SpanBERT synthetic data needs vocabulary >= 8 and context >= 3")
        generator = torch.Generator().manual_seed(seed)
        special = set(config.special_token_ids) | {config.pad_token_id, config.mask_token_id}
        ordinary = [index for index in range(vocab_size) if index not in special]
        if not ordinary:
            raise ValueError("SpanBERT vocabulary has no ordinary token IDs")
        body_len = context_length - 2
        body = torch.tensor(ordinary, dtype=torch.long)
        choices = torch.randint(body.numel(), (batch_size, body_len), generator=generator)
        tokens = torch.cat((torch.full((batch_size, 1), config.cls_token_id, dtype=torch.long),
                            body[choices],
                            torch.full((batch_size, 1), config.sep_token_id, dtype=torch.long)), dim=1)
        word_ids = torch.full_like(tokens, -1)
        word_ids[:, 1:-1] = torch.arange(body_len).div(2, rounding_mode="floor").expand(batch_size, -1)
        collator = SpanMaskingCollator(vocab_size=vocab_size, mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id, special_token_ids=config.special_token_ids,
            masking_ratio=config.masking_ratio, geometric_p=config.geometric_p,
            span_lower=config.span_lower,
            span_upper=min(config.span_upper, max(1, context_length - 2)),
            max_span_length=config.max_span_length, seed=seed)
        return collator(tokens, word_ids)

    def _outputs(self, model, batch):
        if not callable(getattr(model, "forward_pretraining", None)):
            raise ValueError("spanbert-pretraining requires model.forward_pretraining")
        return model.forward_pretraining(batch["tokens"], batch.get("mask"), batch.get("token_type_ids"),
            batch["span_starts"], batch["span_ends"], batch["span_positions"])

    def _loss(self, model, batch, cfg):
        self.validate_batch(batch)
        output = self._outputs(model, batch)
        mlm = F.cross_entropy(output["mlm_logits"].flatten(0, 1), batch["labels"].flatten(),
                              ignore_index=cfg.ignore_index)
        valid = batch["span_mask"].bool() & batch["span_labels"].ne(cfg.ignore_index)
        if not bool(valid.any()):
            raise ValueError("spanbert-pretraining requires at least one valid SBO target")
        sbo = F.cross_entropy(output["sbo_logits"][valid], batch["span_labels"][valid])
        return mlm + cfg.sbo_weight * sbo, mlm, sbo, output

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        loss, mlm, sbo, _ = self._loss(model, batch, cfg)
        return loss, {"loss": float(loss.detach()), "mlm_loss": float(mlm.detach()), "sbo_loss": float(sbo.detach())}

    def validation_step(self, model, batch, cfg, components, adapter=None):
        loss, mlm, sbo, output = self._loss(model, batch, cfg)
        valid = batch["labels"].ne(cfg.ignore_index)
        correct = ((output["mlm_logits"].argmax(-1) == batch["labels"]) & valid).sum()
        return {"loss": float(loss.detach()), "mlm_loss": float(mlm.detach()),
                "sbo_loss": float(sbo.detach()), "accuracy": float(correct / valid.sum().clamp(min=1))}

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        loss, mlm, sbo, output = self._loss(model, batch, cfg)
        return EvaluationStepResult({"loss": float(loss.detach()), "mlm_loss": float(mlm.detach()),
            "sbo_loss": float(sbo.detach())}, EvaluationRecord(output["mlm_logits"], batch["labels"],
            mask=batch["labels"].ne(cfg.ignore_index), metadata=batch))

    def predict_step(self, model, batch, cfg, components, adapter=None):
        return self._outputs(model, batch)

    def summarize_metrics(self, metrics):
        if "loss" in metrics:
            metrics["perplexity"] = math.exp(min(metrics["loss"], 50))
        return metrics
