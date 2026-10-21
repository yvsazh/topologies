from __future__ import annotations

import torch
import torch.nn.functional as F
from pydantic import Field

from topologies.evaluation import EvaluationRecord
from topologies.training.recipe_base import (
    EvaluationStepResult,
    Recipe,
    RecipeConfig,
    register_recipe,
)


class TransformerXLConfig(RecipeConfig):
    segment_length: int = Field(default=32, gt=0)
    ignore_index: int = -100
    vocab_size: int | None = Field(default=None, gt=1)


@register_recipe("transformer-xl-causal-lm", help="Segmented causal LM with explicit within-batch Transformer-XL memory.")
class TransformerXLCausalLMRecipe(Recipe):
    ConfigSchema = TransformerXLConfig
    ComponentsSchema = None

    def default_training_policy(self):
        return {"optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.01, "scheduler": "cosine"}

    def expected_batch_spec(self):
        return {"tokens": "LongTensor[B, T] contiguous stream", "mask": "BoolTensor[B, T] optional"}

    def optional_batch_keys(self):
        return {"mask"}

    def dummy_batch(self, device="cpu"):
        tokens = torch.arange(64, device=device).remainder(32).reshape(2, 32)
        return {"tokens": tokens, "mask": torch.ones_like(tokens, dtype=torch.bool)}

    def _run(self, model, batch, cfg):
        tokens = batch["tokens"]
        if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.shape[1] < 2:
            raise ValueError("transformer-xl-causal-lm needs LongTensor[B,T] with T >= 2")
        mask = batch.get("mask", torch.ones_like(tokens, dtype=torch.bool))
        if not isinstance(mask, torch.Tensor) or mask.shape != tokens.shape:
            raise ValueError("mask must have shape [B, T]")
        mask = mask.to(device=tokens.device, dtype=torch.bool)
        x, labels, valid = tokens[:, :-1], tokens[:, 1:], mask[:, :-1].bool() & mask[:, 1:].bool()
        memory = None
        memory_mask = None
        pieces = []
        for start in range(0, x.shape[1], cfg.segment_length):
            part = x[:, start:start + cfg.segment_length]
            part_valid = valid[:, start:start + cfg.segment_length]
            logits, memory, memory_mask = model.forward_with_mems(part, memory, part_valid, memory_mask)
            pieces.append(logits)
        logits = torch.cat(pieces, dim=1)
        labels = labels.masked_fill(~valid, cfg.ignore_index)
        if not (labels != cfg.ignore_index).any():
            raise ValueError("batch has no valid next-token targets")
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=cfg.ignore_index)
        return loss, logits, labels

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        loss, _, _ = self._run(model, batch, cfg)
        return loss, {"loss": float(loss.detach())}

    def validation_step(self, model, batch, cfg, components, adapter=None):
        loss, _, _ = self._run(model, batch, cfg)
        return {"loss": float(loss.detach())}

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        loss, logits, labels = self._run(model, batch, cfg)
        metrics = {"loss": float(loss.detach())}
        return EvaluationStepResult(metrics, EvaluationRecord(logits, labels, mask=labels != cfg.ignore_index,
            sample_ids=batch.get("sample_ids"), metadata=batch))
