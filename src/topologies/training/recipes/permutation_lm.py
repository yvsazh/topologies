from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from pydantic import Field

from topologies.evaluation import EvaluationRecord
from topologies.training.recipe_base import EvaluationStepResult, Recipe, RecipeConfig, register_recipe
from topologies.training.recipes.lm import _call, _check_tokens, _token_val_metrics


class PermutationLMConfig(RecipeConfig):
    vocab_size: int | None = Field(default=None, gt=1)
    predict_fraction: float = Field(default=1 / 6, gt=0, le=1)
    ignore_index: int = -100
    evaluation_seed: int = 0


def permutation_batch(tokens, mask, predict_fraction=1 / 6, generator=None):
    if tokens.ndim != 2 or tokens.dtype != torch.long or mask.shape != tokens.shape:
        raise ValueError("permutation data requires LongTensor tokens and matching mask [B, T]")
    batch, length = tokens.shape
    ranks = torch.rand(tokens.shape, device=tokens.device, generator=generator).argsort(-1).argsort(-1)
    ranks = ranks.masked_fill(~mask.bool(), -1)
    labels = torch.full_like(tokens, -100)
    for row in range(batch):
        valid = mask[row].bool().nonzero().flatten()
        if valid.numel():
            count = max(1, math.ceil(valid.numel() * predict_fraction))
            selected = valid[ranks[row, valid].argsort()[-count:]]
            labels[row, selected] = tokens[row, selected]
    blocked = ranks[:, None, :] >= ranks[:, :, None]
    blocked = blocked | ~mask[:, None, :].bool()
    return {"tokens": tokens, "mask": mask.bool(), "perm_mask": blocked, "labels": labels}


@register_recipe("permutation-lm", help="XLNet two-stream permutation language modeling with uncorrupted tokens.")
class PermutationLMRecipe(Recipe):
    ConfigSchema = PermutationLMConfig
    ComponentsSchema = None

    def default_training_policy(self):
        return {"optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.01, "scheduler": "linear"}

    def expected_batch_spec(self):
        return {"tokens": "LongTensor[B, T] (uncorrupted)",
                "mask": "BoolTensor[B, T] (optional, true = real token)",
                "token_type_ids": "LongTensor[B, T] (optional segment IDs)",
                "perm_mask": "BoolTensor[B, T, T] (optional, true = blocked)",
                "target_mapping": "one-hot Tensor[B, N, T] (optional)",
                "labels": "LongTensor[B, T or N] (optional, -100 ignored)"}

    def optional_batch_keys(self):
        return set(self.expected_batch_spec()) - {"tokens"}

    def synthetic_batch(self, *, vocab_size, context_length, batch_size, seed, config):
        if vocab_size < 2 or context_length < 1:
            raise ValueError("permutation-lm synthetic data requires vocab_size >= 2 and context_length >= 1")
        generator = torch.Generator().manual_seed(seed)
        start = torch.randint(0, vocab_size, (batch_size, 1), generator=generator)
        tokens = (start + torch.arange(context_length)[None]) % vocab_size
        return {"tokens": tokens, "mask": torch.ones_like(tokens, dtype=torch.bool)}

    def dummy_batch(self, device="cpu"):
        batch = self.synthetic_batch(vocab_size=64, context_length=8, batch_size=4,
                                     seed=0, config=self.ConfigSchema())
        return {key: value.to(device) for key, value in batch.items()}

    def preflight(self, model, batch, cfg, components, adapter=None):
        _check_tokens(self.name, model, batch["tokens"], cfg.vocab_size)
        if not callable(getattr(model, "forward_pretraining", None)):
            raise ValueError("permutation-lm requires a two-stream forward_pretraining model")

    def _logits_and_targets(self, model, batch, cfg, adapter=None, evaluation=False):
        self.validate_batch(batch)
        tokens = batch["tokens"]
        mask = batch.get("mask", torch.ones_like(tokens, dtype=torch.bool)).bool()
        pad = getattr(model, "pad_token_id", None)
        if pad is not None:
            mask = mask & tokens.ne(pad)
        inputs = {"tokens": tokens, "mask": mask, "query_stream": True}
        if batch.get("perm_mask") is None:
            if "target_mapping" in batch or "labels" in batch:
                raise ValueError("explicit labels/target_mapping require perm_mask to define their factorization")
            generator = torch.Generator(device=tokens.device).manual_seed(cfg.evaluation_seed) if evaluation else None
            prepared = permutation_batch(tokens, mask, cfg.predict_fraction, generator)
            inputs["perm_mask"] = prepared["perm_mask"]
            labels = prepared["labels"]
            labels = labels.masked_fill(labels == -100, cfg.ignore_index)
        else:
            inputs["perm_mask"] = batch["perm_mask"]
            if "labels" not in batch:
                raise ValueError("explicit perm_mask requires labels aligned with target_mapping or tokens")
            labels = batch["labels"]
        for key in ("token_type_ids", "target_mapping"):
            if key in batch:
                inputs[key] = batch[key]
        logits = _call(model, inputs, list(inputs), adapter)
        if labels.dtype != torch.long or labels.shape != logits.shape[:-1]:
            raise ValueError("labels must be LongTensor aligned with the predicted positions")
        target_valid = mask
        if "target_mapping" in inputs:
            target_valid = mask.gather(1, inputs["target_mapping"].long().argmax(-1))
        labels = labels.masked_fill(~target_valid, cfg.ignore_index)
        if not (labels != cfg.ignore_index).any():
            raise ValueError("permutation-lm requires at least one nonpadding prediction target")
        return logits, labels

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        logits, labels = self._logits_and_targets(model, batch, cfg, adapter)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=cfg.ignore_index)
        return loss, {"loss": float(loss.detach())}

    def validation_step(self, model, batch, cfg, components, adapter=None):
        logits, labels = self._logits_and_targets(model, batch, cfg, adapter, evaluation=True)
        return _token_val_metrics(cfg, logits, labels)

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        logits, labels = self._logits_and_targets(model, batch, cfg, adapter, evaluation=True)
        return EvaluationStepResult(_token_val_metrics(cfg, logits, labels),
            EvaluationRecord(logits, labels, mask=labels != cfg.ignore_index,
                             sample_ids=batch.get("sample_ids"), metadata=batch))

    def predict_step(self, model, batch, cfg, components, adapter=None):
        inputs = {key: batch[key] for key in self.expected_batch_spec() if key in batch and key != "labels"}
        inputs["query_stream"] = True
        return _call(model, inputs, list(inputs), adapter)

    def summarize_metrics(self, metrics):
        if "loss" in metrics:
            metrics["ppl"] = math.exp(min(metrics["loss"], 50))
        return metrics
