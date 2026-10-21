"""Language-modeling recipes: causal / masked / seq2seq / ELECTRA.

Conventions shared by the token recipes:

- ``tokens`` is the model input; an optional ``mask`` ([B, T], 1 = real
  token) is forwarded to models whose ``forward`` accepts ``mask`` (or
  ``attention_mask``) and dropped otherwise.
- causal-lm derives ``labels`` from ``tokens`` when omitted and shifts them
  internally; padded positions (``mask == 0``) are ignored in the loss.
- ``vocab_size`` in the recipe config is optional; when set (or when a
  tokenizer is given to ``Trainer``) it is checked against the model.
- causal-lm / masked-lm accept ``loss`` / ``loss_kwargs`` (see
  ``topologies.training.losses``); validation keeps cross-entropy as ``loss``.
- A model may return ``{"logits": ..., "aux_loss": tensor | {name: tensor}}``
  or set ``aux_loss`` on submodules during the forward. causal-lm adds those
  terms to the training loss and logs ``aux/<name>``. Validation ``loss``
  stays pure cross-entropy.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import Field, field_validator

from topologies.build.contract import embedding_sizes, model_context_length, model_vocab_size
from topologies.evaluation import EvaluationRecord
from topologies.training.adapters import call_with_adapter
from topologies.training.errors import TrainingError
from topologies.training.losses import LossOverrideConfig, compute_loss
from topologies.training.recipe_base import (
    EvaluationStepResult,
    Recipe,
    RecipeComponents,
    RecipeConfig,
    register_recipe,
)


def _forward(model: nn.Module, batch: dict, keys: list[str], adapter=None,
             primary: str | None = None) -> Any:
    try:
        return call_with_adapter(adapter, model, batch, keys, primary=primary or keys[0])
    except IndexError as e:
        raise _index_error(model, batch, e) from e


def _as_logits(out: Any) -> tuple[torch.Tensor, Any]:
    """Split a model output into logits and an optional ``aux_loss`` payload."""
    aux = None
    if isinstance(out, dict):
        aux = out.get("aux_loss")
        if "logits" in out:
            out = out["logits"]
        else:
            rest = [v for k, v in out.items() if k != "aux_loss"]
            out = rest[0] if rest else None
    if isinstance(out, (tuple, list)):
        for item in out:
            if isinstance(item, torch.Tensor) and item.is_floating_point():
                return item, aux
        return out[0], aux
    return out, aux


def _call(model: nn.Module, batch: dict, keys: list[str], adapter=None,
          primary: str | None = None) -> torch.Tensor:
    logits, _aux = _as_logits(_forward(model, batch, keys, adapter, primary=primary))
    return logits


def _index_error(model: nn.Module, batch: dict, err: Exception) -> TrainingError:
    tokens = batch.get("tokens", batch.get("src"))
    facts = []
    if isinstance(tokens, torch.Tensor):
        facts.append(f"max token id={int(tokens.max())}, sequence length T={tokens.shape[-1]}")
    sizes = embedding_sizes(model)
    if sizes:
        facts.append("model embeddings: " + ", ".join(f"{n}={k}" for n, k in sizes))
    return TrainingError(
        f"Model forward failed with an index error ({err}).",
        stage="loss",
        received="; ".join(facts) or None,
        hint="Either a token id is >= the model's vocabulary size (build the model with "
             "vocab_size=tokenizer.vocab_size) or T exceeds the model's position table "
             "(use a shorter block_size or a model built with a larger max_seq).",
    )


def _token_inputs(batch: dict) -> tuple[dict, list[str]]:
    inputs = {"tokens": batch["tokens"]}
    if batch.get("mask") is not None:
        inputs["mask"] = batch["mask"].bool()
    return inputs, list(inputs)


def _check_tokens(recipe: str, model: nn.Module, tokens: torch.Tensor,
                  cfg_vocab: int | None) -> None:
    model_vocab = model_vocab_size(model)
    if cfg_vocab is not None and model_vocab is not None and cfg_vocab != model_vocab:
        raise TrainingError(
            f"recipe_config vocab_size={cfg_vocab} but the model declares "
            f"vocab_size={model_vocab}.",
            recipe=recipe, stage="config", config_key="vocab_size",
            hint="Use one source of truth: vocab_size=tokenizer.vocab_size for both.",
        )
    limit = cfg_vocab or model_vocab
    if tokens.numel() and limit is not None:
        hi, lo = int(tokens.max()), int(tokens.min())
        if hi >= limit or lo < 0:
            raise TrainingError(
                f"Token ids span [{lo}, {hi}] but the vocabulary size is {limit}.",
                recipe=recipe, stage="batch",
                hint="Encode with the same tokenizer the model was built for "
                     "(vocab_size=tokenizer.vocab_size).",
            )
    ctx = model_context_length(model)
    if ctx is not None and tokens.dim() >= 2 and tokens.shape[-1] > ctx:
        raise TrainingError(
            f"Sequence length {tokens.shape[-1]} exceeds the model context length {ctx}.",
            recipe=recipe, stage="batch",
            hint=f"Use block_size<={ctx} in the dataset, or build the model with a larger "
                 "max_seq.",
        )


def _token_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int):
    valid = labels != ignore_index
    n = int(valid.sum())
    if n == 0:
        return 0.0, 0
    correct = (logits.argmax(-1) == labels) & valid
    return float(correct.sum()) / n, n


def _token_loss(cfg, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Built-in cross-entropy, or the configured override on flattened tokens."""
    flat_logits, flat_labels = logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
    if getattr(cfg, "loss", None) is None:
        return F.cross_entropy(flat_logits, flat_labels, ignore_index=cfg.ignore_index)
    return compute_loss(cfg, flat_logits, flat_labels, ignore_index=cfg.ignore_index)


def _token_val_metrics(cfg, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                           ignore_index=cfg.ignore_index)
    acc, n = _token_accuracy(logits, labels, cfg.ignore_index)
    out = {"loss": float(loss), "acc": acc, "_weight": float(max(n, 1))}
    if getattr(cfg, "loss", None) is not None:
        out["objective"] = float(_token_loss(cfg, logits, labels))
    return out


class CausalLMConfig(LossOverrideConfig):
    vocab_size: int | None = Field(
        default=None, description="Optional; checked against the model and tokenizer."
    )
    shift_labels: bool = True
    ignore_index: int = -100


@register_recipe("causal-lm", help="Causal language modeling (next-token cross-entropy).")
class CausalLMRecipe(Recipe):
    level = "loss"
    ConfigSchema = CausalLMConfig
    ComponentsSchema = None

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.01, "scheduler": "cosine"}

    def expected_batch_spec(self) -> dict[str, str]:
        return {
            "tokens": "LongTensor[B, T]",
            "labels": "LongTensor[B, T] (optional; defaults to tokens, shifted internally)",
            "mask": "LongTensor[B, T] (optional; 1 = token, 0 = padding)",
        }

    def optional_batch_keys(self) -> set[str]:
        return {"labels", "mask"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "tokens": torch.randint(0, 64, (4, 8), device=dev),
            "labels": torch.randint(0, 64, (4, 8), device=dev),
        }

    def preflight(self, model, batch, cfg: CausalLMConfig, components, adapter=None):
        _check_tokens(self.name, model, batch["tokens"], cfg.vocab_size)

    def _targets(self, batch: dict, cfg: CausalLMConfig) -> torch.Tensor:
        labels = batch.get("labels")
        labels = batch["tokens"] if labels is None else labels
        labels = labels.long()
        mask = batch.get("mask")
        if mask is not None:
            labels = labels.masked_fill(mask == 0, cfg.ignore_index)
        return labels

    def _logits_and_targets(self, model, batch, cfg: CausalLMConfig, adapter):
        self.validate_batch(batch, strict=True)
        inputs, keys = _token_inputs(batch)
        logits, aux = _as_logits(_forward(model, inputs, keys, adapter))
        labels = self._targets(batch, cfg)
        if cfg.shift_labels:
            logits, labels = logits[:, :-1], labels[:, 1:]
        return logits, labels, aux

    def batch_to_loss(self, model, batch, cfg: CausalLMConfig, components, adapter=None):
        from topologies.training.aux import apply_aux_losses

        logits, labels, aux = self._logits_and_targets(model, batch, cfg, adapter)
        loss = _token_loss(cfg, logits, labels)
        total, extra = apply_aux_losses(loss, model, aux)
        return total, {"loss": float(total.detach()), **extra}

    def validation_step(self, model, batch, cfg: CausalLMConfig, components, adapter=None):
        logits, labels, _aux = self._logits_and_targets(model, batch, cfg, adapter)
        return _token_val_metrics(cfg, logits, labels)

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        logits, labels, _aux = self._logits_and_targets(model, batch, cfg, adapter)
        return EvaluationStepResult(_token_val_metrics(cfg, logits, labels), EvaluationRecord(
            logits, labels, mask=labels != cfg.ignore_index, sample_ids=batch.get("sample_ids"), metadata=batch))

    def summarize_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if "loss" in metrics:
            metrics["ppl"] = math.exp(min(metrics["loss"], 50.0))
        return metrics

    def predict_step(self, model, batch, cfg, components, adapter=None):
        inputs, keys = _token_inputs(batch)
        return _call(model, inputs, keys, adapter)


class MaskedLMConfig(LossOverrideConfig):
    vocab_size: int | None = None
    ignore_index: int = -100


@register_recipe("masked-lm", help="Masked language modeling (BERT-style cross-entropy).")
class MaskedLMRecipe(Recipe):
    level = "loss"
    ConfigSchema = MaskedLMConfig
    ComponentsSchema = None

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.01, "scheduler": "linear"}

    def expected_batch_spec(self) -> dict[str, str]:
        return {
            "tokens": "LongTensor[B, T]",
            "labels": "LongTensor[B, T] (use -100 unmasked)",
            "mask": "LongTensor[B, T] (optional; 1 = token, 0 = padding)",
        }

    def optional_batch_keys(self) -> set[str]:
        return {"mask"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        tokens = torch.randint(0, 64, (4, 8), device=dev)
        labels = torch.full((4, 8), -100, dtype=torch.long, device=dev)
        labels[:, :2] = torch.randint(0, 64, (4, 2), device=dev)
        return {"tokens": tokens, "labels": labels}

    def preflight(self, model, batch, cfg: MaskedLMConfig, components, adapter=None):
        _check_tokens(self.name, model, batch["tokens"], cfg.vocab_size)

    def _logits(self, model, batch, adapter):
        self.validate_batch(batch, strict=True)
        inputs, keys = _token_inputs(batch)
        return _call(model, inputs, keys, adapter)

    def batch_to_loss(self, model, batch, cfg: MaskedLMConfig, components, adapter=None):
        logits = self._logits(model, batch, adapter)
        loss = _token_loss(cfg, logits, batch["labels"].long())
        return loss, {"loss": float(loss.detach())}

    def validation_step(self, model, batch, cfg: MaskedLMConfig, components, adapter=None):
        logits = self._logits(model, batch, adapter)
        return _token_val_metrics(cfg, logits, batch["labels"].long())

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        logits, labels = self._logits(model, batch, adapter), batch["labels"].long()
        return EvaluationStepResult(_token_val_metrics(cfg, logits, labels), EvaluationRecord(
            logits, labels, mask=labels != cfg.ignore_index, sample_ids=batch.get("sample_ids"), metadata=batch))

    def summarize_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if "loss" in metrics:
            metrics["ppl"] = math.exp(min(metrics["loss"], 50.0))
        return metrics

    def predict_step(self, model, batch, cfg, components, adapter=None):
        inputs, keys = _token_inputs(batch)
        return _call(model, inputs, keys, adapter)


class AlbertPretrainingConfig(RecipeConfig):
    vocab_size: int | None = None
    ignore_index: int = -100
    sop_weight: float = Field(default=1.0, ge=0.0)
    pad_token_id: int = 0
    unknown_token_id: int = 1
    cls_token_id: int = 2
    sep_token_id: int = 3
    mask_token_id: int = 4
    special_token_ids: tuple[int, ...] = (0, 1, 2, 3, 4)


@register_recipe("albert-pretraining", help="Joint ALBERT masked-language modeling and sentence-order prediction.")
class AlbertPretrainingRecipe(Recipe):
    level = "loss"
    ConfigSchema = AlbertPretrainingConfig
    ComponentsSchema = None

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 1e-4, "weight_decay": 0.01, "scheduler": "linear"}

    def expected_batch_spec(self) -> dict[str, str]:
        return {"tokens": "LongTensor[B, T]", "labels": "LongTensor[B, T] (masked tokens; -100 elsewhere)",
                "sop_labels": "LongTensor[B] (0=original order, 1=swapped)",
                "token_type_ids": "LongTensor[B, T]", "mask": "BoolTensor[B, T] (optional)"}

    def optional_batch_keys(self) -> set[str]:
        return {"mask"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        batch = self.synthetic_batch(vocab_size=64, context_length=8, batch_size=2,
                                     seed=0, config=type(self).ConfigSchema())
        return {key: value.to(device) for key, value in batch.items()}

    def synthetic_batch(self, *, vocab_size: int, context_length: int, batch_size: int = 4,
                        seed: int = 0, config=None) -> dict[str, torch.Tensor]:
        if config is None:
            config = type(self).ConfigSchema(vocab_size=vocab_size)
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        if context_length < 5:
            raise ValueError("ALBERT sentence-pair pretraining requires context_length >= 5")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        named_special = []
        for name in ("pad_token_id", "unknown_token_id", "cls_token_id", "sep_token_id", "mask_token_id"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < vocab_size:
                raise ValueError(f"{name} must be an integer in [0, vocab_size)")
            named_special.append(value)
        if len(set(named_special)) != len(named_special):
            raise ValueError("ALBERT pad/unknown/CLS/SEP/MASK IDs must be distinct")
        special = tuple({*config.special_token_ids, *named_special})
        if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < vocab_size
               for value in special):
            raise ValueError("special_token_ids must contain vocabulary IDs")
        ordinary = [token for token in range(vocab_size) if token not in special]
        if len(ordinary) < 2:
            raise ValueError("vocabulary needs at least two ordinary tokens for sentence-pair smoke data")
        from topologies.models.bert_encoder import ALBERTPretrainingCollator

        documents = []
        for row in range((batch_size + 1) // 2):
            left = ordinary[(seed + row * 2) % len(ordinary)]
            right = ordinary[(seed + row * 2 + 1) % len(ordinary)]
            documents.append([[left], [right]])
        collator = ALBERTPretrainingCollator(
            vocab_size=vocab_size, mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id, cls_token_id=config.cls_token_id,
            sep_token_id=config.sep_token_id, special_token_ids=special, seed=seed)
        batch = collator(documents)
        return {key: value[:batch_size] for key, value in batch.items()}

    def _outputs(self, model, batch, adapter=None):
        self.validate_batch(batch, strict=True)
        inputs = {key: batch[key] for key in ("tokens", "token_type_ids")}
        if batch.get("mask") is not None:
            inputs["mask"] = batch["mask"].bool()
        if hasattr(model, "forward_pretraining"):
            model_inputs = adapter.adapted_batch(inputs) if adapter is not None and hasattr(adapter, "adapted_batch") else inputs
            outputs = model.forward_pretraining(**model_inputs)
            if adapter is not None and getattr(adapter, "output_map", None) is not None:
                outputs = adapter.output_map(outputs)
        elif adapter is not None:
            outputs = call_with_adapter(adapter, model, inputs, list(inputs), primary="tokens")
        else:
            raise TrainingError("albert-pretraining requires model.forward_pretraining().", recipe=self.name, stage="model")
        if not isinstance(outputs, dict) or not {"mlm_logits", "sop_logits"} <= outputs.keys():
            raise TrainingError("forward_pretraining must return mlm_logits and sop_logits.", recipe=self.name, stage="model")
        return outputs

    def _losses(self, outputs, batch, cfg):
        labels = batch["labels"].long()
        labels = batch["labels"].long()
        if labels.ne(cfg.ignore_index).any():
            mlm_loss = F.cross_entropy(outputs["mlm_logits"].reshape(-1, outputs["mlm_logits"].size(-1)),
                                       labels.reshape(-1), ignore_index=cfg.ignore_index)
        else:
            mlm_loss = outputs["mlm_logits"].sum() * 0
        sop_loss = F.cross_entropy(outputs["sop_logits"], batch["sop_labels"].long())
        return mlm_loss, sop_loss, mlm_loss + cfg.sop_weight * sop_loss

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        _check_tokens(self.name, model, batch["tokens"], cfg.vocab_size)
        outputs = self._outputs(model, batch, adapter)
        mlm, sop, total = self._losses(outputs, batch, cfg)
        return total, {"loss": float(total.detach()), "mlm_loss": float(mlm.detach()),
                       "sop_loss": float(sop.detach())}

    def _metrics(self, outputs, batch, cfg):
        mlm, sop, total = self._losses(outputs, batch, cfg)
        valid = batch["labels"].ne(cfg.ignore_index)
        mlm_accuracy, _ = _token_accuracy(outputs["mlm_logits"], batch["labels"].long(), cfg.ignore_index)
        sop_accuracy = float((outputs["sop_logits"].argmax(-1) == batch["sop_labels"]).float().mean())
        return {"loss": float(total.detach()), "mlm_loss": float(mlm.detach()),
                "sop_loss": float(sop.detach()), "mlm_acc": mlm_accuracy,
                "sop_acc": sop_accuracy, "_weight": float(max(int(valid.sum()), 1))}

    def validation_step(self, model, batch, cfg, components, adapter=None):
        return self._metrics(self._outputs(model, batch, adapter), batch, cfg)

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        outputs = self._outputs(model, batch, adapter)
        metrics = self._metrics(outputs, batch, cfg)
        return EvaluationStepResult(metrics, EvaluationRecord(outputs["mlm_logits"], batch["labels"].long(),
            mask=batch["labels"].ne(cfg.ignore_index), sample_ids=batch.get("sample_ids"), metadata=batch))

    def predict_step(self, model, batch, cfg, components, adapter=None):
        return self._outputs(model, batch, adapter)


class DistilBertPretrainingConfig(RecipeConfig):
    vocab_size: int | None = None
    ignore_index: int = -100
    temperature: float = Field(default=2.0, gt=0)
    mlm_weight: float = Field(default=2.0, ge=0)
    distillation_weight: float = Field(default=5.0, ge=0)
    cosine_weight: float = Field(default=1.0, ge=0)
    pad_token_id: int = 0
    unknown_token_id: int = 100
    cls_token_id: int = 101
    sep_token_id: int = 102
    mask_token_id: int = 103
    special_token_ids: tuple[int, ...] = (0, 100, 101, 102, 103)

    @field_validator("temperature", "mlm_weight", "distillation_weight", "cosine_weight")
    @classmethod
    def _finite_hyperparameter(cls, value):
        if not math.isfinite(value):
            raise ValueError("DistilBERT loss weights and temperature must be finite")
        return value


def distilbert_special_token_ids(vocab_size: int) -> tuple[int, int, int, int, int]:
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 6:
        raise ValueError("DistilBERT pretraining needs a vocabulary of at least six tokens")
    canonical = (0, 100, 101, 102, 103)
    return canonical if vocab_size > max(canonical) else (0, 1, 2, 3, 4)


class DistilBertPretrainingComponents(RecipeComponents):
    teacher_model: nn.Module


@register_recipe("distilbert-pretraining", help="DistilBERT MLM plus frozen-teacher KL and hidden-state cosine distillation.")
class DistilBertPretrainingRecipe(Recipe):
    level = "loss"
    ConfigSchema = DistilBertPretrainingConfig
    ComponentsSchema = DistilBertPretrainingComponents

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 1e-4, "weight_decay": 0.01, "scheduler": "linear"}

    def expected_batch_spec(self) -> dict[str, str]:
        return {"tokens": "LongTensor[B, T] (same masked view for student and teacher)",
                "labels": "LongTensor[B, T] (-100 except selected MLM targets)",
                "mask": "BoolTensor[B, T] (1=valid token, 0=padding; optional)"}

    def optional_batch_keys(self) -> set[str]:
        return {"mask"}

    def dummy_batch(self, device="cpu"):
        controls = distilbert_special_token_ids(64)
        cfg = DistilBertPretrainingConfig(vocab_size=64, pad_token_id=0,
            unknown_token_id=controls[1], cls_token_id=controls[2],
            sep_token_id=controls[3], mask_token_id=controls[4],
            special_token_ids=controls)
        return {key: value.to(device) for key, value in self.synthetic_batch(
            vocab_size=64, context_length=8, batch_size=2, seed=0, config=cfg).items()}

    def synthetic_batch(self, *, vocab_size: int, context_length: int, batch_size: int = 4,
                        seed: int = 0, mask_seed: int | None = None, config=None):
        from topologies.models.bert_encoder import DynamicMaskingCollator

        default_controls = distilbert_special_token_ids(vocab_size)
        cfg = config or self.ConfigSchema(vocab_size=vocab_size, pad_token_id=default_controls[0],
            unknown_token_id=default_controls[1], cls_token_id=default_controls[2],
            sep_token_id=default_controls[3], mask_token_id=default_controls[4],
            special_token_ids=default_controls)
        for name, value in (("vocab_size", vocab_size), ("context_length", context_length),
                            ("batch_size", batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if context_length < 3:
            raise ValueError("DistilBERT pretraining requires context_length >= 3")
        if cfg.vocab_size is not None and cfg.vocab_size != vocab_size:
            raise ValueError("recipe vocab_size must match generated batch vocab_size")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if mask_seed is None:
            mask_seed = seed
        if isinstance(mask_seed, bool) or not isinstance(mask_seed, int) or mask_seed < 0:
            raise ValueError("mask_seed must be a non-negative integer")
        controls = (cfg.pad_token_id, cfg.unknown_token_id, cfg.cls_token_id,
                    cfg.sep_token_id, cfg.mask_token_id, *cfg.special_token_ids)
        if len(set(controls[:5])) != 5 or any(isinstance(i, bool) or not isinstance(i, int)
                                              or not 0 <= i < vocab_size for i in controls):
            raise ValueError("all configured special-token IDs must be distinct vocabulary IDs")
        ordinary = [i for i in range(vocab_size) if i not in set(controls)]
        if not ordinary:
            raise ValueError("vocabulary must contain at least one ordinary token")
        rows = []
        for row in range(batch_size):
            half_context = max(1, context_length // 2)
            body_len = max(1, min(context_length - 2,
                                  half_context + row % max(1, half_context)))
            body = [ordinary[(seed + row + offset) % len(ordinary)] for offset in range(body_len)]
            rows.append([cfg.cls_token_id, *body, cfg.sep_token_id])
        tokens = nn.utils.rnn.pad_sequence([torch.tensor(row, dtype=torch.long) for row in rows],
                                           batch_first=True, padding_value=cfg.pad_token_id)
        collator = DynamicMaskingCollator(vocab_size, cfg.mask_token_id, cfg.pad_token_id,
            tuple(set(controls)), probability=0.15, seed=mask_seed)
        return collator(tokens)

    def _outputs(self, model, batch, teacher, cfg):
        if not isinstance(batch, dict) or "tokens" not in batch or "labels" not in batch:
            raise TrainingError("DistilBERT batches require tokens and MLM labels",
                                recipe=self.name, stage="batch")
        tokens = batch["tokens"]
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long:
            raise TrainingError("tokens must be a LongTensor shaped [B,T]",
                                recipe=self.name, stage="batch")
        if not tokens.shape[0] or not tokens.shape[1]:
            raise TrainingError("tokens must have non-empty batch and time dimensions",
                                recipe=self.name, stage="batch")
        _check_tokens(self.name, model, tokens, cfg.vocab_size)
        mask = batch.get("mask")
        mask = tokens.ne(cfg.pad_token_id) if mask is None else mask
        labels = batch["labels"]
        if not isinstance(labels, torch.Tensor):
            raise TrainingError("labels must be a LongTensor", recipe=self.name, stage="batch")
        if not isinstance(mask, torch.Tensor):
            raise TrainingError("mask must contain boolean or numeric 0/1 values",
                                recipe=self.name, stage="batch")
        if labels.shape != tokens.shape or mask.shape != tokens.shape:
            raise TrainingError("tokens, labels, and mask must have matching [B,T] shapes",
                                recipe=self.name, stage="batch")
        if mask.device != tokens.device or labels.device != tokens.device:
            raise TrainingError("tokens, labels, and mask must be on the same device",
                                recipe=self.name, stage="batch")
        if mask.dtype != torch.bool and (mask.is_floating_point() or mask.dtype in
                (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
            if not bool(torch.isfinite(mask).all()) or not bool(((mask == 0) | (mask == 1)).all()):
                raise TrainingError("mask must contain finite 0/1 values", recipe=self.name, stage="batch")
        elif mask.dtype != torch.bool:
            raise TrainingError("mask must contain boolean or numeric 0/1 values", recipe=self.name, stage="batch")
        mask = mask.bool()
        if labels.dtype != torch.long:
            raise TrainingError("labels must be a LongTensor", recipe=self.name, stage="batch")
        if bool((labels.ne(cfg.ignore_index) & ~mask).any()):
            raise TrainingError("padding positions cannot carry MLM labels", recipe=self.name, stage="batch")
        if not bool((labels.ne(cfg.ignore_index) & mask).any()):
            raise TrainingError("batch contains no supervised MLM tokens", recipe=self.name, stage="batch")
        if not hasattr(model, "forward_pretraining") or not hasattr(teacher, "forward_pretraining"):
            raise TrainingError("student and teacher must provide forward_pretraining()", recipe=self.name, stage="model")
        student_vocab = model_vocab_size(model)
        teacher_vocab = model_vocab_size(teacher)
        if student_vocab is None or teacher_vocab != student_vocab or getattr(
                teacher, "context_length", 0) < tokens.shape[1]:
            raise TrainingError("teacher vocabulary and context must cover the student's batch",
                                recipe=self.name, stage="model")
        if ({id(module) for module in model.modules()} & {id(module) for module in teacher.modules()}
                or {id(parameter) for parameter in model.parameters()} &
                {id(parameter) for parameter in teacher.parameters()}):
            raise TrainingError("teacher and student must not share modules or parameters",
                                recipe=self.name, stage="components")
        targets = labels[labels.ne(cfg.ignore_index)]
        if bool((targets < 0).any()) or bool((targets >= student_vocab).any()):
            raise TrainingError("supervised MLM labels must be valid vocabulary IDs",
                                recipe=self.name, stage="batch")
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.to(tokens.device)
        student_output = model.forward_pretraining(tokens, mask)
        with torch.no_grad():
            teacher_output = teacher.forward_pretraining(tokens, mask)
        needed = ("mlm_logits", "last_hidden_state")
        if any(not isinstance(out, dict) or any(key not in out for key in needed)
               for out in (student_output, teacher_output)):
            raise TrainingError("forward_pretraining must return mlm_logits and last_hidden_state",
                                recipe=self.name, stage="model")
        sl, sh = student_output["mlm_logits"], student_output["last_hidden_state"]
        tl, th = teacher_output["mlm_logits"], teacher_output["last_hidden_state"]
        if sl.shape != tl.shape or sl.shape[:2] != tokens.shape or sh.shape != th.shape or sh.shape[:2] != tokens.shape:
            raise TrainingError("teacher/student logits and hidden states must match batch, context and vocabulary",
                                recipe=self.name, stage="model")
        if sl.shape[-1] != model_vocab_size(model) or tl.shape[-1] != model_vocab_size(teacher):
            raise TrainingError("teacher and student vocabulary dimensions must match their declared vocabularies",
                                recipe=self.name, stage="model")
        return student_output, teacher_output, labels, mask

    def _loss(self, model, batch, cfg, components):
        if components is None or components.teacher_model is None:
            raise TrainingError("distilbert-pretraining requires components.teacher_model",
                                recipe=self.name, stage="components")
        student, teacher, labels, mask = self._outputs(model, batch, components.teacher_model, cfg)
        logits, target_logits = student["mlm_logits"], teacher["mlm_logits"].detach()
        hidden, target_hidden = student["last_hidden_state"], teacher["last_hidden_state"].detach()
        selected = labels.ne(cfg.ignore_index) & mask
        mlm = F.cross_entropy(logits[selected], labels[selected])
        t = cfg.temperature
        valid_student = F.log_softmax(logits[mask] / t, dim=-1)
        valid_teacher = F.softmax(target_logits[mask] / t, dim=-1)
        kl = F.kl_div(valid_student, valid_teacher, reduction="batchmean") * t * t
        cosine = (1 - F.cosine_similarity(hidden[mask], target_hidden[mask], dim=-1)).mean()
        total = cfg.mlm_weight * mlm + cfg.distillation_weight * kl + cfg.cosine_weight * cosine
        return total, {"loss": total, "mlm_loss": mlm, "distillation_loss": kl, "cosine_loss": cosine}

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        total, values = self._loss(model, batch, cfg, components)
        return total, {name: float(value.detach()) for name, value in values.items()}

    def validation_step(self, model, batch, cfg, components, adapter=None):
        _, values = self._loss(model, batch, cfg, components)
        return {name: float(value.detach()) for name, value in values.items()}

    def predict_step(self, model, batch, cfg, components, adapter=None):
        mask = batch.get("mask", batch["tokens"].ne(cfg.pad_token_id)).bool()
        return model.forward_pretraining(batch["tokens"], mask)


class Seq2SeqLMConfig(RecipeConfig):
    vocab_size: int | None = None
    ignore_index: int = -100


@register_recipe("seq2seq-lm", help="Encoder-decoder cross-entropy (T5-style).")
class Seq2SeqLMRecipe(Recipe):
    level = "loss"
    ConfigSchema = Seq2SeqLMConfig
    ComponentsSchema = None

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.01, "scheduler": "linear"}

    def expected_batch_spec(self) -> dict[str, str]:
        return {
            "src": "LongTensor[B, S]",
            "tgt": "LongTensor[B, T]",
            "labels": "LongTensor[B, T]",
        }

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "src": torch.randint(0, 64, (4, 8), device=dev),
            "tgt": torch.randint(0, 64, (4, 8), device=dev),
            "labels": torch.randint(0, 64, (4, 8), device=dev),
        }

    def batch_to_loss(self, model, batch, cfg: Seq2SeqLMConfig, components, adapter=None):
        self.validate_batch(batch, strict=True)
        logits = _call(model, {"src": batch["src"], "tgt": batch["tgt"]}, ["src", "tgt"],
                       adapter)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch["labels"].reshape(-1).long(),
            ignore_index=cfg.ignore_index,
        )
        return loss, {"loss": float(loss.detach())}

    def validation_step(self, model, batch, cfg: Seq2SeqLMConfig, components, adapter=None):
        """Loss plus teacher-forced token accuracy and sequence exact match."""
        logits = _call(model, {"src": batch["src"], "tgt": batch["tgt"]}, ["src", "tgt"],
                       adapter)
        labels = batch["labels"].long()
        out = _token_val_metrics(cfg, logits, labels)
        valid = labels != cfg.ignore_index
        right = (logits.argmax(-1) == labels) | ~valid
        exact = right.all(-1).float().mean()
        out["exact_match"] = float(exact)
        return out

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        logits = _call(model, {"src": batch["src"], "tgt": batch["tgt"]}, ["src", "tgt"], adapter)
        labels = batch["labels"].long()
        out = _token_val_metrics(cfg, logits, labels)
        valid = labels != cfg.ignore_index
        out["exact_match"] = float(((logits.argmax(-1) == labels) | ~valid).all(-1).float().mean())
        return EvaluationStepResult(out, EvaluationRecord(logits, labels, mask=valid,
                                    sample_ids=batch.get("sample_ids"), metadata=batch))

    def predict_step(self, model, batch, cfg, components, adapter=None):
        return _call(model, {"src": batch["src"], "tgt": batch["tgt"]}, ["src", "tgt"], adapter)


class ElectraConfig(RecipeConfig):
    pass


@register_recipe("electra", help="ELECTRA replaced-token detection (per-token BCE).")
class ElectraRecipe(Recipe):
    level = "loss"
    ConfigSchema = ElectraConfig
    ComponentsSchema = None

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 2e-4, "weight_decay": 0.01, "scheduler": "linear"}

    def expected_batch_spec(self) -> dict[str, str]:
        return {"tokens": "LongTensor[B, T]", "labels": "FloatTensor[B, T] (0/1)"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "tokens": torch.randint(0, 64, (4, 8), device=dev),
            "labels": torch.randint(0, 2, (4, 8), device=dev).float(),
        }

    def batch_to_loss(self, model, batch, cfg: ElectraConfig, components, adapter=None):
        self.validate_batch(batch, strict=True)
        logits = _call(model, {"tokens": batch["tokens"]}, ["tokens"], adapter)
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"].float())
        return loss, {"loss": float(loss.detach())}


def electra_pretraining_special_token_ids(vocab_size: int) -> tuple[int, int, int, int, int]:
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 5:
        raise ValueError("ELECTRA vocabulary must have at least six token IDs")
    return (0, 100, 101, 102, 103) if vocab_size > 103 else (0, 1, 2, 3, 4)


class ElectraPretrainingConfig(RecipeConfig):
    vocab_size: int | None = None
    pad_token_id: int = 0
    unknown_token_id: int = 100
    cls_token_id: int = 101
    sep_token_id: int = 102
    mask_token_id: int = 103
    special_token_ids: tuple[int, ...] = (0, 100, 101, 102, 103)
    mask_probability: float = Field(default=0.15, gt=0, le=1)
    temperature: float = Field(default=1.0, gt=0)
    generator_weight: float = Field(default=1.0, ge=0)
    discriminator_weight: float = Field(default=50.0, ge=0)

    @field_validator("mask_probability", "temperature", "generator_weight", "discriminator_weight")
    @classmethod
    def _finite_electra_values(cls, value):
        if not math.isfinite(value):
            raise ValueError("ELECTRA mask probability and loss weights must be finite")
        return value


@register_recipe("electra-pretraining", help="Joint ELECTRA generator MLM and sampled replaced-token discrimination.")
class ElectraPretrainingRecipe(Recipe):
    level = "loss"
    ConfigSchema = ElectraPretrainingConfig
    ComponentsSchema = None

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 2e-4, "weight_decay": 0.01, "scheduler": "linear"}

    def expected_batch_spec(self) -> dict[str, str]:
        return {"tokens": "LongTensor[B,T] original tokens", "mask": "BoolTensor[B,T] valid-token scope (optional)",
                "token_type_ids": "LongTensor[B,T] segments (optional)"}

    def optional_batch_keys(self) -> set[str]:
        return {"mask", "token_type_ids"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        controls = electra_pretraining_special_token_ids(64)
        config = self.ConfigSchema(vocab_size=64,
            pad_token_id=controls[0], unknown_token_id=controls[1],
            cls_token_id=controls[2], sep_token_id=controls[3],
            mask_token_id=controls[4], special_token_ids=controls)
        batch = self.synthetic_batch(vocab_size=64, context_length=8, batch_size=4,
                                     seed=0, config=config)
        return {key: value.to(device) for key, value in batch.items()}

    def synthetic_batch(self, *, vocab_size: int, context_length: int, batch_size: int = 4,
                        seed: int = 0, config=None, mask_seed: int | None = None):
        if config is None:
            controls = electra_pretraining_special_token_ids(vocab_size)
            config = self.ConfigSchema(vocab_size=vocab_size,
                pad_token_id=controls[0], unknown_token_id=controls[1],
                cls_token_id=controls[2], sep_token_id=controls[3],
                mask_token_id=controls[4], special_token_ids=controls)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (vocab_size, context_length, batch_size)):
            raise ValueError("vocab_size, context_length, and batch_size must be positive integers")
        controls = (config.pad_token_id, config.unknown_token_id, config.cls_token_id,
                    config.sep_token_id, config.mask_token_id)
        if len(set(controls)) != len(controls) or any(not 0 <= token < vocab_size for token in controls):
            raise ValueError("ELECTRA control token IDs must be distinct IDs in vocabulary")
        special = set(config.special_token_ids) | set(controls)
        if any(isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < vocab_size
               for token in special):
            raise ValueError("ELECTRA special token IDs must be vocabulary IDs")
        ordinary = [token for token in range(vocab_size) if token not in special]
        if not ordinary:
            raise ValueError("vocabulary needs ordinary non-special tokens")
        rng = torch.Generator().manual_seed(seed if mask_seed is None else mask_seed)
        values = torch.tensor(ordinary, dtype=torch.long)
        tokens = values[torch.randint(len(ordinary), (batch_size, context_length), generator=rng)]
        tokens[:, 0] = config.cls_token_id
        if context_length > 1:
            tokens[:, -1] = config.sep_token_id
        return {"tokens": tokens, "mask": tokens.ne(config.pad_token_id),
                "token_type_ids": torch.zeros_like(tokens)}

    def _forward(self, model, batch, cfg):
        self.validate_batch(batch, strict=True)
        if not isinstance(batch["tokens"], torch.Tensor) or batch["tokens"].dtype != torch.long:
            raise TrainingError("ELECTRA tokens must be a LongTensor.", recipe=self.name, stage="batch")
        original = batch["tokens"]
        if original.ndim != 2 or not original.shape[0] or not original.shape[1]:
            raise TrainingError("ELECTRA tokens must be a non-empty [batch,time] tensor.",
                                recipe=self.name, stage="batch")
        _check_tokens(self.name, model, original, cfg.vocab_size)
        batch_mask = batch.get("mask")
        if batch_mask is None:
            valid = original.ne(cfg.pad_token_id)
        else:
            if (not isinstance(batch_mask, torch.Tensor) or batch_mask.shape != original.shape
                    or batch_mask.device != original.device):
                raise TrainingError("ELECTRA mask must match token batch/time shape.",
                                    recipe=self.name, stage="batch")
            if batch_mask.dtype != torch.bool and not bool(((batch_mask == 0) | (batch_mask == 1)).all()):
                raise TrainingError("ELECTRA mask must contain only 0/1 validity values.",
                                    recipe=self.name, stage="batch")
            valid = batch_mask.bool()
            valid = valid & original.ne(cfg.pad_token_id)
        named_controls = (cfg.pad_token_id, cfg.unknown_token_id, cfg.cls_token_id,
                          cfg.sep_token_id, cfg.mask_token_id)
        if len(set(named_controls)) != len(named_controls) or any(
                isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < model.vocab_size
                for i in (*named_controls, *cfg.special_token_ids)):
            raise TrainingError("ELECTRA control IDs must be distinct vocabulary IDs.",
                                recipe=self.name, stage="config")
        if (cfg.pad_token_id != getattr(model, "pad_token_id", cfg.pad_token_id)
                or cfg.mask_token_id != getattr(model, "mask_token_id", cfg.mask_token_id)):
            raise TrainingError("ELECTRA recipe pad/mask IDs must match model constructor IDs.",
                                recipe=self.name, stage="config")
        eligible = valid.clone()
        for token_id in set(named_controls) | set(cfg.special_token_ids):
            eligible &= original.ne(token_id)
        if not bool(eligible.any(dim=1).all()):
            raise TrainingError("Every row must include an ordinary token eligible for ELECTRA masking.",
                                recipe=self.name, stage="batch")
        selected = (torch.rand(original.shape, device=original.device) < cfg.mask_probability) & eligible
        for row in range(selected.shape[0]):
            if not bool(selected[row].any()):
                positions = eligible[row].nonzero(as_tuple=True)[0]
                choice = torch.randint(positions.numel(), (1,), device=original.device)
                selected[row, positions[choice]] = True
        masked = original.masked_fill(selected, cfg.mask_token_id)
        types = batch.get("token_type_ids")
        if types is not None and (not isinstance(types, torch.Tensor) or types.shape != original.shape
                                  or types.dtype != torch.long or types.device != original.device):
            raise TrainingError("ELECTRA token_type_ids must be a matching LongTensor.",
                                recipe=self.name, stage="batch")
        outputs = model.pretraining_forward(original, masked, selected, valid, types,
                                            tuple(set(named_controls) | set(cfg.special_token_ids)),
                                            cfg.temperature)
        if not isinstance(outputs, dict) or not {"generator_logits", "discriminator_logits",
                                                  "replaced_labels"} <= outputs.keys():
            raise TrainingError("ELECTRA forward_pretraining output contract is invalid.",
                                recipe=self.name, stage="model")
        if (outputs["generator_logits"].shape != (*original.shape, model.vocab_size)
                or outputs["discriminator_logits"].shape != original.shape
                or outputs["replaced_labels"].shape != original.shape
                or outputs.get("corrupted_tokens", torch.empty(0)).shape != original.shape):
            raise TrainingError("ELECTRA logits and corruption labels must match model batch/time/vocabulary geometry.",
                                recipe=self.name, stage="model")
        return outputs, original, selected, valid

    def _losses(self, outputs, original, selected, valid, cfg):
        mlm_labels = original.masked_fill(~selected, -100)
        generator_loss = F.cross_entropy(outputs["generator_logits"].reshape(-1, outputs["generator_logits"].size(-1)),
                                         mlm_labels.reshape(-1), ignore_index=-100)
        discriminator_loss = F.binary_cross_entropy_with_logits(
            outputs["discriminator_logits"][valid], outputs["replaced_labels"].float()[valid])
        total = cfg.generator_weight * generator_loss + cfg.discriminator_weight * discriminator_loss
        return mlm_labels, generator_loss, discriminator_loss, total

    def batch_to_loss(self, model, batch, cfg, components, adapter=None):
        outputs, original, selected, valid = self._forward(model, batch, cfg)
        _, gen, disc, total = self._losses(outputs, original, selected, valid, cfg)
        return total, {"loss": float(total.detach()), "generator_loss": float(gen.detach()),
                       "discriminator_loss": float(disc.detach())}

    def validation_step(self, model, batch, cfg, components, adapter=None):
        outputs, original, selected, valid = self._forward(model, batch, cfg)
        labels, gen, disc, total = self._losses(outputs, original, selected, valid, cfg)
        mlm_acc, _ = _token_accuracy(outputs["generator_logits"], labels, -100)
        rtd_acc = float((((outputs["discriminator_logits"] >= 0) == outputs["replaced_labels"].bool())
                         & valid).sum() / valid.sum().clamp_min(1))
        return {"loss": float(total.detach()), "generator_loss": float(gen.detach()),
                "discriminator_loss": float(disc.detach()), "mlm_acc": mlm_acc,
                "rtd_acc": rtd_acc, "_weight": float(valid.sum())}

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        outputs, original, selected, valid = self._forward(model, batch, cfg)
        labels, gen, disc, total = self._losses(outputs, original, selected, valid, cfg)
        mlm_acc, _ = _token_accuracy(outputs["generator_logits"], labels, -100)
        rtd_acc = float((((outputs["discriminator_logits"] >= 0) == outputs["replaced_labels"].bool())
                         & valid).sum() / valid.sum().clamp_min(1))
        metrics = {"loss": float(total.detach()), "generator_loss": float(gen.detach()),
                   "discriminator_loss": float(disc.detach()), "mlm_acc": mlm_acc,
                   "rtd_acc": rtd_acc, "_weight": float(valid.sum())}
        record = EvaluationRecord(outputs["discriminator_logits"],
            outputs["replaced_labels"].long(), mask=valid,
            sample_ids=batch.get("sample_ids"), metadata=batch)
        return EvaluationStepResult(metrics, record)


class _EmptyComponents(RecipeComponents):
    pass
