"""Classification / regression recipes for text, tokens and tabular data.

- ``text-classify``: ``{"tokens", "labels"}`` (+ optional ``mask``) -> logits ``[B, C]``;
  cross-entropy with label smoothing (or a ``loss`` override).
- ``token-classify``: per-token labels ``[B, T]`` (``-100`` ignored, padding masked) ->
  logits ``[B, T, C]`` (NER / span tagging).
- ``multilabel-classify``: ``{"tokens" | "features", "labels": FloatTensor[B, C]}`` ->
  logits ``[B, C]``; binary cross-entropy, ``acc`` = exact set match, ``f1`` = micro F1.
- ``tabular-classify``: ``{"features": FloatTensor[B, F], "labels"}`` (+ optional
  ``categorical`` LongTensor[B, K]) -> logits ``[B, C]``.
- ``regression``: ``{"inputs", "targets"}`` -> predictions shaped like ``targets``;
  MSE or Huber; ``mae`` reported.

Models receive ``mask`` / ``categorical`` only when their ``forward`` accepts them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pydantic import Field, model_validator

from topologies.evaluation import EvaluationRecord
from topologies.training.adapters import call_with_adapter
from topologies.training.errors import TrainingError
from topologies.training.losses import LossOverrideConfig, compute_loss
from topologies.training.recipe_base import (
    EvaluationStepResult,
    Recipe,
    RecipeConfig,
    register_recipe,
)


def _logits(out: object, recipe: str) -> torch.Tensor:
    if isinstance(out, dict):
        out = out.get("logits", next(iter(out.values())))
    if isinstance(out, (tuple, list)):
        out = next((o for o in out if isinstance(o, torch.Tensor) and o.is_floating_point()),
                   out[0])
    if not isinstance(out, torch.Tensor):
        raise TrainingError(f"Model returned {type(out).__name__}; {recipe} needs a tensor.",
                            recipe=recipe, stage="loss",
                            hint="Return logits from forward() (or output_map=...).")
    return out


def _inputs(batch: dict, primary: str, optional: tuple[str, ...] = ("mask",)) -> tuple[dict,
                                                                                       list[str]]:
    inputs = {primary: batch[primary]}
    for k in optional:
        if batch.get(k) is not None:
            inputs[k] = batch[k]
    return inputs, list(inputs)


def _primary(batch: dict, candidates: tuple[str, ...], recipe: str) -> str:
    for k in candidates:
        if k in batch:
            return k
    raise TrainingError(f"{recipe} needs one of {list(candidates)} in the batch.",
                        recipe=recipe, stage="batch",
                        received=", ".join(sorted(batch)) or "(empty batch)",
                        hint=f"Name the input field {candidates[0]!r}.")


def _check_classes(recipe: str, logits: torch.Tensor, labels: torch.Tensor, num_classes: int | None,
                   ignore_index: int | None = None) -> None:
    n = logits.shape[-1]
    if num_classes is not None and n != num_classes:
        raise TrainingError(f"Model outputs {n} classes but recipe_config num_classes="
                            f"{num_classes}.", recipe=recipe, stage="config",
                            config_key="num_classes",
                            hint="Build the model with num_classes matching the labels.")
    lab = labels if ignore_index is None else labels[labels != ignore_index]
    if lab.numel():
        lo, hi = int(lab.min()), int(lab.max())
        if lo < 0 or hi >= n:
            raise TrainingError(f"Labels span [{lo}, {hi}] but the model outputs {n} classes.",
                                recipe=recipe, stage="batch",
                                hint=f"Build the model with num_classes>={hi + 1}.")


class TextClassifyConfig(LossOverrideConfig):
    """``loss`` / ``loss_kwargs`` override ``fn(logits, labels, **kw)``."""

    num_classes: int | None = None
    vocab_size: int | None = None
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _smoothing_with_override(self) -> TextClassifyConfig:
        if self.loss is not None and self.label_smoothing:
            raise ValueError("label_smoothing applies to the built-in loss only.")
        return self


@register_recipe("text-classify", help="Sequence classification from token ids (cross-entropy).")
class TextClassifyRecipe(Recipe):
    level = "loss"
    ConfigSchema = TextClassifyConfig
    ComponentsSchema = None

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 5e-4, "weight_decay": 0.01, "scheduler": "cosine"}

    def expected_batch_spec(self) -> dict[str, str]:
        return {"tokens": "LongTensor[B, T]", "labels": "LongTensor[B]",
                "mask": "LongTensor[B, T] (optional; 1 = token, 0 = padding)"}

    def optional_batch_keys(self) -> set[str]:
        return {"mask"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {"tokens": torch.randint(0, 64, (4, 8), device=dev),
                "labels": torch.randint(0, 3, (4,), device=dev)}

    def forward(self, model, batch: dict, adapter=None) -> torch.Tensor:
        inputs, keys = _inputs(batch, "tokens")
        return _logits(call_with_adapter(adapter, model, inputs, keys, primary="tokens"),
                       self.name)

    def preflight(self, model, batch, cfg: TextClassifyConfig, components, adapter=None):
        from topologies.training.recipes.lm import _check_tokens

        _check_tokens(self.name, model, batch["tokens"], cfg.vocab_size)
        was = model.training
        model.eval()
        try:
            with torch.no_grad():
                logits = self.forward(model, batch, adapter)
        finally:
            model.train(was)
        if logits.dim() != 2:
            raise TrainingError(f"Model output has shape {tuple(logits.shape)}; text-classify "
                                "needs [B, C].", recipe=self.name, stage="loss",
                                hint="Pool the sequence (CLS / mean) before the head, e.g. "
                                     "topologies.nn.TransformerClassifier.")
        _check_classes(self.name, logits, batch["labels"], cfg.num_classes)

    def batch_to_loss(self, model, batch, cfg: TextClassifyConfig, components, adapter=None):
        self.validate_batch(batch, strict=True)
        logits = self.forward(model, batch, adapter)
        labels = batch["labels"].long()
        loss = F.cross_entropy(logits, labels, label_smoothing=cfg.label_smoothing) \
            if cfg.loss is None else compute_loss(cfg, logits, labels)
        acc = (logits.argmax(-1) == labels).float().mean()
        return loss, {"loss": loss.detach(), "acc": acc.detach()}  # tensors: no per-step sync

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        logits = self.forward(model, batch, adapter)
        labels = batch["labels"].long()
        loss = F.cross_entropy(logits, labels, label_smoothing=cfg.label_smoothing) if cfg.loss is None else compute_loss(cfg, logits, labels)
        metrics = {"loss": float(loss), "acc": float((logits.argmax(-1) == labels).float().mean())}
        return EvaluationStepResult(metrics, EvaluationRecord(logits, labels, sample_ids=batch.get("sample_ids"), metadata=batch))

    def predict_step(self, model, batch, cfg, components, adapter=None):
        return self.forward(model, batch, adapter)


class TokenClassifyConfig(RecipeConfig):
    num_classes: int | None = None
    vocab_size: int | None = None
    ignore_index: int = -100
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)


@register_recipe("token-classify", help="Per-token classification (NER / tagging).")
class TokenClassifyRecipe(Recipe):
    level = "loss"
    ConfigSchema = TokenClassifyConfig
    ComponentsSchema = None

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 5e-4, "weight_decay": 0.01, "scheduler": "cosine"}

    def expected_batch_spec(self) -> dict[str, str]:
        return {"tokens": "LongTensor[B, T]", "labels": "LongTensor[B, T] (-100 = ignore)",
                "mask": "LongTensor[B, T] (optional)"}

    def optional_batch_keys(self) -> set[str]:
        return {"mask"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {"tokens": torch.randint(0, 64, (4, 8), device=dev),
                "labels": torch.randint(0, 3, (4, 8), device=dev)}

    def _targets(self, batch: dict, cfg: TokenClassifyConfig) -> torch.Tensor:
        labels = batch["labels"].long()
        if batch.get("mask") is not None:
            labels = labels.masked_fill(batch["mask"] == 0, cfg.ignore_index)
        return labels

    def forward(self, model, batch: dict, adapter=None) -> torch.Tensor:
        inputs, keys = _inputs(batch, "tokens")
        return _logits(call_with_adapter(adapter, model, inputs, keys, primary="tokens"),
                       self.name)

    def preflight(self, model, batch, cfg: TokenClassifyConfig, components, adapter=None):
        from topologies.training.recipes.lm import _check_tokens

        _check_tokens(self.name, model, batch["tokens"], cfg.vocab_size)
        with torch.no_grad():
            logits = self.forward(model, batch, adapter)
        if logits.dim() != 3:
            raise TrainingError(f"Model output has shape {tuple(logits.shape)}; token-classify "
                                "needs [B, T, C].", recipe=self.name, stage="loss",
                                hint="Use a per-token head, e.g. topologies.nn."
                                     "TransformerTagger.")
        _check_classes(self.name, logits, batch["labels"], cfg.num_classes, cfg.ignore_index)

    def _metrics(self, logits, labels, cfg) -> tuple[torch.Tensor, dict[str, float]]:
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                               ignore_index=cfg.ignore_index,
                               label_smoothing=cfg.label_smoothing)
        valid = labels != cfg.ignore_index
        n = int(valid.sum())
        acc = float(((logits.argmax(-1) == labels) & valid).sum()) / max(1, n)
        return loss, {"loss": float(loss.detach()), "acc": acc, "_weight": float(max(n, 1))}

    def batch_to_loss(self, model, batch, cfg: TokenClassifyConfig, components, adapter=None):
        self.validate_batch(batch, strict=True)
        loss, m = self._metrics(self.forward(model, batch, adapter), self._targets(batch, cfg),
                                cfg)
        m.pop("_weight")
        return loss, m

    def validation_step(self, model, batch, cfg: TokenClassifyConfig, components, adapter=None):
        _, m = self._metrics(self.forward(model, batch, adapter), self._targets(batch, cfg), cfg)
        return m

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        logits, labels = self.forward(model, batch, adapter), self._targets(batch, cfg)
        _, metrics = self._metrics(logits, labels, cfg)
        return EvaluationStepResult(metrics, EvaluationRecord(logits, labels, mask=labels != cfg.ignore_index, sample_ids=batch.get("sample_ids"), metadata=batch))

    def predict_step(self, model, batch, cfg, components, adapter=None):
        return self.forward(model, batch, adapter)


class MultilabelConfig(RecipeConfig):
    num_classes: int | None = None
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    pos_weight: float | None = Field(default=None, gt=0.0)


@register_recipe("multilabel-classify", help="Multi-label classification (binary cross-entropy).")
class MultilabelRecipe(Recipe):
    level = "loss"
    ConfigSchema = MultilabelConfig
    ComponentsSchema = None

    def expected_batch_spec(self) -> dict[str, str]:
        return {"tokens": "LongTensor[B, T] (token models) - or -",
                "features": "FloatTensor[B, F] (feature models)",
                "labels": "FloatTensor[B, C] (0/1 per class)",
                "mask": "LongTensor[B, T] (optional)"}

    def optional_batch_keys(self) -> set[str]:
        return {"tokens", "features", "mask"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {"tokens": torch.randint(0, 64, (4, 8), device=dev),
                "labels": torch.randint(0, 2, (4, 3), device=dev).float()}

    def forward(self, model, batch: dict, adapter=None) -> torch.Tensor:
        key = _primary(batch, ("tokens", "features"), self.name)
        inputs, keys = _inputs(batch, key)
        return _logits(call_with_adapter(adapter, model, inputs, keys, primary=key), self.name)

    def _metrics(self, logits, labels, cfg) -> tuple[torch.Tensor, dict[str, float]]:
        pw = torch.full((logits.shape[-1],), cfg.pos_weight, device=logits.device) \
            if cfg.pos_weight else None
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)
        pred = torch.sigmoid(logits) >= cfg.threshold
        gold = labels >= 0.5
        exact = (pred == gold).all(-1).float().mean()
        tp = float((pred & gold).sum())
        fp, fn = float((pred & ~gold).sum()), float((~pred & gold).sum())
        f1 = 2 * tp / max(1.0, 2 * tp + fp + fn)
        return loss, {"loss": float(loss.detach()), "acc": float(exact), "f1": f1}

    def batch_to_loss(self, model, batch, cfg: MultilabelConfig, components, adapter=None):
        if "labels" not in batch:
            self.validate_batch(batch, strict=True)
        logits = self.forward(model, batch, adapter)
        if logits.shape != batch["labels"].shape:
            raise TrainingError(f"Logits {tuple(logits.shape)} do not match labels "
                                f"{tuple(batch['labels'].shape)}.", recipe=self.name,
                                stage="loss", hint="Build the head with num_classes outputs.")
        return self._metrics(logits, batch["labels"].float(), cfg)

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        logits = self.forward(model, batch, adapter)
        labels = batch["labels"].float()
        _, metrics = self._metrics(logits, labels, cfg)
        return EvaluationStepResult(metrics, EvaluationRecord(logits, labels, sample_ids=batch.get("sample_ids"), metadata=batch))

    def predict_step(self, model, batch, cfg, components, adapter=None):
        return self.forward(model, batch, adapter)


class TabularConfig(LossOverrideConfig):
    num_classes: int | None = None
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)


@register_recipe("tabular-classify", help="Classification of feature vectors (cross-entropy).")
class TabularClassifyRecipe(Recipe):
    level = "loss"
    ConfigSchema = TabularConfig
    ComponentsSchema = None

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "scheduler": "cosine"}

    def expected_batch_spec(self) -> dict[str, str]:
        return {"features": "FloatTensor[B, F]", "labels": "LongTensor[B]",
                "categorical": "LongTensor[B, K] (optional)"}

    def optional_batch_keys(self) -> set[str]:
        return {"categorical"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {"features": torch.randn(4, 6, device=dev),
                "labels": torch.randint(0, 2, (4,), device=dev)}

    def forward(self, model, batch: dict, adapter=None) -> torch.Tensor:
        inputs, keys = _inputs(batch, "features", ("categorical",))
        return _logits(call_with_adapter(adapter, model, inputs, keys, primary="features"),
                       self.name)

    def preflight(self, model, batch, cfg: TabularConfig, components, adapter=None):
        with torch.no_grad():
            logits = self.forward(model, batch, adapter)
        _check_classes(self.name, logits, batch["labels"], cfg.num_classes)

    def batch_to_loss(self, model, batch, cfg: TabularConfig, components, adapter=None):
        self.validate_batch(batch, strict=True)
        logits = self.forward(model, batch, adapter)
        labels = batch["labels"].long()
        loss = F.cross_entropy(logits, labels, label_smoothing=cfg.label_smoothing) \
            if cfg.loss is None else compute_loss(cfg, logits, labels)
        acc = (logits.argmax(-1) == labels).float().mean()
        return loss, {"loss": loss.detach(), "acc": acc.detach()}  # tensors: no per-step sync

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        logits = self.forward(model, batch, adapter)
        labels = batch["labels"].long()
        loss = F.cross_entropy(logits, labels, label_smoothing=cfg.label_smoothing) if cfg.loss is None else compute_loss(cfg, logits, labels)
        return EvaluationStepResult({"loss": float(loss), "acc": float((logits.argmax(-1) == labels).float().mean())}, EvaluationRecord(logits, labels, sample_ids=batch.get("sample_ids"), metadata=batch))

    def predict_step(self, model, batch, cfg, components, adapter=None):
        return self.forward(model, batch, adapter)


class RegressionConfig(LossOverrideConfig):
    loss_type: str = Field(default="mse", pattern="^(mse|huber|l1)$")
    huber_delta: float = Field(default=1.0, gt=0.0)


@register_recipe("regression", help="Regression (MSE / Huber / L1) on any input tensor.")
class RegressionRecipe(Recipe):
    level = "loss"
    ConfigSchema = RegressionConfig
    ComponentsSchema = None

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "scheduler": "cosine"}

    def expected_batch_spec(self) -> dict[str, str]:
        return {"inputs": "FloatTensor[B, ...]", "targets": "FloatTensor[B] or [B, K]",
                "mask": "FloatTensor like targets (optional; 1 = observed)"}

    def optional_batch_keys(self) -> set[str]:
        return {"mask"}

    def target_keys(self) -> set[str]:
        return {"targets", "target", "labels", "label"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {"inputs": torch.randn(4, 6, device=dev), "targets": torch.randn(4, device=dev)}

    def forward(self, model, batch: dict, adapter=None) -> torch.Tensor:
        return _logits(call_with_adapter(adapter, model, {"inputs": batch["inputs"]},
                                         ["inputs"], primary="inputs"), self.name)

    def _metrics(self, pred, target, mask, cfg) -> tuple[torch.Tensor, dict[str, float]]:
        if cfg.loss == "pinball":
            loss = compute_loss(cfg, pred, target)
            median = pred[..., pred.shape[-1] // 2]
            mae = float((median - target).abs().mean().detach())
            return loss, {"loss": float(loss.detach()), "mae": mae}
        if pred.shape != target.shape:
            if pred.numel() == target.numel():
                pred = pred.reshape(target.shape)
            else:
                raise TrainingError(f"Predictions {tuple(pred.shape)} do not match targets "
                                    f"{tuple(target.shape)}.", recipe=self.name, stage="loss",
                                    hint="Set the model's out_features to the target size.")
        w = mask.float() if mask is not None else torch.ones_like(target)
        n = w.sum().clamp(min=1)
        if cfg.loss_type == "mse":
            per = (pred - target) ** 2
        elif cfg.loss_type == "l1":
            per = (pred - target).abs()
        else:
            per = F.huber_loss(pred, target, reduction="none", delta=cfg.huber_delta)
        loss = ((per * w).sum() / n if cfg.loss is None
                else compute_loss(cfg, pred, target))
        mae = float(((pred - target).abs() * w).sum() / n)
        return loss, {"loss": float(loss.detach()), "mae": mae}

    def batch_to_loss(self, model, batch, cfg: RegressionConfig, components, adapter=None):
        self.validate_batch(batch, strict=True)
        pred = self.forward(model, batch, adapter)
        return self._metrics(pred, batch["targets"].float(), batch.get("mask"), cfg)

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        pred = self.forward(model, batch, adapter)
        target = batch["targets"].float()
        _loss, metrics = self._metrics(pred, target, batch.get("mask"), cfg)
        return EvaluationStepResult(metrics, EvaluationRecord(pred, target, mask=batch.get("mask"), weights=batch.get("weights"), sample_ids=batch.get("sample_ids"), metadata=batch))

    def predict_step(self, model, batch, cfg, components, adapter=None):
        return self.forward(model, batch, adapter)


__all__ = ["MultilabelRecipe", "RegressionRecipe", "TabularClassifyRecipe",
           "TextClassifyRecipe", "TokenClassifyRecipe"]
