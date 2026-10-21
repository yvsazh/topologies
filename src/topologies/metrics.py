"""Evaluation metrics on ``(logits_or_predictions, labels)`` tensors.

All functions accept either logits ``[N, C]`` (argmax is taken) or
predicted class ids ``[N]``; token tensors ``[B, T, C]`` / ``[B, T]`` are
flattened. ``ignore_index`` entries (default ``-100``) are skipped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


def _preds_and_labels(outputs: torch.Tensor, labels: torch.Tensor,
                      ignore_index: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    outputs, labels = outputs.detach().cpu(), labels.detach().cpu().long()
    preds = outputs.argmax(-1) if outputs.is_floating_point() and outputs.dim() == \
        labels.dim() + 1 else outputs.long()
    preds, labels = preds.reshape(-1), labels.reshape(-1)
    if preds.shape != labels.shape:
        raise ValueError(f"predictions {tuple(preds.shape)} and labels "
                         f"{tuple(labels.shape)} do not match.")
    if ignore_index is not None:
        keep = labels != ignore_index
        preds, labels = preds[keep], labels[keep]
    return preds, labels


def accuracy(outputs: torch.Tensor, labels: torch.Tensor, *,
             ignore_index: int | None = -100) -> float:
    """Fraction of correct predictions."""
    preds, labels = _preds_and_labels(outputs, labels, ignore_index)
    return float((preds == labels).float().mean()) if labels.numel() else 0.0


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5, *,
                   ignore_index: int | None = -100) -> float:
    """Fraction of samples whose label is among the ``k`` highest logits."""
    logits, labels = logits.detach().cpu(), labels.detach().cpu().long()
    logits = logits.reshape(-1, logits.shape[-1])
    labels = labels.reshape(-1)
    if ignore_index is not None:
        keep = labels != ignore_index
        logits, labels = logits[keep], labels[keep]
    if not labels.numel():
        return 0.0
    top = logits.topk(min(k, logits.shape[-1]), dim=-1).indices
    return float((top == labels[:, None]).any(dim=-1).float().mean())


def perplexity(loss_or_logits: float | torch.Tensor, labels: torch.Tensor | None = None, *,
               ignore_index: int = -100, shift: bool = False) -> float:
    """``exp(mean cross-entropy)`` from a loss value, or from logits + labels.

    ``shift=True`` applies the causal next-token shift (logits[:, :-1] vs labels[:, 1:]).
    """
    if labels is None:
        loss = float(loss_or_logits)
    else:
        logits = loss_or_logits.detach().float()
        labels = labels.detach().long()
        if shift:
            logits, labels = logits[:, :-1], labels[:, 1:]
        loss = float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                     labels.reshape(-1), ignore_index=ignore_index))
    return math.exp(min(loss, 50.0))


def bits_per_byte(loss_nats: float, bytes_per_token: float) -> float:
    """Tokenizer-independent LM loss: ``loss_nats / (ln 2 * bytes_per_token)``.

    Per-token cross-entropy is not comparable across tokenizers (they split
    the same text into different numbers of tokens); dividing by the bytes
    each token covers makes it so. ``bytes_per_token`` comes from
    ``topologies.tokenizers.tokenizer_stats`` (UTF-8 bytes of the evaluated
    documents, +1 per document boundary, over their token count). Pass
    characters per token instead to get bits per character.

    >>> round(bits_per_byte(math.log(2) * 4.0, 2.0), 6)  # 4 bits/token, 2 bytes/token
    2.0
    """
    bpt = float(bytes_per_token)
    if not bpt > 0:
        raise ValueError(f"bytes_per_token must be > 0; got {bytes_per_token!r}.")
    return float(loss_nats) / (math.log(2.0) * bpt)


def confusion_matrix(outputs: torch.Tensor, labels: torch.Tensor,
                     num_classes: int | None = None, *,
                     ignore_index: int | None = -100) -> torch.Tensor:
    """``[C, C]`` counts; rows = true class, columns = predicted class."""
    preds, labels = _preds_and_labels(outputs, labels, ignore_index)
    if num_classes is None:
        num_classes = int(max(int(preds.max()) if preds.numel() else 0,
                              int(labels.max()) if labels.numel() else 0)) + 1
        if outputs.is_floating_point():
            num_classes = max(num_classes, int(outputs.shape[-1]))
    idx = labels * num_classes + preds
    return torch.bincount(idx, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes)


@dataclass
class ClassificationReport:
    """Per-class precision / recall / F1 / support plus macro/weighted averages."""

    classes: list[Any]
    precision: list[float]
    recall: list[float]
    f1: list[float]
    support: list[int]
    accuracy: float
    matrix: torch.Tensor

    def as_dict(self) -> dict[str, Any]:
        rows = {
            str(c): {"precision": p, "recall": r, "f1": f, "support": s}
            for c, p, r, f, s in zip(self.classes, self.precision, self.recall, self.f1,
                                     self.support)
        }
        total = max(1, sum(self.support))
        rows["macro avg"] = {
            "precision": sum(self.precision) / max(1, len(self.classes)),
            "recall": sum(self.recall) / max(1, len(self.classes)),
            "f1": sum(self.f1) / max(1, len(self.classes)),
            "support": sum(self.support),
        }
        rows["weighted avg"] = {
            "precision": sum(p * s for p, s in zip(self.precision, self.support)) / total,
            "recall": sum(r * s for r, s in zip(self.recall, self.support)) / total,
            "f1": sum(f * s for f, s in zip(self.f1, self.support)) / total,
            "support": sum(self.support),
        }
        rows["accuracy"] = self.accuracy
        return rows

    def __str__(self) -> str:
        width = max([len(str(c)) for c in self.classes] + [12])
        lines = [f"{'':>{width}}  precision  recall  f1-score  support"]
        for c, p, r, f, s in zip(self.classes, self.precision, self.recall, self.f1,
                                 self.support):
            lines.append(f"{c!s:>{width}}  {p:9.3f}  {r:6.3f}  {f:8.3f}  {s:7d}")
        d = self.as_dict()
        lines.append("")
        lines.append(f"{'accuracy':>{width}}  {'':9}  {'':6}  {self.accuracy:8.3f}  "
                     f"{sum(self.support):7d}")
        for name in ("macro avg", "weighted avg"):
            row = d[name]
            lines.append(f"{name:>{width}}  {row['precision']:9.3f}  {row['recall']:6.3f}  "
                         f"{row['f1']:8.3f}  {row['support']:7d}")
        return "\n".join(lines)


def per_class_report(outputs: torch.Tensor, labels: torch.Tensor, *,
                     class_names: Sequence[Any] | None = None,
                     num_classes: int | None = None,
                     ignore_index: int | None = -100) -> ClassificationReport:
    """Precision/recall/F1 per class; ``print(report)`` gives a table."""
    if num_classes is None and class_names is not None:
        num_classes = len(class_names)
    cm = confusion_matrix(outputs, labels, num_classes, ignore_index=ignore_index)
    tp = cm.diag().float()
    pred_n = cm.sum(dim=0).float()
    true_n = cm.sum(dim=1).float()
    precision = torch.where(pred_n > 0, tp / pred_n.clamp(min=1), torch.zeros_like(tp))
    recall = torch.where(true_n > 0, tp / true_n.clamp(min=1), torch.zeros_like(tp))
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom.clamp(min=1e-12),
                     torch.zeros_like(tp))
    n = cm.shape[0]
    names = list(class_names) if class_names is not None else list(range(n))
    return ClassificationReport(
        classes=names, precision=precision.tolist(), recall=recall.tolist(), f1=f1.tolist(),
        support=[int(x) for x in true_n.tolist()],
        accuracy=float(tp.sum() / max(1.0, float(cm.sum()))), matrix=cm,
    )


def register_metric(name: str, factory: Any, *, metadata: dict[str, Any] | None = None) -> None:
    """Register a stateful evaluator factory without altering legacy helpers."""
    from topologies.evaluation import register_metric as _register
    _register(name, factory, metadata=metadata)


def stateful_metric(name: str, **kwargs: Any) -> Any:
    """Create a fresh stateful metric by registered name."""
    from topologies.evaluation import metric as _metric
    return _metric(name, **kwargs)


__all__ = [
    "ClassificationReport",
    "accuracy",
    "bits_per_byte",
    "confusion_matrix",
    "per_class_report",
    "perplexity",
    "register_metric",
    "stateful_metric",
    "top_k_accuracy",
]
