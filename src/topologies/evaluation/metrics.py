"""Built-in mergeable metrics for regression and single-label classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .api import EvaluationRecord, MetricValue


def _observations(record: EvaluationRecord):
    pred = torch.as_tensor(record.predictions).detach().cpu()
    target = torch.as_tensor(record.targets).detach().cpu()
    if pred.shape[: target.ndim] != target.shape:
        if pred.numel() == target.numel():
            pred = pred.reshape(target.shape)
        elif pred.ndim == target.ndim + 1 and pred.shape[:-1] == target.shape:
            pass
        else:
            raise ValueError(
                f"prediction/target shapes are incompatible: {pred.shape}, {target.shape}"
            )
    mask = (
        torch.ones_like(target, dtype=torch.bool)
        if record.mask is None
        else torch.as_tensor(record.mask).detach().cpu().bool()
    )
    try:
        if mask.ndim == 1 and target.ndim > 1 and mask.shape[0] == target.shape[0]:
            mask = mask.reshape((mask.shape[0],) + (1,) * (target.ndim - 1))
        mask = torch.broadcast_to(mask, target.shape)
    except RuntimeError as exc:
        raise ValueError("metric mask is not broadcastable to target shape") from exc
    if record.weights is None:
        weight = torch.ones_like(target, dtype=torch.float64)
    else:
        weight = torch.as_tensor(record.weights).detach().cpu().double()
        if weight.ndim == 1 and target.ndim > 1 and weight.shape[0] == target.shape[0]:
            weight = weight.reshape((weight.shape[0],) + (1,) * (target.ndim - 1))
        weight = torch.broadcast_to(weight, target.shape)
    mask &= weight > 0
    return pred, target, mask, weight


def _sum_reduce(tensor: torch.Tensor) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)


@dataclass
class RegressionMetric:
    """Weighted, masked exact MAE/MSE/RMSE sufficient statistics."""

    kind: str = "mae"

    def __post_init__(self):
        if self.kind not in {"mae", "mse", "rmse"}:
            raise ValueError("kind must be mae, mse, or rmse")
        self.reset()

    def reset(self):
        self.error = 0.0
        self.weight = 0.0
        self.support = 0

    def update(self, record: EvaluationRecord):
        pred, target, keep, weight = _observations(record)
        pred, target = torch.broadcast_tensors(pred.double(), target.double())
        delta = pred - target
        val = delta.abs() if self.kind == "mae" else delta.square()
        self.error += float((val[keep] * weight[keep]).sum())
        self.weight += float(weight[keep].sum())
        self.support += int(keep.sum())

    def _state(self):
        return torch.tensor([self.error, self.weight, self.support], dtype=torch.float64)

    def merge_state(self):
        state = self._state()
        _sum_reduce(state)
        self.error, self.weight, self.support = float(state[0]), float(state[1]), int(state[2])

    all_reduce = merge_state

    def compute(self):
        if self.weight == 0:
            return MetricValue(None, "insufficient_support", 0, unit=self.kind)
        value = self.error / self.weight
        if self.kind == "rmse":
            value = value**0.5
        return MetricValue(value, "ok", self.support, unit=self.kind, direction="min")


class ConfusionMetric:
    """Single-label confusion-count metric; states merge by integer/weighted sum."""

    def __init__(self, num_classes: int, metric: str = "f1", average: str = "macro"):
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if metric not in {"accuracy", "precision", "recall", "f1"}:
            raise ValueError("unsupported confusion metric")
        if average not in {"macro", "micro", "weighted"}:
            raise ValueError("average must be macro, micro, or weighted")
        self.num_classes, self.metric, self.average = num_classes, metric, average
        self.reset()

    def reset(self):
        self.matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64)

    def update(self, record: EvaluationRecord):
        pred, target, keep, weight = _observations(record)
        if pred.ndim == target.ndim + 1:
            pred = pred.argmax(-1)
        pred, target = torch.broadcast_tensors(pred.long(), target.long())
        keep &= (
            (target >= 0) & (target < self.num_classes) & (pred >= 0) & (pred < self.num_classes)
        )
        indices = target[keep] * self.num_classes + pred[keep]
        self.matrix += torch.bincount(
            indices, weights=weight[keep], minlength=self.num_classes**2
        ).reshape_as(self.matrix)

    def merge_state(self):
        _sum_reduce(self.matrix)

    all_reduce = merge_state

    def compute(self):
        cm = self.matrix
        support = float(cm.sum())
        if support <= 0:
            return MetricValue(None, "insufficient_support", 0, direction="max")
        tp, true, predicted = cm.diag(), cm.sum(1), cm.sum(0)
        if self.metric == "accuracy":
            value = float(tp.sum() / support)
        else:
            p = torch.where(predicted > 0, tp / predicted.clamp_min(1e-30), 0)
            r = torch.where(true > 0, tp / true.clamp_min(1e-30), 0)
            f = torch.where(p + r > 0, 2 * p * r / (p + r).clamp_min(1e-30), 0)
            vals = {"precision": p, "recall": r, "f1": f}[self.metric]
            if self.average == "micro":
                value = float(tp.sum() / support)
            elif self.average == "weighted":
                value = float((vals * true).sum() / support)
            else:
                value = float(vals.mean())
        return MetricValue(value, "ok", support, direction="max")


class ExactAUROC:
    """Exact binary AUROC retaining at most ``max_observations`` scored labels."""

    def __init__(self, max_observations: int = 100_000, positive_label: int = 1):
        if max_observations < 1:
            raise ValueError("max_observations must be positive")
        self.max_observations, self.positive_label = max_observations, positive_label
        self.reset()

    def reset(self):
        self.scores = []
        self.labels = []

    def update(self, record: EvaluationRecord):
        p, y, mask, w = _observations(record)
        if p.ndim > y.ndim:
            if p.shape[-1] != 2:
                raise ValueError("ExactAUROC is binary; supply a score tensor or two-class outputs")
            p = p[..., 1]
        score = p.reshape(-1)[mask.reshape(-1)].double()
        label = y.reshape(-1)[mask.reshape(-1)].long() == self.positive_label
        if w[mask].numel() and not torch.all(w[mask] == 1):
            raise ValueError("ExactAUROC does not accept weights; use a weighted-compatible metric")
        if len(self.scores) + score.numel() > self.max_observations:
            raise MemoryError(f"exact AUROC exceeded max_observations={self.max_observations}")
        self.scores.extend(score.tolist())
        self.labels.extend(label.tolist())

    def compute(self):
        if not self.labels or not any(self.labels) or all(self.labels):
            return MetricValue(None, "insufficient_support", len(self.labels), direction="max")
        s = torch.tensor(self.scores, dtype=torch.float64)
        y = torch.tensor(self.labels)
        order = torch.argsort(s)
        sorted_scores = s[order]
        _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
        ranks_sorted = torch.empty(len(s), dtype=torch.float64)
        start = 0
        for n in counts.tolist():
            ranks_sorted[start : start + n] = (start + 1 + start + n) / 2
            start += n
        ranks = torch.empty_like(ranks_sorted)
        ranks[order] = ranks_sorted
        n1 = int(y.sum())
        n0 = len(y) - n1
        auc = float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        return MetricValue(auc, "ok", len(y), unit="auc", direction="max")

    def merge_state(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            lengths = torch.tensor([len(self.scores)], dtype=torch.long)
            gathered_lengths = [
                torch.zeros_like(lengths) for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(gathered_lengths, lengths)
            if sum(int(x.item()) for x in gathered_lengths) > self.max_observations:
                raise MemoryError(
                    f"distributed exact AUROC exceeded max_observations={self.max_observations}"
                )
            gathered = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered, (self.scores, self.labels))
            scores = []
            labels = []
            for s, y in gathered:
                scores.extend(s)
                labels.extend(y)
            if len(scores) > self.max_observations:
                raise MemoryError(
                    f"distributed exact AUROC exceeded max_observations={self.max_observations}"
                )
            self.scores, self.labels = scores, labels

    all_reduce = merge_state


class BinnedAUROC:
    """Bounded approximate binary AUROC using fixed [0,1] score bins."""

    def __init__(self, bins: int = 256, positive_label: int = 1):
        if bins < 2:
            raise ValueError("bins must be >= 2")
        self.bins, self.positive_label = bins, positive_label
        self.reset()

    def reset(self):
        self.positive = torch.zeros(self.bins, dtype=torch.float64)
        self.negative = self.positive.clone()

    def update(self, record):
        p, y, m, w = _observations(record)
        if p.ndim > y.ndim:
            p = p[..., 1]
        p = p.reshape(-1)[m.reshape(-1)].double().clamp(0, 1)
        y = y.reshape(-1)[m.reshape(-1)].long() == self.positive_label
        bins = (p * self.bins).long().clamp(max=self.bins - 1)
        self.positive += torch.bincount(
            bins[y], weights=w.reshape(-1)[m.reshape(-1)][y], minlength=self.bins
        )
        self.negative += torch.bincount(
            bins[~y], weights=w.reshape(-1)[m.reshape(-1)][~y], minlength=self.bins
        )

    def merge_state(self):
        _sum_reduce(self.positive)
        _sum_reduce(self.negative)

    all_reduce = merge_state

    def compute(self):
        pos, neg = float(self.positive.sum()), float(self.negative.sum())
        if not pos or not neg:
            return MetricValue(
                None, "insufficient_support", pos + neg, unit="auc-approx", direction="max"
            )
        below = torch.cumsum(self.negative, 0) - self.negative
        auc = float((self.positive * (below + 0.5 * self.negative)).sum() / (pos * neg))
        return MetricValue(
            auc,
            "ok",
            pos + neg,
            unit="auc-approx",
            direction="max",
            details={"approximation": "fixed-width score bins", "bins": self.bins},
        )


class CalibrationMetrics:
    """ECE (equal-width bins), multiclass Brier/NLL, and reliability table."""

    def __init__(self, bins: int = 15, input_kind: str = "logits"):
        if bins < 1:
            raise ValueError("bins must be positive")
        if input_kind not in {"logits", "probabilities"}:
            raise ValueError("input_kind must explicitly be logits or probabilities")
        self.bins, self.input_kind = bins, input_kind
        self.reset()

    def reset(self):
        self.count = torch.zeros(self.bins, dtype=torch.float64)
        self.conf = torch.zeros_like(self.count)
        self.correct = torch.zeros_like(self.count)
        self.brier = 0.0
        self.nll = 0.0
        self.weight = 0.0

    def update(self, record):
        p, y, m, w = _observations(record)
        if p.ndim != y.ndim + 1:
            raise ValueError("calibration metrics require multiclass [observations, classes]")
        p = p.reshape(-1, p.shape[-1]).double()
        y = y.reshape(-1).long()
        m = m.reshape(-1)
        w = w.reshape(-1)
        p = p[m]
        y = y[m]
        w = w[m]
        probs = p.softmax(-1) if self.input_kind == "logits" else p
        if ((probs < 0) | (probs > 1)).any() or not torch.allclose(
            probs.sum(-1), torch.ones(len(probs), dtype=probs.dtype), atol=1e-4
        ):
            raise ValueError("probability inputs must be normalized class probabilities")
        conf, pred = probs.max(-1)
        correct = (pred == y).double()
        idx = (conf * self.bins).long().clamp(max=self.bins - 1)
        self.count += torch.bincount(idx, weights=w, minlength=self.bins)
        self.conf += torch.bincount(idx, weights=w * conf, minlength=self.bins)
        self.correct += torch.bincount(idx, weights=w * correct, minlength=self.bins)
        oh = F.one_hot(y, num_classes=probs.shape[-1]).double()
        self.brier += float((((probs - oh) ** 2).sum(-1) * w).sum())
        self.nll += float(
            (-probs.clamp_min(1e-15).log().gather(1, y[:, None]).squeeze(1) * w).sum()
        )
        self.weight += float(w.sum())

    def merge_state(self):
        bins = torch.cat((self.count, self.conf, self.correct))
        _sum_reduce(bins)
        self.count, self.conf, self.correct = bins.chunk(3)
        sums = torch.tensor([self.brier, self.nll, self.weight], dtype=torch.float64)
        _sum_reduce(sums)
        self.brier, self.nll, self.weight = map(float, sums)

    all_reduce = merge_state

    def compute(self):
        if not self.weight:
            return MetricValue(None, "insufficient_support", 0)
        acc = self.correct / self.count.clamp_min(1)
        conf = self.conf / self.count.clamp_min(1)
        ece = float(((self.count / self.weight) * (acc - conf).abs()).sum())
        table = [
            {
                "bin": i,
                "count": float(self.count[i]),
                "confidence": float(conf[i]),
                "accuracy": float(acc[i]),
            }
            for i in range(self.bins)
        ]
        return {
            "ece": MetricValue(
                ece,
                "ok",
                self.weight,
                unit="fraction",
                direction="min",
                details={"binning": "equal-width [0,1]", "bins": self.bins, "reliability": table},
            ),
            "brier": MetricValue(
                self.brier / self.weight,
                "ok",
                self.weight,
                unit="squared-probability",
                direction="min",
            ),
            "nll": MetricValue(
                self.nll / self.weight, "ok", self.weight, unit="nats", direction="min"
            ),
        }


class TemperatureCalibrator:
    """Temperature scaling with explicit logits/probability input semantics."""

    def __init__(
        self, temperature: float = 1.0, input_kind: str = "logits", num_classes: int | None = None
    ):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if input_kind not in {"logits", "probabilities"}:
            raise ValueError("input_kind must be logits or probabilities")
        self.temperature, self.input_kind, self.num_classes = (
            float(temperature),
            input_kind,
            num_classes,
        )

    def fit(self, values: torch.Tensor, targets: torch.Tensor, *, max_iter: int = 100):
        x = torch.as_tensor(values).detach().float().cpu()
        y = torch.as_tensor(targets).detach().long().reshape(-1).cpu()
        if x.ndim != 2 or x.shape[0] != y.numel():
            raise ValueError("fit expects [N, C] and [N] single-label targets")
        if self.num_classes is not None and x.shape[1] != self.num_classes:
            raise ValueError("class count mismatch")
        if self.input_kind == "probabilities":
            if (x < 0).any() or not torch.allclose(x.sum(-1), torch.ones(len(x)), atol=1e-4):
                raise ValueError("probabilities must be normalized")
            x = x.clamp_min(1e-12).log()
        log_t = torch.zeros((), requires_grad=True)
        opt = torch.optim.LBFGS([log_t], max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(x / log_t.exp().clamp(1e-3, 1e3), y)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(log_t.detach().exp().clamp(1e-3, 1e3))
        self.num_classes = x.shape[1]
        return self

    def transform(self, values: torch.Tensor):
        x = torch.as_tensor(values)
        if x.ndim < 2 or (self.num_classes is not None and x.shape[-1] != self.num_classes):
            raise ValueError("last class axis does not match fitted calibration schema")
        if self.input_kind == "probabilities":
            if (x < 0).any() or not torch.allclose(
                x.sum(-1), torch.ones_like(x.sum(-1)), atol=1e-4
            ):
                raise ValueError("probabilities must be normalized")
            x = x.clamp_min(1e-12).log()
        return (x / self.temperature).softmax(-1)

    def state_dict(self):
        return {
            "version": 1,
            "temperature": self.temperature,
            "input_kind": self.input_kind,
            "num_classes": self.num_classes,
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != 1:
            raise ValueError("unsupported calibrator state version")
        return cls(state["temperature"], state["input_kind"], state.get("num_classes"))
