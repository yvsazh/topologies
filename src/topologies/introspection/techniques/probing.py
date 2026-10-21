"""Probes (U) with control tasks (Hewitt & Liang 2019) and concept activation vectors (TCAV).

Probes report **selectivity**, which is task accuracy minus control accuracy, not
only raw accuracy. A probe is correlational: it shows the information is linearly
(or MLP-) decodable at a site, not that the model uses it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from topologies.introspection.objectives import as_objective, target_objective
from topologies.introspection.payloads import Payload, input_fingerprint, make_result
from topologies.introspection.techniques import Technique, builtin
from topologies.introspection.techniques._features import batches, collect_features, pooled
from topologies.introspection.views import bars, finish
from topologies.visualization.views import ViewSpec


@dataclass
class ProbePayload(Payload):
    site: str
    kind: str
    task: str
    train_score: float
    test_score: float
    control_score: float | None
    selectivity: float | None
    direction: torch.Tensor | None
    classes: list[Any] = field(default_factory=list)
    examples: int = 0
    metric: str = "accuracy"

    def to_records(self) -> list[dict[str, Any]]:
        return [{"site": self.site, "kind": self.kind, "task": self.task, "metric": self.metric,
                 "train": self.train_score, "test": self.test_score, "control": self.control_score,
                 "selectivity": self.selectivity, "examples": self.examples}]

    def views(self, **options: Any) -> list[ViewSpec]:
        labels, values = ["train", "test"], [self.train_score, self.test_score]
        if self.control_score is not None:
            labels.append("control")
            values.append(self.control_score)
        return finish([bars(f"{self.kind} probe at {self.site} ({self.metric})", labels, values,
                            caption=f"selectivity = test - control = {self.selectivity}")], **options)


def _fit(x: torch.Tensor, y: torch.Tensor, *, kind: str, task: str, classes: int, epochs: int,
         lr: float, weight_decay: float, seed: int, hidden: int) -> nn.Module:
    torch.manual_seed(seed)
    out_dim = classes if task == "classification" else 1
    model = nn.Linear(x.shape[1], out_dim) if kind == "linear" else nn.Sequential(
        nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    target = y.long() if task == "classification" else y.float().reshape(-1, 1)
    with torch.enable_grad():
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(model(x), target)
            loss.backward()
            optimizer.step()
    return model.eval()


def _score(model: nn.Module, x: torch.Tensor, y: torch.Tensor, task: str) -> float:
    with torch.no_grad():
        out = model(x)
    if task == "classification":
        return float((out.argmax(-1) == y.long()).float().mean())
    residual = ((out.reshape(-1) - y.float()) ** 2).sum()
    total = ((y.float() - y.float().mean()) ** 2).sum().clamp_min(1e-12)
    return float(1 - residual / total)


def train_probe(x: torch.Tensor, y: torch.Tensor, *, kind: str = "linear",
                task: str = "classification", control: str | None = "random_labels",
                split: float = 0.2, seed: int = 0, epochs: int = 300, lr: float = 0.05,
                weight_decay: float = 1e-3, hidden: int = 64) -> dict[str, Any]:
    x = x.float()
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    x = (x - mean) / std
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(x.shape[0], generator=g)
    n_test = max(1, round(split * x.shape[0]))
    test, train = order[:n_test], order[n_test:]
    classes: list[Any] = []
    yy = y
    if task == "classification":
        classes, yy = torch.unique(y, return_inverse=True)
        classes = classes.tolist()
    kw = {"kind": kind, "task": task, "classes": max(len(classes), 1), "epochs": epochs, "lr": lr,
          "weight_decay": weight_decay, "seed": seed, "hidden": hidden}
    model = _fit(x[train], yy[train], **kw)
    result = {"train": _score(model, x[train], yy[train], task),
              "test": _score(model, x[test], yy[test], task), "classes": classes,
              "model": model, "control": None, "selectivity": None}
    if control is not None:
        if control == "random_labels" and task == "classification":
            counts = torch.bincount(yy, minlength=len(classes)).double()
            fake = torch.multinomial(counts / counts.sum(), yy.numel(), replacement=True,
                                     generator=g)
        elif control in ("shuffled", "random_labels"):
            fake = yy[torch.randperm(yy.numel(), generator=g)]
        else:
            raise ValueError("control must be 'random_labels', 'shuffled' or None")
        cmodel = _fit(x[train], fake[train], **kw)
        result["control"] = _score(cmodel, x[test], fake[test], task)
        result["selectivity"] = result["test"] - result["control"]
    if kind == "linear":
        w = model.weight.detach() / std  # back to the unnormalised feature space
        result["direction"] = (w[1] - w[0] if w.shape[0] == 2 else w if w.shape[0] > 2
                               else w[0])
        if result["direction"].ndim == 1:
            result["direction"] = result["direction"] / result["direction"].norm().clamp_min(1e-12)
    else:
        result["direction"] = None
    return result


@builtin
class Probe(Technique):
    name = "probing.probe"
    tier = "U"
    result_kind = "introspection.probe"
    summary = "linear/MLP probes with control tasks (selectivity) and probe directions"

    def run(self, subject: Any, inputs: Any = None, *, labels: Any = None, site: str = "",
            kind: str = "linear", task: str = "classification", control: str | None = "random_labels",
            split: float = 0.2, seed: int = 0, pool: str = "mean", max_batches: int | None = None,
            epochs: int = 300, lr: float = 0.05, weight_decay: float = 1e-3, **params: Any) -> Any:
        path = subject.index.resolve_modules(site)[0]
        feats, ys, notes = collect_features(subject, inputs, [path], pool=pool, labels=labels,
                                            max_batches=max_batches)
        if ys is None:
            raise ValueError("probe needs labels= (tensor, field name, callable) or "
                             "(input, target) data")
        result = train_probe(feats[path], ys, kind=kind, task=task, control=control, split=split,
                             seed=seed, epochs=epochs, lr=lr, weight_decay=weight_decay)
        payload = ProbePayload(path or "<root>", kind, task, result["train"], result["test"],
                               result["control"], result["selectivity"], result["direction"],
                               result["classes"], int(ys.numel()),
                               "accuracy" if task == "classification" else "R^2")
        return make_result(self, payload, subject, assumptions=[
            "probes are correlational: decodable information is not necessarily used by the model",
            (f"{kind} probe, AdamW ({epochs} full-batch steps, weight decay {weight_decay}), "
            f"held-out split {split}, seed {seed}; features standardised"),
            f"control task: {control}", *notes], provenance={"inputs": input_fingerprint(inputs)})


@dataclass
class TCAVPayload(Payload):
    site: str
    score: float
    cav_accuracy: float
    magnitudes: torch.Tensor
    random_scores: list[float] = field(default_factory=list)

    def to_records(self) -> list[dict[str, Any]]:
        return [{"site": self.site, "tcav_score": self.score, "cav_accuracy": self.cav_accuracy,
                 "random_mean": (sum(self.random_scores) / len(self.random_scores)
                                 if self.random_scores else None)}]

    def views(self, **options: Any) -> list[ViewSpec]:
        labels = ["concept", *[f"random {i}" for i in range(len(self.random_scores))]]
        return finish([bars(f"TCAV score at {self.site}", labels, [self.score, *self.random_scores],
                            caption="fraction of inputs whose objective increases along the CAV")],
                      **options)


@builtin
class TCAV(Technique):
    name = "probing.tcav"
    tier = "D"
    result_kind = "introspection.tcav"
    summary = "concept activation vector + directional derivative of an objective"

    def run(self, subject: Any, inputs: Any = None, *, concept: Any = None, random: Any = None,
            site: str = "", target: Any = "predicted", pool: str = "mean", seed: int = 0,
            runs: int = 3, max_batches: int | None = None, **params: Any) -> Any:
        if concept is None or random is None:
            raise ValueError("tcav needs concept=<data> and random=<data> examples")
        path = subject.index.resolve_modules(site)[0]
        cf, _, notes = collect_features(subject, concept, [path], pool=pool, max_batches=max_batches)
        rf, _, _ = collect_features(subject, random, [path], pool=pool, max_batches=max_batches)
        objective = target_objective(target) if not hasattr(target, "compute") else as_objective(target)
        grads = _site_gradients(subject, inputs, path, objective, pool)

        def cav(pos: torch.Tensor, neg: torch.Tensor, s: int) -> tuple[torch.Tensor, float]:
            x = torch.cat([pos, neg])
            y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).long()
            fit = train_probe(x, y, control=None, seed=s, split=0.2)
            return fit["direction"], fit["test"]
        direction, accuracy = cav(cf[path], rf[path], seed)
        magnitudes = grads @ direction.to(grads)
        score = float((magnitudes > 0).float().mean())
        random_scores = []
        half = rf[path].shape[0] // 2
        for r in range(runs if half >= 2 else 0):
            g = torch.Generator().manual_seed(seed + 1 + r)
            perm = torch.randperm(rf[path].shape[0], generator=g)
            d, _ = cav(rf[path][perm[:half]], rf[path][perm[half:]], seed + 1 + r)
            random_scores.append(float(((grads @ d.to(grads)) > 0).float().mean()))
        payload = TCAVPayload(path or "<root>", score, accuracy, magnitudes, random_scores)
        return make_result(self, payload, subject, assumptions=[
            ("TCAV (Kim et al. 2018): CAV = linear probe direction concept vs random; score = "
            "fraction of inputs with positive directional derivative of the objective"),
            "compare with the random-CAV scores before reading meaning into the score", *notes],
            provenance={"inputs": input_fingerprint(inputs)})


def _site_gradients(subject: Any, data: Any, path: str, objective: Any, pool: str) -> torch.Tensor:
    out = []
    for batch in batches(subject, data):
        trace = subject.inspector.trace(batch, sites=f"{path or '<root>'}:grad_output",
                                        keep="tensors", objective=objective, call_tree=False,
                                        max_bytes=None)
        record = next(r for r in trace.records.values() if r.port == "grad_output")
        feats, _ = pooled(record.value, record.axes, pool, path)
        out.append(feats)
    return torch.cat(out)
