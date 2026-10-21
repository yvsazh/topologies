"""Labeling and label-noise detection.

Pipeline / relabel ops (``ds.relabel(op)`` creates a new version):

    synth.ops.label.program(lambda r: "refund" if "money" in r["text"] else r["label"])
    synth.ops.label.vote([label_a, label_b, label_c], min_agreement=2/3)

Confident learning (no cleanlab): ``ds.label_issues()`` trains a quick
cross-validated linear model on hashed n-gram features (CPU, seconds), takes
out-of-fold probabilities, sets per-class thresholds ``t_j = mean p_j`` over
records labelled ``j``, and flags records whose most likely confident class
differs from their label, ranked by self-confidence. ``issues.drop()`` /
``issues.relabel()`` create new versions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import torch

from topologies.synth.errors import SynthError
from topologies.synth.fingerprint import callable_fingerprint
from topologies.synth.ops.base import MapOp, Reject


class ProgramLabel(MapOp):
    evidence = "construction"

    def __init__(self, fn: Callable[[dict], Any], *, field: str = "label",
                 name: str = "label.program") -> None:
        self.fn, self.field, self.name = fn, field, name

    def apply(self, record: dict, ctx: Any) -> dict:
        return {**record, self.field: self.fn(record)}

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "fn": callable_fingerprint(self.fn), "field": self.field}


class Vote(MapOp):
    """Majority label of several labelers; rejects when agreement < ``min_agreement``."""

    evidence = "consistency"

    def __init__(self, labelers: Sequence[Any], *, field: str = "label",
                 min_agreement: float = 0.5, name: str = "label.vote") -> None:
        if len(labelers) < 2:
            raise SynthError("vote() needs at least two labelers.", stage="label")
        self.labelers, self.field, self.min_agreement, self.name = list(labelers), field, \
            float(min_agreement), name

    def setup(self, runtime: Any) -> None:
        for lab in self.labelers:
            if hasattr(lab, "setup"):
                lab.setup(runtime)

    def apply(self, record: dict, ctx: Any) -> dict:
        votes = []
        for k, lab in enumerate(self.labelers):
            sub = ctx.child(f"vote{k}")
            if hasattr(lab, "apply"):
                out = lab.apply(dict(record), sub)
                votes.append(out.get(self.field))
            else:
                out = lab(dict(record))
                votes.append(out)
        top, n = Counter(map(str, votes)).most_common(1)[0]
        share = n / len(votes)
        if share < self.min_agreement:
            raise Reject(f"labelers disagree: {votes}", score=share)
        ctx.log(votes=votes)
        return {**record, self.field: next(v for v in votes if str(v) == top)}

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "labelers": [lab.config() if hasattr(lab, "config") else
                                              callable_fingerprint(lab) for lab in
                                              self.labelers],
                "field": self.field, "min_agreement": self.min_agreement}


def program(fn: Callable[[dict], Any], **kw: Any) -> ProgramLabel:
    return ProgramLabel(fn, **kw)


def vote(labelers: Sequence[Any], **kw: Any) -> Vote:
    return Vote(labelers, **kw)


# ---------------------------------------------------------------- confident learning


def featurize(records: Sequence[dict], *, key: Any = None, dim: int = 2048) -> torch.Tensor:
    """Hashed word uni/bi-gram + char trigram features (L2-normalized)."""
    from topologies.synth.quality.embed import HashedEmbedder
    from topologies.synth.textutil import text_of

    return HashedEmbedder(dim).encode([text_of(r, key) for r in records])


def train_linear(x: torch.Tensor, y: torch.Tensor, n_classes: int, *, epochs: int = 150,
                 lr: float = 0.5, weight_decay: float = 1e-4, seed: int = 0) -> torch.nn.Module:
    g = torch.Generator().manual_seed(seed)
    model = torch.nn.Linear(x.shape[1], n_classes)
    torch.nn.init.normal_(model.weight, std=0.01, generator=g)
    torch.nn.init.zeros_(model.bias)
    opt = torch.optim.Adam(model.parameters(), lr=lr * 0.1, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    return model


def out_of_fold_probs(x: torch.Tensor, y: torch.Tensor, n_classes: int, *, folds: int = 5,
                      seed: int = 0, **kw: Any) -> torch.Tensor:
    n = x.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    probs = torch.zeros(n, n_classes)
    for f in range(folds):
        test = perm[f::folds]
        mask = torch.ones(n, dtype=torch.bool)
        mask[test] = False
        model = train_linear(x[mask], y[mask], n_classes, seed=seed + f, **kw)
        with torch.no_grad():
            probs[test] = torch.softmax(model(x[test]), -1)
    return probs


def confident_issues(probs: torch.Tensor, y: torch.Tensor) -> list[tuple[int, int, float]]:
    """``(index, suggested class, self-confidence)`` of likely mislabels, most suspect first."""
    n, c = probs.shape
    thr = torch.zeros(c)
    for j in range(c):
        sel = y == j
        thr[j] = probs[sel, j].mean() if sel.any() else 1.0
    out = []
    for i in range(n):
        cand = (probs[i] >= thr).nonzero().flatten().tolist()
        if not cand:
            continue
        best = max(cand, key=lambda j: float(probs[i, j]))
        if best != int(y[i]):
            out.append((i, best, float(probs[i, y[i]])))
    return sorted(out, key=lambda t: t[2])


class LabelIssues:
    """Ranked likely mislabels of a dataset version (see module docstring)."""

    def __init__(self, ds: Any, rows: list[dict[str, Any]], field: str, n: int) -> None:
        self.ds, self.rows, self.field, self.n = ds, rows, field, n

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    @property
    def ids(self) -> list[str]:
        return [r["_id"] for r in self.rows]

    def table(self, n: int = 20) -> Any:
        from topologies.experiments.table import Table
        from topologies.synth.report import _short

        rows = [[r["_id"], r["label"], r["suggested"], r["self_confidence"],
                 _short(r["text"], 70)] for r in self.rows[:n]]
        return Table(["id", "label", "suggested", "self-confidence", "text"], rows,
                     meta={"title": f"label issues: {len(self.rows)} of {self.n} records "
                                    f"({len(self.rows) / max(1, self.n):.1%})"})

    def drop(self) -> Any:
        return self.ds.drop(self.ids, reason="label issue")

    def relabel(self) -> Any:
        fix = {r["_id"]: r["suggested"] for r in self.rows}
        f = self.field

        def apply(rec: dict) -> dict:
            return {**rec, f: fix[rec["_id"]]} if rec["_id"] in fix else rec

        return self.ds.edit(apply, note=f"relabel {len(fix)} label issues")

    def __repr__(self) -> str:
        return str(self.table())


def label_issues(ds: Any, *, field: str | None = None, key: Any = None, folds: int = 5,
                 split: str | None = None, seed: int = 0, dim: int = 2048,
                 epochs: int = 150) -> LabelIssues:
    """Confident-learning label issues of a dataset (or a list of records)."""
    if hasattr(ds, "view"):
        records = list(ds.view(split))
        schema = ds.schema
    else:
        records, schema = list(ds), None
    f = field or (schema.label_field() if schema is not None else None) or "label"
    if not records or f not in records[0]:
        raise SynthError(f"No {f!r} labels to check.", stage="label")
    names = sorted({str(r[f]) for r in records})
    if len(names) < 2:
        raise SynthError("label_issues needs at least two classes.", stage="label")
    y = torch.tensor([names.index(str(r[f])) for r in records])
    text_key = key
    if text_key is None:
        text_key = [k for k in records[0] if not str(k).startswith("_") and k != f and
                    isinstance(records[0][k], str)] or None
    x = featurize(records, key=text_key, dim=dim)
    probs = out_of_fold_probs(x, y, len(names), folds=folds, seed=seed, epochs=epochs)
    from topologies.synth.textutil import text_of

    rows = [{"_id": records[i].get("_id", str(i)), "label": names[int(y[i])],
             "suggested": names[j], "self_confidence": round(sc, 4),
             "text": text_of(records[i], text_key)} for i, j, sc in confident_issues(probs, y)]
    return LabelIssues(ds, rows, f, len(records))


__all__ = ["LabelIssues", "ProgramLabel", "Vote", "confident_issues", "featurize",
           "label_issues", "out_of_fold_probs", "program", "train_linear", "vote"]


from topologies.synth.ops._callable import make_callable

make_callable(__name__, "program")
