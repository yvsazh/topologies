"""Per-slice error analysis on real data, with attribute tagging.

Real eval records rarely carry the synthetic attributes (tone, persona, topic,
...). ``AttributeTagger`` assigns them so that errors on real data map back to
the attribute space the generator controls:

for every categorical attribute, a linear classifier on hashed n-gram features
is trained on the *synthetic* records (where attributes are known by
construction) and predicts the attribute of each real record. Records that
already carry ``_attrs`` keep them.

``slice_errors(records, correct)`` -> per attribute value: n, errors, error rate,
lift over the overall error rate.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch

from topologies.synth.errors import SynthError
from topologies.synth.textutil import text_of


class AttributeTagger:
    def __init__(self, attributes: Sequence[str] | None = None, *, key: Any = None,
                 max_values: int = 50, min_accuracy: float = 0.0) -> None:
        self.attributes = list(attributes) if attributes else None
        self.key = key
        self.max_values, self.min_accuracy = int(max_values), float(min_accuracy)
        self.models: dict[str, tuple[torch.nn.Module, list[str], float]] = {}
        self.values: dict[str, list[str]] = {}

    def fit(self, synthetic: Sequence[dict]) -> AttributeTagger:
        from topologies.synth.ops.label import featurize, out_of_fold_probs, train_linear

        recs = list(synthetic)
        if not recs:
            raise SynthError("AttributeTagger.fit needs synthetic records.", stage="slices")
        names = self.attributes or sorted({k for r in recs[:200] for k in r.get("_attrs") or {}})
        for a in names:
            vals = [(r.get("_attrs") or {}).get(a) for r in recs]
            if not all(isinstance(v, (str, int, bool)) for v in vals):
                continue
            uniq = sorted({str(v) for v in vals})
            if len(uniq) < 2 or len(uniq) > self.max_values:
                continue
            self.values[a] = uniq
            x = featurize(recs, key=self.key)
            y = torch.tensor([uniq.index(str(v)) for v in vals])
            oof = out_of_fold_probs(x, y, len(uniq), folds=3, epochs=80)
            acc = float((oof.argmax(1) == y).float().mean())
            if acc < self.min_accuracy:
                continue
            self.models[a] = (train_linear(x, y, len(uniq), epochs=80), uniq, acc)
        return self

    @property
    def accuracy(self) -> dict[str, float]:
        """Cross-validated tagging accuracy per attribute (on synthetic data)."""
        return {a: acc for a, (_, _, acc) in self.models.items()}

    def tag(self, records: Sequence[dict]) -> list[dict]:
        """Records with ``_attrs`` filled in (existing attributes are kept)."""
        from topologies.synth.ops.label import featurize

        recs = [dict(r) for r in records]
        if not self.models:
            return recs
        x = featurize(recs, key=self.key)
        for a, (m, uniq, _) in self.models.items():
            with torch.no_grad():
                pred = m(x).argmax(1).tolist()
            for r, p in zip(recs, pred):
                r["_attrs"] = {**(r.get("_attrs") or {})}
                r["_attrs"].setdefault(a, uniq[p])
        return recs


def slice_errors(records: Sequence[dict], correct: Sequence[bool], *,
                 attributes: Sequence[str] | None = None, min_n: int = 3) -> list[dict[str, Any]]:
    """Error rate per attribute value (from ``_attrs`` or top-level fields)."""
    overall = 1 - sum(correct) / max(1, len(correct))
    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r, ok in zip(records, correct):
        attrs = dict(r.get("_attrs") or {})
        for a in attributes or []:
            if a in r and a not in attrs:
                attrs[a] = r[a]
        for a, v in attrs.items():
            if attributes is not None and a not in attributes:
                continue
            if not isinstance(v, (str, int, bool)):
                continue
            c = cells[(a, str(v))]
            c[0] += 1
            c[1] += 0 if ok else 1
    out = []
    for (a, v), (n, e) in cells.items():
        if n < min_n:
            continue
        rate = e / n
        out.append({"attribute": a, "value": v, "n": n, "errors": e, "error_rate": rate,
                    "lift": rate - overall})
    return sorted(out, key=lambda d: (-d["lift"], d["attribute"], d["value"]))


def text_key(records: Sequence[dict]) -> Any:
    return None if not records else [k for k, v in records[0].items()
                                     if not str(k).startswith("_") and isinstance(v, str)]


__all__ = ["AttributeTagger", "slice_errors", "text_key"]

_ = text_of
