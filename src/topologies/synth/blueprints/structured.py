"""Structured blueprints: tabular SCMs and time series."""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from topologies.synth.blueprints._base import blueprint, default_splits, spec_kwargs
from topologies.synth.errors import SynthError


def json_schema_of(model: Any) -> dict[str, Any]:
    if isinstance(model, Mapping):
        return dict(model)
    fn = getattr(model, "model_json_schema", None)
    if callable(fn):
        return fn()
    raise SynthError(f"Cannot get a JSON schema from {model!r}.", stage="blueprint",
                     hint="Pass a pydantic model class or a JSON schema dict.")


def schema_example(rng: Any, *, schema: Mapping[str, Any]) -> Any:
    """A random schema-valid object (realistic values need ``make=fn(rng)``)."""
    from topologies.synth.jsonutil import example_from_schema

    return example_from_schema(dict(schema), rng)


def target_json(attrs: Mapping[str, Any]) -> str:
    v = attrs["target"]
    if hasattr(v, "model_dump"):
        v = v.model_dump(mode="json")
    return json.dumps(v, ensure_ascii=False, default=str)


def make_invoice(rng: Any) -> dict[str, Any]:
    """Example ``make=`` program: a realistic invoice (vendor, dates, line items, total)."""
    from topologies.synth.lib.text.faker import Faker

    f = Faker(rng.choice(["en", "de"]), rng)
    items = [{"description": f.product(), "quantity": rng.randint(1, 5),
              "unit_price": round(f.amount(40), 2)} for _ in range(rng.randint(1, 3))]
    total = round(sum(i["quantity"] * i["unit_price"] for i in items), 2)
    return {"vendor": f.company(), "invoice_number": f.id("INV-", 6), "date": f.date(),
            "currency": f.currency(), "items": items, "total": total}


INVOICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"}, "invoice_number": {"type": "string"},
        "date": {"type": "string"}, "currency": {"type": "string"},
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "description": {"type": "string"}, "quantity": {"type": "integer"},
            "unit_price": {"type": "number"}},
            "required": ["description", "quantity", "unit_price"]}},
        "total": {"type": "number"}},
    "required": ["vendor", "invoice_number", "date", "currency", "items", "total"]}


# -- tabular ----------------------------------------------------------------------------------
def linear_score(*, coef: Mapping[str, float], cat_coef: Mapping[str, Mapping[str, float]],
                 means: Mapping[str, float], stds: Mapping[str, float], bias: float,
                 **cols: Any) -> Any:
    """Logistic-regression score over SCM parent columns (numeric standardized, categorical
    one weight per value) — the label mechanism ``tabular(real=...)`` fits."""
    out: Any = bias
    for name, w in coef.items():
        x = cols[name]
        out = out + w * (x - means[name]) / stds[name]
    for name, ws in cat_coef.items():
        c = cols[name]
        for value, w in ws.items():
            out = out + w * c.isin([value]).double()
    return out


def _fit_logistic(rows: list[dict[str, Any]], label: str, positive: Any, numeric: list[str],
                  categorical: dict[str, list[Any]]) -> dict[str, Any]:
    import torch

    means = {c: sum(float(r[c]) for r in rows if r.get(c) is not None) /
             max(1, sum(1 for r in rows if r.get(c) is not None)) for c in numeric}
    stds = {}
    for c in numeric:
        vals = [float(r[c]) for r in rows if r.get(c) is not None]
        m = means[c]
        stds[c] = math.sqrt(sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) or 1.0
    feats = []
    for r in rows:
        f = [((float(r[c]) if r.get(c) is not None else means[c]) - means[c]) / stds[c]
             for c in numeric]
        for c, vals in categorical.items():
            f += [1.0 if r.get(c) == v else 0.0 for v in vals]
        feats.append(f)
    x = torch.tensor(feats, dtype=torch.float64)
    y = torch.tensor([1.0 if r.get(label) == positive else 0.0 for r in rows],
                     dtype=torch.float64)
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200)

    def closure() -> Any:
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, y) + \
            1e-3 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    wl = [round(float(v), 6) for v in w.detach()]
    coef = dict(zip(numeric, wl[:len(numeric)]))
    cat_coef: dict[str, dict[str, float]] = {}
    k = len(numeric)
    for c, vals in categorical.items():
        cat_coef[c] = {str(v): wl[k + i] for i, v in enumerate(vals)}
        k += len(vals)
    return {"coef": coef, "cat_coef": cat_coef, "means": {k: round(v, 6) for k, v in
                                                          means.items()},
            "stds": {k: round(v, 6) for k, v in stds.items()}, "bias": round(float(b), 6)}


def scm_from_real(real: Any, *, label: str | None = None, rows: int = 256) -> Any:
    """An SCM fitted to ``real``: numeric columns (normal + quantile calibration and a
    Gaussian copula), categorical columns (frequencies), and a binary ``label`` as a
    logistic mechanism of the other columns with the real prevalence."""
    from topologies.synth.lib.tabular import SCM
    from topologies.synth.records import as_records

    data = as_records(real, what="real")
    if not data:
        raise SynthError("tabular(real=...) needs at least one real row.", stage="blueprint")
    cols = [c for c in data[0] if not str(c).startswith("_")]
    numeric, categorical = [], {}
    for c in cols:
        vals = [r.get(c) for r in data if r.get(c) is not None]
        if c == label:
            continue
        if vals and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            numeric.append(c)
        else:
            categorical[c] = sorted({str(v) for v in vals})
    m = SCM(rows=rows)
    for c in numeric:
        vals = [float(r[c]) for r in data if r.get(c) is not None]
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) or 1.0
        m.normal(c, mean=round(mean, 6), std=round(std, 6))
    for c, values in categorical.items():
        counts = {v: sum(1 for r in data if str(r.get(c)) == v) for v in values}
        m.categorical(c, {v: counts[v] / len(data) for v in values})
    if label is not None:
        values = sorted({r.get(label) for r in data if r.get(label) is not None}, key=str)
        if len(values) == 2:
            pos = values[1]
            fit = _fit_logistic(data, label, pos, numeric, categorical)
            prev = sum(1 for r in data if r.get(label) == pos) / len(data)
            m.label(label, parents=[*numeric, *categorical],
                    score=functools.partial(linear_score, **fit),
                    prevalence=min(0.999, max(0.001, prev)), values=(values[0], pos))
        else:
            counts = {v: sum(1 for r in data if r.get(label) == v) for v in values}
            m.categorical(label, {v: counts[v] / len(data) for v in values})
    m.calibrate(data, columns=numeric + list(categorical))
    return m


def default_fraud_scm(*, prevalence: float = 0.02) -> Any:
    """The scenario-4 SCM: transactions with a rare, mechanism-driven fraud label."""
    from topologies.synth.lib.tabular import fraud_scm

    return fraud_scm(prevalence=prevalence)


@blueprint(example={"scm": {"call": "topologies.synth.blueprints.structured:default_fraud_scm",
                            "prevalence": 0.02},
                    "label": "fraud", "size": 5000})
def tabular(name: str, *, scm: Any = None, real: Any = None, label: str | None = None,
            size: int = 10000, rows: int = 1024, splits: Any = None, **spec: Any) -> Any:
    """Tabular rows from a structural causal model (scenario 4): give an ``scm`` (exact
    control of imbalance, drift via ``scm.to_records(shift=...)``, missingness), or
    ``real`` rows to fit one (marginals + copula + logistic label mechanism)."""
    from topologies.synth.spec import Spec

    if scm is None:
        if real is None:
            raise SynthError("tabular needs scm= (synth.SCM) or real= rows to fit one.",
                             stage="blueprint",
                             hint="See docs/synth/blueprints/tabular.md for an SCM example.")
        scm = scm_from_real(real, label=label, rows=rows)
    return Spec(name, generator=scm, size=size,
                splits=splits if splits is not None else default_splits(label),
                **spec_kwargs(spec))


# -- time series --------------------------------------------------------------------------------
def anomaly_label(info: Mapping[str, Any]) -> str:
    mask = info.get("anomaly_mask") or []
    return "anomalous" if any(mask) else "normal"


@blueprint(example={"length": 128, "task": "anomaly", "size": 500})
def time_series(name: str, *, length: Any = 128, channels: int = 1,
                components: Sequence[Any] | None = None, task: str = "anomaly",
                cross_corr: float = 0.0, size: int = 2000, splits: Any = None,
                **spec: Any) -> Any:
    """Series composed from trend / seasonality / AR / regimes / spikes / level shifts /
    noise / missing values, with exact anomaly and event masks. ``task``: ``anomaly``
    (per-step masks), ``classify`` (series label anomalous / normal) or ``forecast``
    (use ``ds.for_recipe("regression", horizon=h)``)."""
    from topologies.synth.dists import uniform
    from topologies.synth.lib import series as ts
    from topologies.synth.spec import Spec

    if task not in ("anomaly", "classify", "forecast"):
        from topologies.synth.errors import unknown

        raise unknown("time_series task", task, ["anomaly", "classify", "forecast"],
                      stage="blueprint")
    comps = list(components) if components is not None else [
        ts.trend(slope=uniform(-0.02, 0.02)),
        ts.seasonality(periods=[24], amplitude=uniform(0.5, 2.0)),
        ts.ar(coeffs=[0.6, -0.2], sigma=0.3),
        ts.spikes(rate=0.01 if task != "forecast" else 0.0, scale=5.0),
        ts.level_shift(rate=0.004 if task != "forecast" else 0.0, size=3.0),
        ts.noise(sigma=0.2)]
    gen = ts.TimeSeries(length=length, channels=channels, components=comps,
                        cross_corr=cross_corr,
                        label=anomaly_label if task == "classify" else None)
    return Spec(name, generator=gen, size=size,
                splits=splits if splits is not None else default_splits(
                    "label" if task == "classify" else None),
                **spec_kwargs(spec))


__all__ = ["default_fraud_scm", "make_invoice", "scm_from_real",
           "tabular", "time_series"]
