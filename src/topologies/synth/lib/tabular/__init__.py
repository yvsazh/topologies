"""Structural causal models for tabular data (vectorized torch sampling).

    scm = synth.tabular.SCM()
    scm.categorical("region", {"north": 0.3, "south": 0.5, "west": 0.2})
    scm.normal("age", mean=42, std=13, clip=(18, 90))
    scm.latent("thrift", synth.tabular.Normal(0, 1))                  # hidden confounder
    scm.node("income", parents=["age", "region", "thrift"],
             fn=lambda age, region, thrift: 18_000 + 650 * age + 9_000 * (region == "north")
                                            - 3_000 * thrift,
             noise=synth.tabular.Normal(0, 6_000), clip=(0, None))
    scm.node("amount", parents=["income"], fn=lambda income: 0.002 * income,
             noise=synth.tabular.LogNormal(0, 0.8), mode="mul")
    scm.label("fraud", parents=["amount", "age", "thrift"],
              score=lambda amount, age, thrift: 0.8 * amount.log() - 0.03 * age + thrift,
              prevalence=0.01)                                       # threshold solved once
    scm.missing("income", "MAR", rate=0.1, depends_on="age")
    data = scm.sample(1_000_000, seed=0)                             # dict of tensors, seconds
    drift = scm.sample(100_000, shift={"region": {"north": 0.7, "south": 0.2, "west": 0.1},
                                       "age": {"mean": 55}})
    scm.calibrate(real_records)              # quantile-match marginals + Gaussian copula roots

Inside ``fn`` / ``score`` numeric parents are float tensors and categorical
parents are ``Cat`` columns (``region == "north"``, ``region.isin([...])`` give
bool tensors). An SCM is a node: as a generator it emits ``rows`` records per raw
index (vectorized per call; seeded by the node's ``(seed, index, path)``), and
``scm.sample(n, seed=...)`` draws ``n`` rows directly. Labels are exact by
construction (evidence ``construction``). ``SCM(preset="fraud")`` is a ready-made
transaction model with a rare, mechanism-driven fraud label.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from topologies import synth

Recorded = synth.Recorded
logged = synth.logged
SynthError = synth.SynthError
unknown = synth.unknown
fp_canonical = synth.canonical

# ---------------------------------------------------------------- noise models


class Noise(Recorded):
    def sample(self, n: int, g: torch.Generator) -> torch.Tensor:
        raise NotImplementedError

    def to_config(self) -> dict[str, Any]:
        return {"noise": type(self).__name__, **{k: v for k, v in vars(self).items()}}


class Normal(Noise):
    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean, self.std = float(mean), float(std)

    def sample(self, n: int, g: torch.Generator) -> torch.Tensor:
        return torch.randn(n, generator=g, dtype=torch.float64) * self.std + self.mean


class LogNormal(Noise):
    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean, self.std = float(mean), float(std)

    def sample(self, n: int, g: torch.Generator) -> torch.Tensor:
        return torch.exp(torch.randn(n, generator=g, dtype=torch.float64) * self.std + self.mean)


class StudentT(Noise):
    def __init__(self, df: float = 3.0, scale: float = 1.0) -> None:
        self.df, self.scale = float(df), float(scale)

    def sample(self, n: int, g: torch.Generator) -> torch.Tensor:
        z = torch.randn(n, generator=g, dtype=torch.float64)
        k = max(1, round(self.df))
        chi = (torch.randn(n, k, generator=g, dtype=torch.float64) ** 2).sum(1)
        return z / torch.sqrt(chi / k) * self.scale


class Laplace(Noise):
    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        self.loc, self.scale = float(loc), float(scale)

    def sample(self, n: int, g: torch.Generator) -> torch.Tensor:
        u = torch.rand(n, generator=g, dtype=torch.float64) - 0.5
        return self.loc - self.scale * torch.sign(u) * torch.log1p(-2 * u.abs().clamp(
            max=0.4999999))


class Uniform(Noise):
    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        self.low, self.high = float(low), float(high)

    def sample(self, n: int, g: torch.Generator) -> torch.Tensor:
        return torch.rand(n, generator=g, dtype=torch.float64) * (self.high - self.low) + \
            self.low


# ---------------------------------------------------------------- categorical columns


class Cat(Recorded):
    """A categorical column inside ``fn``: compare with values, ``isin``, ``codes``."""

    def __init__(self, codes: torch.Tensor, values: Sequence[Any]) -> None:
        self.codes, self.values = codes, list(values)

    def __eq__(self, other: object) -> torch.Tensor:  # type: ignore[override]
        if other not in self.values:
            return torch.zeros_like(self.codes, dtype=torch.bool)
        return self.codes == self.values.index(other)

    def __ne__(self, other: object) -> torch.Tensor:  # type: ignore[override]
        return ~self.__eq__(other)

    __hash__ = None  # type: ignore[assignment]

    def isin(self, values: Sequence[Any]) -> torch.Tensor:
        out = torch.zeros_like(self.codes, dtype=torch.bool)
        for v in values:
            out |= self == v
        return out

    def onehot(self) -> torch.Tensor:
        return torch.nn.functional.one_hot(self.codes, len(self.values)).double()


# ---------------------------------------------------------------- nodes


class _Node:
    def __init__(self, name: str, kind: str, parents: Sequence[str] = (), **cfg: Any) -> None:
        self.name, self.kind, self.parents, self.cfg = name, kind, list(parents), cfg
        self.hidden = kind == "latent"

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "parents": self.parents,
                "cfg": fp_canonical(self.cfg)}


def _call(fn: Callable[..., Any], cols: Mapping[str, Any], parents: Sequence[str]) -> Any:
    try:
        names = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        names = list(parents)
    if any(p in names for p in parents):
        return fn(**{p: cols[p] for p in parents if p in names})
    return fn(*[cols[p] for p in parents])


class SCM(synth.Node):
    """A structural causal model (see module docstring)."""

    name = "scm"
    evidence = "construction"
    tags = ("tabular", "causal", "scm")
    example = {"rows": 8, "preset": "fraud"}

    def __init__(self, *, rows: int = 64, name: str | None = None,
                 preset: str | None = None, prevalence: float = 0.02) -> None:
        self.nodes: dict[str, _Node] = {}
        self.rows = int(rows)
        if name:
            self.name = name
        self._thresholds: dict[str, float] = {}
        self._missing: list[dict[str, Any]] = []
        self._calibration: dict[str, Any] = {}
        if preset is not None:
            if preset not in PRESETS:
                raise unknown("SCM preset", preset, PRESETS, stage="scm")
            self.__dict__["_call_depth"] = 1  # preset declarations are not builder calls
            try:
                PRESETS[preset](self, prevalence=float(prevalence))
            finally:
                self.__dict__["_call_depth"] = 0

    def params(self) -> dict[str, Any]:
        return {}

    def node_kind(self) -> str:
        return self.name if isinstance(self.name, str) else "scm"

    # -- declaration ---------------------------------------------------------------
    def _add(self, node: _Node) -> SCM:
        if node.name in self.nodes:
            raise SynthError(f"SCM node {node.name!r} defined twice.", stage="scm")
        for p in node.parents:
            if p not in self.nodes:
                raise unknown("parent node", p, self.nodes, stage="scm",
                              extra="Declare parents before children (the model is a DAG).")
        self.nodes[node.name] = node
        self._thresholds.clear()
        return self

    @logged
    def categorical(self, name: str, probs: Mapping[Any, float] | Sequence[Any], *,
                    parents: Sequence[str] = (), fn: Callable[..., Any] | None = None) -> SCM:
        """Root categorical (``probs``) or conditional: ``fn(**parents) -> logits [n, K]``."""
        if isinstance(probs, Mapping):
            values, w = list(probs), [float(x) for x in probs.values()]
        else:
            values, w = list(probs), [1.0] * len(probs)
        if parents and fn is None:
            raise SynthError(f"{name}: conditional categorical needs fn=...", stage="scm")
        return self._add(_Node(name, "categorical", parents, values=values, probs=w, fn=fn))

    @logged
    def normal(self, name: str, *, mean: float = 0.0, std: float = 1.0,
               clip: tuple[float | None, float | None] | None = None, integer: bool = False
               ) -> SCM:
        return self._add(_Node(name, "normal", (), mean=mean, std=std, clip=clip,
                               integer=integer))

    @logged
    def uniform(self, name: str, *, low: float = 0.0, high: float = 1.0,
                integer: bool = False) -> SCM:
        return self._add(_Node(name, "uniform", (), low=low, high=high, integer=integer))

    @logged
    def lognormal(self, name: str, *, mean: float = 0.0, std: float = 1.0,
                  clip: tuple[float | None, float | None] | None = None) -> SCM:
        return self._add(_Node(name, "lognormal", (), mean=mean, std=std, clip=clip))

    @logged
    def poisson(self, name: str, *, rate: float = 1.0) -> SCM:
        return self._add(_Node(name, "poisson", (), rate=rate))

    @logged
    def latent(self, name: str, dist: Noise | None = None) -> SCM:
        """A hidden confounder (used by children, never emitted)."""
        return self._add(_Node(name, "latent", (), dist=dist or Normal()))

    @logged
    def node(self, name: str, *, parents: Sequence[str], fn: Callable[..., Any],
             noise: Noise | None = None, mode: str = "add",
             clip: tuple[float | None, float | None] | None = None, integer: bool = False,
             values: Sequence[Any] | None = None) -> SCM:
        """Numeric child ``fn(**parents) (+|*) noise``; ``values=[...]`` makes ``fn`` return
        category codes (or logits ``[n, K]``)."""
        if mode not in ("add", "mul"):
            raise unknown("noise mode", mode, ["add", "mul"], stage="scm")
        return self._add(_Node(name, "derived", parents, fn=fn, noise=noise, mode=mode,
                               clip=clip, integer=integer, values=list(values) if values
                               else None))

    @logged
    def label(self, name: str, *, parents: Sequence[str], score: Callable[..., Any],
              prevalence: float | None = None, threshold: float | None = None,
              noise: Noise | None = None, values: tuple[Any, Any] = (0, 1)) -> SCM:
        """Binary label ``score + noise > threshold``; ``prevalence`` solves the threshold
        once (on a 200k calibration sample) so the base rate is exact in expectation."""
        if (prevalence is None) == (threshold is None):
            raise SynthError(f"label {name!r} needs exactly one of prevalence / threshold.",
                             stage="scm")
        if prevalence is not None and not 0 < prevalence < 1:
            raise SynthError("prevalence must be in (0, 1).", stage="scm")
        return self._add(_Node(name, "label", parents, score=score, prevalence=prevalence,
                               threshold=threshold, noise=noise or Normal(0, 0.0),
                               values=list(values)))

    @logged
    def missing(self, name: str, mechanism: str = "MCAR", *, rate: float = 0.1,
                depends_on: str | None = None, strength: float = 2.0) -> SCM:
        """Missingness: ``MCAR`` (uniform), ``MAR`` (depends on another observed column) or
        ``MNAR`` (depends on the column itself); ``rate`` is the expected share."""
        if mechanism not in ("MCAR", "MAR", "MNAR"):
            raise unknown("missingness mechanism", mechanism, ["MCAR", "MAR", "MNAR"],
                          stage="scm")
        if name not in self.nodes:
            raise unknown("node", name, self.nodes, stage="scm")
        if mechanism == "MAR" and depends_on not in self.nodes:
            raise SynthError("MAR missingness needs depends_on=<another node>.", stage="scm")
        self._missing.append({"name": name, "mechanism": mechanism, "rate": float(rate),
                              "depends_on": depends_on, "strength": float(strength)})
        return self

    # -- sampling ----------------------------------------------------------------------
    def _root(self, node: _Node, n: int, g: torch.Generator, shift: Mapping[str, Any]) -> Any:
        c = {**node.cfg, **(shift if isinstance(shift, Mapping) else {})}
        k = node.kind
        if k == "categorical":
            values = node.cfg["values"]
            probs = c["probs"]
            if isinstance(shift, Mapping) and shift and \
                    set(shift) <= set(map(str, values)) | set(values):
                probs = [float(shift.get(v, shift.get(str(v), 0.0))) for v in values]
            p = torch.tensor(probs, dtype=torch.float64)
            return Cat(torch.multinomial(p / p.sum(), n, replacement=True, generator=g), values)
        if k == "normal":
            x = torch.randn(n, generator=g, dtype=torch.float64) * c["std"] + c["mean"]
        elif k == "uniform":
            x = torch.rand(n, generator=g, dtype=torch.float64) * (c["high"] - c["low"]) + \
                c["low"]
        elif k == "lognormal":
            x = torch.exp(torch.randn(n, generator=g, dtype=torch.float64) * c["std"] +
                          c["mean"])
        elif k == "poisson":
            x = torch.poisson(torch.full((n,), float(c["rate"]), dtype=torch.float64),
                              generator=g)
        elif k == "latent":
            x = c["dist"].sample(n, g)
        else:  # pragma: no cover
            raise SynthError(f"unknown root kind {k}", stage="scm")
        return _post(x, c.get("clip"), c.get("integer", False))

    def _solve_thresholds(self) -> None:
        todo = [n for n in self.nodes.values() if n.kind == "label" and
                n.cfg["prevalence"] is not None and n.name not in self._thresholds and
                n.name not in self._calibration.get("logistic", {})]
        if not todo:
            return
        cols = self._forward(200_000, torch.Generator().manual_seed(12345), {}, scores=True)
        for node in todo:
            s = cols[f"__score_{node.name}"].float()
            q = torch.quantile(s[:200_000].float(), 1 - node.cfg["prevalence"])
            self._thresholds[node.name] = float(q)

    def _forward(self, n: int, g: torch.Generator, shift: Mapping[str, Any], *,
                 scores: bool = False) -> dict[str, Any]:
        cols: dict[str, Any] = {}
        copula = self._calibration.get("copula")
        if copula is not None:
            cols.update(self._sample_copula(n, g, copula))
        for name, node in self.nodes.items():
            sh = shift.get(name, {})
            if name in cols:
                continue
            if callable(sh) and not isinstance(sh, Mapping):
                pass
            if node.kind in ("categorical",) and not node.parents or node.kind in ("normal", "uniform", "lognormal", "poisson", "latent"):
                cols[name] = self._root(node, n, g, sh)
            elif node.kind == "categorical":
                logits = torch.as_tensor(_call(node.cfg["fn"], cols, node.parents),
                                         dtype=torch.float64)
                codes = torch.multinomial(torch.softmax(logits, -1), 1, generator=g).squeeze(-1)
                cols[name] = Cat(codes, node.cfg["values"])
            elif node.kind == "derived":
                x = torch.as_tensor(_call(node.cfg["fn"], cols, node.parents),
                                    dtype=torch.float64)
                if x.dim() == 0:
                    x = x.expand(n).clone()
                if node.cfg["values"]:
                    if x.dim() == 2:
                        x = torch.multinomial(torch.softmax(x, -1), 1, generator=g).squeeze(-1)
                    cols[name] = Cat(x.long().clamp(0, len(node.cfg["values"]) - 1),
                                     node.cfg["values"])
                    continue
                nz = node.cfg["noise"]
                if nz is not None:
                    e = nz.sample(n, g)
                    x = x + e if node.cfg["mode"] == "add" else x * e
                cols[name] = _post(x, node.cfg["clip"], node.cfg["integer"])
            elif node.kind == "label":
                fitted = self._calibration.get("logistic", {}).get(name)
                if fitted is not None:
                    design = _runtime_design(cols, node.parents, fitted)
                    probability = (torch.full((n,), fitted["constant"], dtype=torch.float64)
                                   if "constant" in fitted else torch.sigmoid(design @ fitted["beta"]))
                    cols[name] = Cat((torch.rand(n, generator=g, dtype=torch.float64) <
                                      probability).long(), node.cfg["values"])
                    continue
                s = torch.as_tensor(_call(node.cfg["score"], cols, node.parents),
                                    dtype=torch.float64)
                nz = node.cfg["noise"]
                if getattr(nz, "std", 1.0) > 0:
                    s = s + nz.sample(n, g)
                if scores:
                    cols[f"__score_{name}"] = s
                    cols[name] = torch.zeros(n)
                    continue
                thr = node.cfg["threshold"] if node.cfg["threshold"] is not None else \
                    self._thresholds[name]
                cols[name] = Cat((s > thr).long(), node.cfg["values"])
            # Calibration is applied as each node is made: descendants and fitted
            # label mechanisms must see the same calibrated parent coordinates.
            if name in cols and name not in (self._calibration.get("copula") or {}).get("roots", []):
                self._calibrate_column(name, cols)
            if isinstance(sh, Mapping) and node.kind not in ("categorical", "normal",
                                                              "uniform", "lognormal",
                                                              "poisson", "latent"):
                if "scale" in sh and not isinstance(cols[name], Cat):
                    cols[name] = cols[name] * float(sh["scale"])
                if "offset" in sh and not isinstance(cols[name], Cat):
                    cols[name] = cols[name] + float(sh["offset"])
            if callable(sh) and not isinstance(sh, Mapping):
                cols[name] = sh(cols[name])
        return cols

    def sample(self, n: Any, *, seed: int = 0, shift: Mapping[str, Any] | None = None,
               hidden: bool = False) -> Any:
        """``n`` rows as ``{column: tensor}`` (categoricals as code tensors + ``.values``
        in ``self.values(name)``). ``shift`` overrides root parameters / probabilities, or
        scales / offsets / transforms derived columns (drift scenarios).

        Given a node context instead of ``n`` (inside a generator), returns ``rows``
        records."""
        if not isinstance(n, int):
            return self._records_for(n)
        self._solve_thresholds()
        g = torch.Generator().manual_seed(int(seed))
        cols = self._forward(int(n), g, dict(shift or {}))
        out: dict[str, Any] = {}
        for name, node in self.nodes.items():
            if node.hidden and not hidden:
                continue
            v = cols[name]
            v = v.codes if isinstance(v, Cat) else v
            out[name] = v
        self._apply_missing(out, g)
        return out

    def values(self, name: str) -> list[Any] | None:
        node = self.nodes[name]
        return node.cfg.get("values")

    def _apply_missing(self, out: dict[str, Any], g: torch.Generator) -> None:
        for m in self._missing:
            col = out.get(m["name"])
            if col is None:
                continue
            n = col.shape[0]
            if m["mechanism"] == "MCAR":
                p = torch.full((n,), m["rate"], dtype=torch.float64)
            else:
                ref = out[m["depends_on"]] if m["mechanism"] == "MAR" else col
                z = (ref.double() - ref.double().nanmean()) / ref.double().std().clamp(min=1e-9)
                z = torch.nan_to_num(z)
                offset = _solve_offset(z * m["strength"], m["rate"])
                p = torch.sigmoid(z * m["strength"] + offset)
            mask = torch.rand(n, generator=g, dtype=torch.float64) < p
            if col.is_floating_point():
                out[m["name"]] = col.masked_fill(mask, float("nan"))
            else:
                out[m["name"]] = col.masked_fill(mask, -1)

    def to_records(self, n: int, *, seed: int = 0, shift: Mapping[str, Any] | None = None
                   ) -> list[dict[str, Any]]:
        cols = self.sample(n, seed=seed, shift=shift)
        out_cols = {}
        for name, v in cols.items():
            vals = self.values(name)
            if vals is not None:
                out_cols[name] = [vals[i] if i >= 0 else None for i in v.tolist()]
            else:
                out_cols[name] = [None if (isinstance(x, float) and math.isnan(x)) else x
                                  for x in v.tolist()]
        names = list(out_cols)
        return [dict(zip(names, row)) for row in zip(*(out_cols[c] for c in names))]

    # -- calibration -----------------------------------------------------------------
    @logged
    def calibrate(self, real: Any, *, columns: Sequence[str] | None = None,
                  copula: bool = True, source_size: int = 16384,
                  source_seed: int = 24680) -> SCM:
        """Match marginals of ``real`` (quantile mapping of numeric columns, frequencies of
        root categoricals) and the correlation of root numeric nodes (Gaussian copula)."""
        as_records = synth.as_records

        rows = as_records(real, what="calibration data")
        names = list(columns) if columns is not None else [
            n for n, node in self.nodes.items() if not node.hidden and n in (rows[0] if rows
                                                                             else {})]
        if source_size < 2:
            raise SynthError("source_size must be at least 2.", stage="scm")
        # A recorded, fixed source window defines the pointwise baseline CDF.
        # It is independent of requested sample batch size and seed.
        self._solve_thresholds()
        self._calibration = {}
        source = self._forward(source_size, torch.Generator().manual_seed(source_seed), {})
        quant: dict[str, torch.Tensor] = {}
        baseline: dict[str, torch.Tensor] = {}
        clean: dict[str, list[Any]] = {}
        for name in names:
            node = self.nodes[name]
            vals = [r.get(name) for r in rows if r.get(name) is not None]
            clean[name] = vals
            if not vals:
                continue
            if node.kind == "categorical" and not node.parents:
                counts = {v: sum(1 for x in vals if x == v) for v in node.cfg["values"]}
                total = sum(counts.values()) or 1
                node.cfg["probs"] = [counts[v] / total + 1e-9 for v in node.cfg["values"]]
            elif node.cfg.get("values") is None and node.kind != "label":
                x = torch.tensor([float(v) for v in vals], dtype=torch.float64)
                x = x[torch.isfinite(x)]
                if x.numel() == 0:
                    continue
                quant[name] = torch.quantile(x, torch.linspace(0, 1, 101, dtype=torch.float64))
                if name in source and not isinstance(source[name], Cat):
                    baseline[name] = torch.sort(source[name].double()).values
        self._calibration = {"quantiles": quant, "baseline": baseline, "logistic": {},
                             "source_size": source_size, "source_seed": source_seed}
        roots = [n for n in quant if not self.nodes[n].parents and
                 self.nodes[n].kind in ("normal", "uniform", "lognormal", "poisson")]
        if copula and len(roots) >= 2:
            z = []
            for n in roots:
                x = torch.tensor([float(r[n]) if r.get(n) is not None else float("nan")
                                  for r in rows], dtype=torch.float64)
                z.append(_normal_scores(x))
            zt = torch.nan_to_num(torch.stack(z, 1))
            corr = torch.corrcoef(zt.T)
            self._calibration["copula"] = {"roots": roots, "corr": corr,
                                           "quantiles": {n: quant[n] for n in roots}}
        for node in self.nodes.values():
            if node.kind != "label" or node.name not in clean:
                continue
            fit = _fit_reference_logistic(node, self.nodes, rows, clean[node.name])
            if fit is not None:
                self._calibration["logistic"][node.name] = fit
        return self

    def _sample_copula(self, n: int, g: torch.Generator, cop: dict[str, Any]) -> dict[str, Any]:
        corr = cop["corr"]
        L = torch.linalg.cholesky(corr + 1e-6 * torch.eye(corr.shape[0], dtype=corr.dtype))
        z = torch.randn(n, corr.shape[0], generator=g, dtype=torch.float64) @ L.T
        u = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        out = {}
        for k, name in enumerate(cop["roots"]):
            out[name] = _interp_quantile(u[:, k], cop["quantiles"][name])
        return out

    def _apply_calibration(self, out: dict[str, Any]) -> None:
        quant = self._calibration.get("quantiles") or {}
        roots = set((self._calibration.get("copula") or {}).get("roots", []))
        for name, q in quant.items():
            if name in roots or name not in out:
                continue
            x = out[name].double()
            ranks = torch.argsort(torch.argsort(x)).double() / max(1, x.numel() - 1)
            out[name] = _interp_quantile(ranks, q)

    def _calibrate_column(self, name: str, cols: dict[str, Any]) -> None:
        q = self._calibration.get("quantiles", {}).get(name)
        base = self._calibration.get("baseline", {}).get(name)
        if q is None or base is None or isinstance(cols[name], Cat):
            return
        x = cols[name].double()
        idx = torch.searchsorted(base.contiguous(), x.contiguous()).clamp(0, base.numel() - 1)
        lo = (idx - 1).clamp(min=0)
        hi = idx
        exact = base[hi] == x
        rank = (lo.double() + hi.double()) / 2
        rank = torch.where(exact, rank, idx.double() - 0.5).clamp(0, base.numel() - 1)
        u = rank / (base.numel() - 1)
        cols[name] = _interp_quantile(u, q)

    # -- synth Generator protocol ---------------------------------------------------------
    @property
    def produces(self) -> Any:
        Category, Label, Number, Schema = synth.Category, synth.Label, synth.Number, \
            synth.Schema

        fields: dict[str, Any] = {}
        for name, node in self.nodes.items():
            if node.hidden:
                continue
            vals = node.cfg.get("values")
            optional = any(m["name"] == name for m in self._missing)
            if node.kind == "label":
                fields[name] = Label([str(v) for v in vals], optional=optional)
            elif vals is not None:
                fields[name] = Category([str(v) for v in vals], optional=optional)
            else:
                fields[name] = Number(optional=optional)
        return Schema(fields)

    def _records_for(self, ctx: Any) -> list[dict[str, Any]]:
        recs = self.to_records(self.rows, seed=ctx.rng.seed_value, shift=ctx.attrs.get("shift"))
        for r in recs:
            for k, v in list(r.items()):
                vals = self.values(k)
                if vals is not None and v is not None:
                    r[k] = str(v)
        return recs

    def setup(self, env: Any) -> None:
        self._solve_thresholds()

    def config(self) -> dict[str, Any]:
        return {"type": "scm", "nodes": [n.config() for n in self.nodes.values()],
                "missing": self._missing, "rows": self.rows,
                "calibration": fp_canonical({k: v for k, v in self._calibration.items()})}

    def fingerprint(self) -> str:
        return synth.fingerprint_of(self.config())

    def describe(self) -> str:
        return f"scm({len(self.nodes)} nodes, {self.rows} rows/index)"


def _post(x: torch.Tensor, clip: Any, integer: bool) -> torch.Tensor:
    if clip is not None:
        lo, hi = clip
        x = x.clamp(min=lo if lo is not None else -math.inf, max=hi if hi is not None else
                    math.inf)
    return x.round() if integer else x


def _solve_offset(z: torch.Tensor, rate: float) -> float:
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if float(torch.sigmoid(z + mid).mean()) > rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _normal_scores(x: torch.Tensor) -> torch.Tensor:
    ok = ~torch.isnan(x)
    out = torch.full_like(x, float("nan"))
    v = x[ok]
    ranks = (torch.argsort(torch.argsort(v)).double() + 0.5) / max(1, v.numel())
    out[ok] = math.sqrt(2) * torch.erfinv(2 * ranks - 1)
    return out


def _interp_quantile(u: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    pos = u.clamp(0, 1) * (q.numel() - 1)
    lo = pos.floor().long().clamp(max=q.numel() - 2)
    frac = pos - lo.double()
    return q[lo] * (1 - frac) + q[lo + 1] * frac


def _fit_reference_logistic(node: _Node, nodes: Mapping[str, _Node], rows: list[dict[str, Any]],
                            labels: list[Any]) -> dict[str, Any] | None:
    values = node.cfg["values"]
    selected = [(r, y) for r in rows for y in [r.get(node.name)]
                if y is not None and y in values]
    if not selected:
        return None
    positive = values[1]
    y = torch.tensor([float(label == positive) for _, label in selected], dtype=torch.float64)
    if y.min() == y.max():
        return {"constant": float(y[0]), "parents": list(node.parents), "encoding": [],
                "beta": torch.zeros(1, dtype=torch.float64)}
    columns = []
    enc = []
    keep = torch.ones(len(selected), dtype=torch.bool)
    for parent in node.parents:
        pnode = nodes[parent]
        vals = [r.get(parent) for r, _ in selected]
        if pnode.cfg.get("values") is not None:
            categories = pnode.cfg["values"]
            enc.append({"parent": parent, "categories": categories[1:]})
            columns.extend([torch.tensor([float(v == c) if v is not None else float("nan")
                                           for v in vals], dtype=torch.float64)
                            for c in categories[1:]])
        else:
            enc.append({"parent": parent, "mean": 0.0, "scale": 1.0})
            arr = torch.tensor([float(v) if v is not None else float("nan") for v in vals],
                               dtype=torch.float64)
            keep &= torch.isfinite(arr)
            mean = arr[torch.isfinite(arr)].mean() if torch.isfinite(arr).any() else torch.tensor(0.)
            scale = arr[torch.isfinite(arr)].std().clamp(min=1e-9) if torch.isfinite(arr).any() else torch.tensor(1.)
            enc[-1].update(mean=float(mean), scale=float(scale))
            columns.append((arr - mean) / scale)
    X = torch.stack([torch.ones(len(selected), dtype=torch.float64), *columns], 1)
    keep &= torch.isfinite(X).all(1)
    X, y = X[keep], y[keep]
    if not len(y):
        return None
    beta = torch.zeros(X.shape[1], dtype=torch.float64)
    for _ in range(80):
        p = torch.sigmoid(X @ beta).clamp(1e-8, 1 - 1e-8)
        w = (p * (1 - p)).clamp(min=1e-8)
        h = X.T @ (w[:, None] * X) + 1e-7 * torch.eye(X.shape[1], dtype=X.dtype)
        step = torch.linalg.solve(h, X.T @ (y - p))
        beta += step
        if step.abs().max() < 1e-9:
            break
    return {"parents": list(node.parents), "encoding": enc, "beta": beta}


def _runtime_design(cols: Mapping[str, Any], parents: Sequence[str], fit: Mapping[str, Any]
                    ) -> torch.Tensor:
    n = (cols[parents[0]].codes.numel() if isinstance(cols[parents[0]], Cat)
         else cols[parents[0]].numel())
    parts = [torch.ones(n, dtype=torch.float64)]
    for spec in fit["encoding"]:
        value = cols[spec["parent"]]
        if "categories" in spec:
            codes = value.codes if isinstance(value, Cat) else value.long()
            categories = value.values if isinstance(value, Cat) else []
            parts.extend([(codes == categories.index(c)).double() if c in categories else
                          torch.zeros(n, dtype=torch.float64) for c in spec["categories"]])
        else:
            x = value.double()
            parts.append((x - spec["mean"]) / spec["scale"])
    if "constant" in fit:
        return torch.zeros(n, 1, dtype=torch.float64)
    return torch.stack(parts, 1)


# ---------------------------------------------------------------- presets


def fraud_score(*, amount: Any, tx_last_hour: Any, account_age_days: Any, country: Any,
                channel: Any) -> Any:
    """Fraud mechanism: big amounts, bursts, new accounts, foreign web payments."""
    return (0.9 * torch.log1p(amount) + 0.8 * tx_last_hour.double()
            - 0.002 * account_age_days + 1.5 * country.isin(["foreign"]).double()
            + 0.7 * channel.isin(["web"]).double())


def _fraud(m: SCM, *, prevalence: float = 0.02) -> None:
    m.categorical("channel", {"web": 0.5, "app": 0.35, "pos": 0.15})
    m.categorical("country", {"home": 0.85, "foreign": 0.15})
    m.lognormal("amount", mean=3.5, std=1.0)
    m.poisson("tx_last_hour", rate=1.2)
    m.uniform("account_age_days", low=1, high=3000)
    m.label("fraud", parents=["amount", "tx_last_hour", "account_age_days", "country",
                              "channel"], score=fraud_score, prevalence=prevalence)


PRESETS = {"fraud": _fraud}


def fraud_scm(*, prevalence: float = 0.02, rows: int = 1024) -> SCM:
    """The transaction model with a rare fraud label (``SCM(preset="fraud")``)."""
    return SCM(rows=rows, preset="fraud", prevalence=prevalence)


__all__ = ["PRESETS", "SCM", "Cat", "Laplace", "LogNormal", "Noise", "Normal", "StudentT",
           "Uniform", "fraud_scm", "fraud_score"]
