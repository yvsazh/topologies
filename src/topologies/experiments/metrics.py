"""Metric registry: optimization direction, display format, per-token comparability.

    from topologies.experiments import register_metric
    register_metric("bleu", "max", fmt=".2f", description="corpus BLEU")

Lookups strip ``train_`` / ``val_`` / ``test_`` prefixes, so registering
``bleu`` covers ``val_bleu``. Unknown metrics get a mode inferred from the
name (``acc|f1|auc|bleu|rouge|_per_s|throughput|score`` -> max, otherwise
min), flagged ``inferred=True`` (tables show ``?``; override with ``goal=``).

``per_token`` marks metrics that depend on how text is split into tokens
(``loss``, ``ppl`` and token ``acc`` under the token recipes); they are not
comparable across tokenizers, unlike ``bpb`` / ``bpc``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

PREFIXES = ("train_", "val_", "test_")
TOKEN_RECIPES = frozenset({"causal-lm", "masked-lm", "permutation-lm"})
_MAX_RE = re.compile(r"(acc|f1|auc|bleu|rouge|_per_s|throughput|score|precision|recall|top\d)",
                     re.IGNORECASE)


@dataclass(frozen=True)
class MetricInfo:
    name: str
    mode: str  # "min" | "max"
    per_token: bool = False
    fmt: str = ".3f"
    description: str = ""
    inferred: bool = False

    @property
    def higher_is_better(self) -> bool:
        return self.mode == "max"

    def better(self, a: float, b: float) -> bool:
        """True when ``a`` is strictly better than ``b``."""
        return a > b if self.mode == "max" else a < b

    def format(self, v: float | None) -> str:
        if v is None:
            return "—"
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)
        if math.isnan(f):
            return "nan"
        a = abs(f)
        if self.fmt.endswith("f") and a != 0 and (a < 1e-3 or a >= 1e6):
            return f"{f:.2e}"
        try:
            return format(f, self.fmt)
        except (TypeError, ValueError):
            return str(f)


_REGISTRY: dict[str, MetricInfo] = {}


def register_metric(name: str, mode: str, *, per_token: bool = False, fmt: str = ".3f",
                    higher_is_better: bool | None = None, description: str = "") -> MetricInfo:
    """Declare how ``name`` is optimized and shown (``mode`` = ``"min"`` | ``"max"``)."""
    if higher_is_better is not None:
        mode = "max" if higher_is_better else "min"
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max'; got {mode!r}.")
    info = MetricInfo(name=name, mode=mode, per_token=per_token, fmt=fmt,
                      description=description)
    _REGISTRY[name] = info
    return info


for _n, _m, _pt, _f, _d in (
    ("loss", "min", True, ".4f", "cross-entropy / objective (nats)"),
    ("ppl", "min", True, ".2f", "perplexity, exp(loss)"),
    ("acc", "max", True, ".4f", "accuracy"),
    ("bpb", "min", False, ".4f", "bits per UTF-8 byte (tokenizer-independent)"),
    ("bpc", "min", False, ".4f", "bits per character (tokenizer-independent)"),
    ("top5", "max", False, ".4f", "top-5 accuracy"),
    ("acc_answer", "max", False, ".4f", "accuracy on answer positions (probes)"),
    ("exact_match", "max", False, ".4f", "sequence exact match"),
    ("f1", "max", False, ".4f", "micro F1"),
    ("mae", "min", False, ".4f", "mean absolute error"),
    ("synth_seconds", "min", False, ".1f", "synthetic data generation time (seconds)"),
    ("objective", "min", True, ".4f", "training objective (loss override)"),
    ("wall_s", "min", False, ".1f", "wall-clock seconds per trial"),
    ("train_s", "min", False, ".1f", "seconds in fit()"),
    ("eval_s", "min", False, ".1f", "seconds in final evaluation"),
    ("params", "min", False, ".0f", "parameters"),
    ("trainable_params", "min", False, ".0f", "trainable parameters"),
    ("embedding_params", "min", False, ".0f", "embedding parameters"),
    ("tokens_per_s", "max", False, ".0f", "training tokens per second"),
    ("samples_per_s", "max", False, ".1f", "training samples per second"),
    ("peak_mem_mb", "min", False, ".1f", "peak memory (MB)"),
    ("flops_est", "min", False, ".3g", "estimated training FLOPs"),
    ("steps", "min", False, ".0f", "optimizer steps"),
    ("tokens_seen", "min", False, ".0f", "training tokens"),
    ("fwd_ms", "min", False, ".3f", "forward latency (ms, median)"),
    ("fwd_bwd_ms", "min", False, ".3f", "forward+backward latency (ms, median)"),
    ("fwd_flops", "min", False, ".3g", "FLOPs of one forward pass (counted)"),
):
    register_metric(_n, _m, per_token=_pt, fmt=_f, description=_d)


def base_name(name: str) -> str:
    for p in PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def metric_info(name: str, *, recipe: str | None = None) -> MetricInfo:
    """Registered info for ``name`` (prefix-aware); per-token only under token recipes."""
    info = _REGISTRY.get(name) or _REGISTRY.get(base_name(name))
    if info is None:
        mode = "max" if _MAX_RE.search(name) else "min"
        return MetricInfo(name=name, mode=mode, inferred=True)
    per_token = info.per_token
    if per_token and info.name in ("loss", "ppl", "acc", "objective"):
        per_token = recipe in TOKEN_RECIPES if recipe is not None else info.per_token
    return replace(info, name=name, per_token=per_token)


def list_metrics() -> list[MetricInfo]:
    return [ _REGISTRY[k] for k in sorted(_REGISTRY)]


__all__ = ["MetricInfo", "base_name", "list_metrics", "metric_info", "register_metric"]
