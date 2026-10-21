"""Evidence-guided, read-only optimization and pipeline recommendations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Recommendation:
    proposal: str
    evidence: tuple[str, ...]
    workload: str | None
    confidence: str
    limitations: tuple[str, ...] = ()
    validation: str = "Run a matched paired experiment before adopting this change."


def recommend(
    *,
    profile: Mapping[str, Any],
    compatibility: Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
) -> list[Recommendation]:
    """Consume already collected evidence only; never invokes model/profile callbacks."""
    compatibility, evaluation = compatibility or {}, evaluation or {}
    out: list[Recommendation] = []
    workload = profile.get("workload") or profile.get("workload_identity")
    weights = profile.get("weight_storage_bytes")
    if isinstance(weights, (int, float)) and weights > 0:
        out.append(
            Recommendation(
                "trial lower-bit weight-only candidates on supported modules",
                (f"weight storage={weights} bytes",),
                workload,
                "low",
                ("Storage alone does not establish latency benefit or quality tolerance.",),
            )
        )
    for op in profile.get("operators", ()):
        if (
            op.get("evidence_kind", "measured") == "measured"
            and op.get("supported") is True
            and op.get("duration_ms", 0) > 0
        ):
            out.append(
                Recommendation(
                    f"benchmark a backend transformation for operator {op.get('name', 'unknown')}",
                    (f"measured operator duration={op['duration_ms']} ms",),
                    workload,
                    "medium",
                    ("Operator attribution may differ from fused kernel timing.",),
                )
            )
    wait = profile.get("loader_wait_fraction")
    if isinstance(wait, (int, float)) and wait >= 0.2:
        out.append(
            Recommendation(
                "investigate loader transforms, batching, and worker settings",
                (f"loader wait fraction={wait}",),
                workload,
                "medium",
                ("Consumer-visible wait is not worker decode attribution.",),
            )
        )
    transfer = profile.get("transfer_fraction")
    if isinstance(transfer, (int, float)) and transfer >= 0.2:
        out.append(
            Recommendation(
                "test pinned memory/nonblocking transfer where supported",
                (f"measured transfer fraction={transfer}",),
                workload,
                "medium",
            )
        )
    for item in evaluation.get("sensitivity", ()):
        if item.get("evidence_kind") == "paired_experiment" and item.get("status") == "sensitive":
            out.append(
                Recommendation(
                    f"retain higher precision or reduce pruning for {item.get('layer', 'layer')}",
                    tuple(item.get("evidence", ())),
                    workload,
                    item.get("confidence", "low"),
                    ("Applies only to the tested cohort, seed, and perturbation.",),
                )
            )
    return out


def sensitivity_sweep(
    model: Any = None,
    layers: list[str] | None = None,
    experiment: Any = None,
    *,
    max_layers: int,
    max_trials: int,
    restore_source: Any = None,
    source_factory: Any = None,
    baseline: Mapping[str, Any] | None = None,
    cohort_identity: str | None = None,
) -> list[Any]:
    """Run a bounded explicit ablation, one independently reconstructed candidate at a time.

    New callers provide ``source_factory``, a callback ``experiment(candidate, layer)``
    and a paired ``baseline`` mapping. Output must attest measured paired evidence
    and matching cohort. The legacy explicit ``model`` path remains supported but
    always invokes ``restore_source`` in a finally block.
    """
    layers = list(layers or [])
    if max_layers < 1 or max_trials < 1 or len(layers) > max_layers or len(layers) > max_trials:
        raise ValueError("sensitivity sweep exceeds explicit layer/trial budget")
    if experiment is None or (source_factory is None and model is None):
        raise ValueError("provide experiment and either source_factory or model")
    try:
        import torch

        torch_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    except ImportError:
        torch = None
        torch_rng = cuda_rng = None
    import random

    py_rng = random.getstate()
    try:
        import numpy as np

        np_rng = np.random.get_state()
    except ImportError:
        np = None
        np_rng = None
    results = []
    try:
        for layer in layers:
            candidate = source_factory() if source_factory is not None else model
            result = experiment(candidate, layer)
            if source_factory is not None:
                if (
                    not isinstance(result, Mapping)
                    or result.get("evidence_kind") != "paired_experiment"
                ):
                    raise ValueError(
                        "sensitivity result must declare evidence_kind='paired_experiment'"
                    )
                if result.get("status") not in {"sensitive", "not_sensitive", "indeterminate"}:
                    raise ValueError("sensitivity result has invalid status")
                if not result.get("cohort_identity") or (
                    cohort_identity and result["cohort_identity"] != cohort_identity
                ):
                    raise ValueError("sensitivity result cohort identity mismatch")
                if baseline is not None and not result.get("baseline_metrics"):
                    raise ValueError("paired sensitivity requires baseline_metrics")
            results.append(result)
    finally:
        if restore_source is not None and model is not None:
            restore_source(model)
        random.setstate(py_rng)
        if np is not None:
            np.random.set_state(np_rng)
        if torch is not None:
            torch.random.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
    return results
