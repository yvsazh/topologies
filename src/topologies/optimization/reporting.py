"""Explicit adapters from optimization/evaluation observations to analysis reports.

Runtime modules and callables are deliberately never traversed or serialized here.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from topologies.analysis.core import AnalysisResult


def _report_value(value: Any) -> Any:
    """Convert supported record values without repr()/str() fallbacks."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("report mapping keys must be strings")
        return {key: _report_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_value(child) for child in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _report_value(getattr(value, field.name)) for field in fields(value)}
    method = getattr(value, "as_dict", None)
    if callable(method):
        return _report_value(method())
    method = getattr(value, "to_dict", None)
    if callable(method):
        return _report_value(method())
    raise TypeError(
        f"No explicit report serializer for {type(value).__module__}.{type(value).__qualname__}"
    )


def evaluation_analysis(result: Any) -> AnalysisResult:
    """Adapt an EvaluationResult to a JSON-safe analysis report."""
    from topologies.evaluation.api import EvaluationResult

    if not isinstance(result, EvaluationResult):
        raise TypeError("result must be EvaluationResult")
    return AnalysisResult(
        _report_value(result),
        kind="evaluation_report",
        name="evaluation",
        provenance={"adapter": "evaluation_analysis"},
    )


def profile_analysis(profile: Any) -> AnalysisResult:
    """Adapt a DataProfile/PipelineProfile using their declared report serializer."""
    if type(profile).__name__ not in {"DataProfile", "PipelineProfile"}:
        raise TypeError("profile must be DataProfile or PipelineProfile")
    return AnalysisResult(
        _report_value(profile),
        kind="profile_report",
        name="profile",
        provenance={"adapter": "profile_analysis"},
    )


def optimization_analysis(result: Any) -> AnalysisResult:
    """Adapt result metadata only; never serialize its live model/runtime."""
    from topologies.optimization.contracts import OptimizationResult

    if not isinstance(result, OptimizationResult):
        raise TypeError("result must be OptimizationResult")
    payload = {
        "config": _report_value(result.config),
        "compatibility": _report_value(result.compatibility),
        "backend": result.backend,
        "validation": result.validation,
        "history": _report_value(result.history),
        "codec": result.codec,
        "storage": _report_value(result.storage),
        "trainable": result.trainable,
        "movement": result.movement,
        "load_targets": list(result.load_targets),
        "evaluation": _report_value(result.evaluation),
        "profile": _report_value(result.profile),
        "source": _report_value(result.source),
        "target": _report_value(result.target) if result.target is not None else None,
        "runtime_included": False,
    }
    return AnalysisResult(
        payload,
        kind="optimization_report",
        name="optimization",
        provenance={"adapter": "optimization_analysis"},
    )
