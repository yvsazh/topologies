"""What happens inside a network when it processes an input?

``topologies.introspection`` captures inputs/outputs/gradients at any module,
functional op or parameter through scoped hooks, summarises activations over
datasets with streaming reducers, and runs techniques (attribution, attention,
ablation, patching, probing, similarity, lenses, process trajectories ...).
Every technique returns an ``AnalysisResult``; rendering is always explicit.

    import topologies
    insp = topologies.introspect(model_or_path)
    trace = insp.trace(x)              # summaries of every module output
    print(trace.summary())

Exports are loaded lazily; importing this package never imports Matplotlib.
"""
from __future__ import annotations

import importlib
from typing import Any

_LAZY: dict[str, tuple[str, str]] = {
    "introspect": ("topologies.introspection.api", "introspect"),
    "Inspector": ("topologies.introspection.api", "Inspector"),
    "trace": ("topologies.introspection.api", "trace"),
    "compare": ("topologies.introspection.techniques.similarity", "compare"),
    "Introspect": ("topologies.introspection.integrations", "Introspect"),
    "IntrospectionCallback": ("topologies.introspection.integrations", "IntrospectionCallback"),
    "Session": ("topologies.introspection.session", "Session"),
    "Trace": ("topologies.introspection.trace", "Trace"),
    "SiteRecord": ("topologies.introspection.trace", "SiteRecord"),
    "CallTree": ("topologies.introspection.trace", "CallTree"),
    "Collection": ("topologies.introspection.collection", "Collection"),
    "Report": ("topologies.introspection.report", "Report"),
    "Finding": ("topologies.introspection.report", "Finding"),
    "CapturePolicy": ("topologies.introspection.capture", "CapturePolicy"),
    "Axes": ("topologies.introspection.semantics", "Axes"),
    "SemanticSource": ("topologies.introspection.semantics", "SemanticSource"),
    "Unit": ("topologies.introspection.semantics", "Unit"),
    "Select": ("topologies.introspection.sites", "Select"),
    "SiteIndex": ("topologies.introspection.sites", "SiteIndex"),
    "Adapter": ("topologies.introspection.adapters", "Adapter"),
    "AttentionSite": ("topologies.introspection.adapters", "AttentionSite"),
    "ProcessSpec": ("topologies.introspection.adapters", "ProcessSpec"),
    "register_adapter": ("topologies.introspection.adapters", "register_adapter"),
    "Technique": ("topologies.introspection.techniques", "Technique"),
    "register_technique": ("topologies.introspection.techniques", "register_technique"),
    "unregister_technique": ("topologies.introspection.techniques", "unregister_technique"),
    "register_reducer": ("topologies.introspection.reducers", "register_reducer"),
    "Reducer": ("topologies.introspection.reducers", "Reducer"),
    "topk": ("topologies.introspection.reducers", "topk"),
    "Ablate": ("topologies.introspection.interventions", "Ablate"),
    "Patch": ("topologies.introspection.interventions", "Patch"),
    "Steer": ("topologies.introspection.interventions", "Steer"),
    "Scale": ("topologies.introspection.interventions", "Scale"),
    "Clamp": ("topologies.introspection.interventions", "Clamp"),
    "Replace": ("topologies.introspection.interventions", "Replace"),
    "HeadAblate": ("topologies.introspection.interventions", "HeadAblate"),
    "Objective": ("topologies.introspection.objectives", "Objective"),
    "Output": ("topologies.introspection.objectives", "Output"),
    "Logit": ("topologies.introspection.objectives", "Logit"),
    "LogitDiff": ("topologies.introspection.objectives", "LogitDiff"),
    "Loss": ("topologies.introspection.objectives", "Loss"),
    "SiteNorm": ("topologies.introspection.objectives", "SiteNorm"),
    "IntrospectionError": ("topologies.introspection.errors", "IntrospectionError"),
    "SemanticsRequired": ("topologies.introspection.errors", "SemanticsRequired"),
    "CaptureBudgetExceeded": ("topologies.introspection.errors", "CaptureBudgetExceeded"),
    "UnsupportedModel": ("topologies.introspection.errors", "UnsupportedModel"),
    "ObjectiveRequired": ("topologies.introspection.errors", "ObjectiveRequired"),
}

__all__ = sorted(_LAZY)  # noqa: PLE0605


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'topologies.introspection' has no attribute {name!r}")
    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
