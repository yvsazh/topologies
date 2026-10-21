"""Small, dependency-free primitives for reusable, data-first analysis.

Analysis never fetches data or runs models implicitly. A plain callable receives
``data`` and, only when its signature declares a ``context`` parameter, context.
"""

from topologies.analysis.core import (
    Aggregate,
    Analysis,
    AnalysisContext,
    AnalysisResult,
    Derive,
    Filter,
    GroupBy,
    GroupedRecords,
    Pipeline,
    RecordSet,
    Select,
    Stage,
    analyze,
    register_serializer,
)
from topologies.analysis.engineering import (
    ComputeProviderRegistry,
    HardwareInfo,
    InferenceFeasibility,
    KVCacheSpec,
    ModelAnalysis,
    ModelCompute,
    ModelMemory,
    ModelStructure,
    TrainingFeasibility,
    analyze_hardware,
    analyze_inference,
    analyze_model,
    analyze_training,
)
from topologies.analysis.profiling import RuntimeProfile
from topologies.analysis.records import (
    compare,
    correlation,
    near_optimal,
    normalize,
    pareto_front,
)
from topologies.analysis.statistics import Fit, FitArtifact, fit_linear, fit_power_law

register_serializer(
    FitArtifact,
    "fit-artifact",
    lambda artifact: {key: getattr(artifact, key) for key in artifact.__dataclass_fields__},
    lambda values: FitArtifact(**values),
)

__all__ = [
    "Aggregate",
    "Analysis",
    "AnalysisContext",
    "AnalysisResult",
    "ComputeProviderRegistry",
    "DecodeProfile",
    "Derive",
    "Filter",
    "Fit",
    "FitArtifact",
    "GroupBy",
    "GroupedRecords",
    "HardwareInfo",
    "InferenceFeasibility",
    "KVCacheSpec",
    "ModelAnalysis",
    "ModelCompute",
    "ModelMemory",
    "ModelStructure",
    "Pipeline",
    "RecordSet",
    "RuntimeProfile",
    "Select",
    "Stage",
    "TrainingFeasibility",
    "analyze",
    "analyze_hardware",
    "analyze_inference",
    "analyze_model",
    "analyze_training",
    "compare",
    "correlation",
    "fit_linear",
    "fit_power_law",
    "near_optimal",
    "normalize",
    "pareto_front",
    "register_serializer",
]


def __getattr__(name: str):
    """Load measurement helpers on first use so the package stays import-light."""
    if name == "DecodeProfile":
        from topologies.analysis.decode import DecodeProfile
        return DecodeProfile
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
