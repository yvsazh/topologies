"""Composable model optimization primitives. Optional backends load on demand."""

from .artifacts import load_artifact, register_artifact_codec, save_artifact
from .contracts import CompatibilityReport, ModelCall, OptimizationError, OptimizationResult, Target
from .distillation import (
    DiskTeacherCacheBackend,
    DistillationRun,
    FeatureCapture,
    TeacherCache,
    distill,
)
from .integrations import (
    Constraint,
    ConstraintVerdict,
    check_constraint,
    exclusive_device,
    feasible_pareto,
    optimization_trial,
    trial_output,
)
from .pruning import (
    MaskController,
    StructuralPruningSpec,
    magnitude_prune,
    structural_prune_sequential,
)
from .quantization import (
    BackendSession,
    ModuleRule,
    QuantizationConfig,
    QuantizationPlan,
    inspect_quantization,
    output_diagnostics,
    plan_quantization,
    prepare_quantization,
    quantize,
    quantize_with_backend,
    register_quantization_backend,
)
from .recommendations import Recommendation, recommend, sensitivity_sweep
from .torchao_pt2e import QATConfig, StaticPTQConfig, TorchAOQuantizationSession
from .workflow import (
    FunctionalStage,
    OptimizationContext,
    OptimizationPipeline,
    TransformationStage,
)

__all__ = [
    "BackendSession",
    "CompatibilityReport",
    "Constraint",
    "ConstraintVerdict",
    "DiskTeacherCacheBackend",
    "DistillationRun",
    "FeatureCapture",
    "FunctionalStage",
    "MaskController",
    "ModelCall",
    "ModuleRule",
    "OptimizationContext",
    "OptimizationError",
    "OptimizationPipeline",
    "OptimizationResult",
    "QATConfig",
    "QuantizationConfig",
    "QuantizationPlan",
    "Recommendation",
    "StaticPTQConfig",
    "StructuralPruningSpec",
    "Target",
    "TeacherCache",
    "TorchAOQuantizationSession",
    "TransformationStage",
    "check_constraint",
    "distill",
    "exclusive_device",
    "feasible_pareto",
    "inspect_quantization",
    "load_artifact",
    "magnitude_prune",
    "optimization_trial",
    "output_diagnostics",
    "plan_quantization",
    "prepare_quantization",
    "quantize",
    "quantize_with_backend",
    "recommend",
    "register_artifact_codec",
    "register_quantization_backend",
    "save_artifact",
    "sensitivity_sweep",
    "structural_prune_sequential",
    "trial_output",
]
