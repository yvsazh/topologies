"""Task-agnostic evaluation with explicit batch semantics."""
from topologies._execution import ModelCall

from .api import (
    EvaluationAdapter,
    EvaluationError,
    EvaluationRecord,
    EvaluationResult,
    MetricFactory,
    MetricValue,
    ModelAdapterBridge,
    compare_models,
    evaluate,
    metric,
    register_metric,
)
from .metrics import (
    BinnedAUROC,
    CalibrationMetrics,
    ConfusionMetric,
    ExactAUROC,
    RegressionMetric,
    TemperatureCalibrator,
)

for _name, _factory, _metadata in (
    ("mae", lambda: RegressionMetric("mae"), {"direction": "min", "unit": "target"}),
    ("mse", lambda: RegressionMetric("mse"), {"direction": "min", "unit": "target^2"}),
    ("rmse", lambda: RegressionMetric("rmse"), {"direction": "min", "unit": "target"}),
    ("auroc_exact", lambda **kw: ExactAUROC(**kw), {"direction": "max", "unit": "auc", "approximate": False}),
    ("auroc_binned", lambda **kw: BinnedAUROC(**kw), {"direction": "max", "unit": "auc-approx", "approximate": True}),
    ("calibration", lambda **kw: CalibrationMetrics(**kw), {"direction": "min", "unit": "mixed"}),
):
    try:
        register_metric(_name, _factory, metadata=_metadata)
    except ValueError:
        pass

__all__ = [
    "BinnedAUROC",
    "CalibrationMetrics",
    "ConfusionMetric",
    "EvaluationAdapter",
    "EvaluationError",
    "EvaluationRecord",
    "EvaluationResult",
    "ExactAUROC",
    "MetricFactory",
    "MetricValue",
    "ModelAdapterBridge",
    "ModelCall",
    "RegressionMetric",
    "TemperatureCalibrator",
    "compare_models",
    "evaluate",
    "metric",
    "register_metric",
]
