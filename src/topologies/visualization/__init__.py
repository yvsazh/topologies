"""Producer-neutral visualization contracts and dispatch (stdlib-only on import).

Use ``create_visualizer`` to select a sink, then ``handle`` normalized live
events. Static results are adapted with ``adapt_result`` and passed to
``render``; they are never represented as synthetic events. Heavier views
(terminal dashboards, Matplotlib) are imported lazily on first attribute access.
"""

from importlib import import_module
from typing import Any

from .adapters import adapt_result, render_result, visualize
from .core import (
    DEFAULT_REGISTRY,
    DashboardVisualizer,
    EventKind,
    ExperimentRunStateAdapter,
    MetricSeries,
    PlotSpec,
    Progress,
    SessionInfo,
    StandardVisualizer,
    VisualizationEvent,
    VisualizationResult,
    Visualizer,
    VisualizerGroup,
    VisualizerRegistry,
    create_visualizer,
    normalize_event,
    register_visualizer,
)
from .matplotlib import MatplotlibVisualizer

_LAZY = {
    "LiveVisualizer": ("topologies.visualization.live", "LiveVisualizer"),
    "TrainingTracker": ("topologies.visualization.state", "TrainingTracker"),
    "ExperimentTracker": ("topologies.visualization.state", "ExperimentTracker"),
    "Series": ("topologies.visualization.state", "Series"),
    "Theme": ("topologies.visualization.theme", "Theme"),
    "PlainTrainingVisualizer": ("topologies.visualization.training_views",
                                "PlainTrainingVisualizer"),
    "StandardTrainingVisualizer": ("topologies.visualization.training_views",
                                   "StandardTrainingVisualizer"),
    "TrainingDashboard": ("topologies.visualization.training_views", "TrainingDashboard"),
    "ExperimentLinesVisualizer": ("topologies.visualization.experiment_views",
                                  "ExperimentLinesVisualizer"),
    "ExperimentLiveVisualizer": ("topologies.visualization.experiment_views",
                                 "ExperimentLiveVisualizer"),
    "ExperimentDashboard": ("topologies.visualization.experiment_views", "ExperimentDashboard"),
    "simulate_training": ("topologies.visualization.testing", "simulate_training"),
    "simulate_experiment": ("topologies.visualization.testing", "simulate_experiment"),
    "RecordingVisualizer": ("topologies.visualization.testing", "RecordingVisualizer"),
    "VirtualTerminal": ("topologies.visualization.testing", "VirtualTerminal"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'topologies.visualization' has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


__all__ = [
    "DEFAULT_REGISTRY",
    "DashboardVisualizer",
    "EventKind",
    "ExperimentDashboard",
    "ExperimentLinesVisualizer",
    "ExperimentLiveVisualizer",
    "ExperimentRunStateAdapter",
    "ExperimentTracker",
    "LiveVisualizer",
    "MatplotlibVisualizer",
    "MetricSeries",
    "PlainTrainingVisualizer",
    "PlotSpec",
    "Progress",
    "RecordingVisualizer",
    "Series",
    "SessionInfo",
    "StandardTrainingVisualizer",
    "StandardVisualizer",
    "Theme",
    "TrainingDashboard",
    "TrainingTracker",
    "VirtualTerminal",
    "VisualizationEvent",
    "VisualizationResult",
    "Visualizer",
    "VisualizerGroup",
    "VisualizerRegistry",
    "adapt_result",
    "create_visualizer",
    "normalize_event",
    "register_visualizer",
    "render_result",
    "simulate_experiment",
    "simulate_training",
    "visualize",
]
