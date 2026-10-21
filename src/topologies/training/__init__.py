"""Universal PyTorch training engine (standalone; works with any nn.Module)."""

from __future__ import annotations

# Materialized training artifacts enter the same public registries as user
# factories. Local built-ins retain their original names/defaults; artifact
# aliases register only after both registries are initialized.
from pathlib import Path as _Path

from topologies.training import recipes as _recipes  # noqa: F401  (registers recipes)
from topologies.training.adapters import ModelAdapter, ModelSpec
from topologies.training.callbacks import Callback
from topologies.training.components import load_training_components
from topologies.training.configs import (
    LoggingConfig,
    MonitoringConfig,
    OptimizerConfig,
    TrainerConfig,
)
from topologies.training.data import make_synthetic_loader
from topologies.training.engine import Trainer
from topologies.training.errors import TrainingError
from topologies.training.introspect import all_recipe_schemas, list_recipes, recipe_schema
from topologies.training.losses import list_losses, register_loss
from topologies.training.monitoring import BaseMonitor, StandardMonitor
from topologies.training.recipe_base import (
    EpochContext,
    EpochResult,
    FitContext,
    FitResult,
    Recipe,
    RecipeComponents,
    RecipeConfig,
    StepContext,
    StepResult,
    auto_recipe_for,
    get_recipe,
    register_recipe,
)
from topologies.training.registries import (
    list_optimizers,
    list_schedulers,
    register_optimizer,
    register_scheduler,
)
from topologies.training.reporting import TrainingReporter
from topologies.training.resolve import resolve_recipe

load_training_components(_Path(__file__).resolve().parents[1])

__all__ = [
    "BaseMonitor",
    "Callback",
    "EpochContext",
    "EpochResult",
    "FitContext",
    "FitResult",
    "LoggingConfig",
    "ModelAdapter",
    "ModelSpec",
    "MonitoringConfig",
    "OptimizerConfig",
    "Recipe",
    "RecipeComponents",
    "RecipeConfig",
    "StandardMonitor",
    "StepContext",
    "StepResult",
    "Trainer",
    "TrainerConfig",
    "TrainingError",
    "TrainingReporter",
    "all_recipe_schemas",
    "auto_recipe_for",
    "get_recipe",
    "list_losses",
    "list_optimizers",
    "list_recipes",
    "list_schedulers",
    "load_training_components",
    "make_synthetic_loader",
    "recipe_schema",
    "register_loss",
    "register_optimizer",
    "register_recipe",
    "register_scheduler",
    "resolve_recipe",
]
