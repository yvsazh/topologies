"""Construction toolkit: presets, parameter budgets, validation, config capture.

Works for library stacks (``topologies.nn``), catalog models
(``topologies.models``) and your own ``nn.Module`` classes alike.

The package itself is callable, so both spellings work:

    from topologies import build
    model = build(TransformerLM, "small", vocab_size=tok.vocab_size)
    from topologies.build import build, presets, size_table
"""

from __future__ import annotations

import sys
import types
from typing import Any

from topologies.build.api import build, resolve_architecture, size_table
from topologies.build.audit import audit_constructor
from topologies.build.budget import BudgetResult, fit_budget, measure_params
from topologies.build.config import (
    Configurable,
    class_path,
    import_path,
    init_config,
    set_init_config,
)
from topologies.build.params import (
    count_params,
    count_params_as,
    embedding_params,
    format_params,
    parse_param_count,
)
from topologies.build.presets import (
    budget_knobs,
    get_budget_knobs,
    get_presets,
    presets,
    register_budget_knobs,
    register_presets,
)
from topologies.build.validation import (
    ConfigError,
    check_choice,
    check_divisible,
    check_non_negative,
    check_positive,
    check_range,
)

__all__ = [
    "BudgetResult",
    "ConfigError",
    "Configurable",
    "audit_constructor",
    "budget_knobs",
    "build",
    "check_choice",
    "check_divisible",
    "check_non_negative",
    "check_positive",
    "check_range",
    "class_path",
    "count_params",
    "count_params_as",
    "embedding_params",
    "fit_budget",
    "format_params",
    "get_budget_knobs",
    "get_presets",
    "import_path",
    "init_config",
    "measure_params",
    "parse_param_count",
    "presets",
    "register_budget_knobs",
    "register_presets",
    "resolve_architecture",
    "set_init_config",
    "size_table",
]


class _CallableModule(types.ModuleType):
    """``topologies.build(...)`` == ``topologies.build.build(...)``."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return build(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
