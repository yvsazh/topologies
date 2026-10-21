"""Backend-neutral optimization contracts and stable diagnostics."""

import copy
from dataclasses import dataclass, field
from typing import Any

from topologies._execution import ModelCall  # noqa: F401 - re-exported


@dataclass(frozen=True)
class Target:
    device: str = "cpu"
    backend: str | None = None
    # Optional measured identity and declared workload envelope; mappings are
    # copied per instance so one target cannot mutate another's evidence.
    hardware_fingerprint: dict[str, Any] = field(default_factory=dict)
    workload_constraints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hardware_fingerprint", copy.deepcopy(self.hardware_fingerprint))
        object.__setattr__(self, "workload_constraints", copy.deepcopy(self.workload_constraints))


@dataclass
class CompatibilityReport:
    supported: list[str] = field(default_factory=list)
    unsupported: dict[str, str] = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)
    remedies: dict[str, str] = field(default_factory=dict)
    operator_mapping: dict[str, str] = field(default_factory=dict)
    boundaries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def compatible(self) -> bool:
        return not self.unsupported


@dataclass
class OptimizationResult:
    model: Any
    config: dict[str, Any]
    compatibility: CompatibilityReport
    backend: str
    validation: str = "not_run"
    history: list[dict[str, Any]] = field(default_factory=list)
    codec: str | None = None
    storage: dict[str, Any] = field(default_factory=dict)
    trainable: bool | None = None
    movement: str | None = None
    load_targets: tuple[str, ...] = ()
    runtime: Any = None
    evaluation: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    target: Target | None = None

    def __post_init__(self) -> None:
        if self.model is not None and self.runtime is None:
            self.runtime = self.model
        elif (
            self.model is None
            and self.runtime is not None
            and callable(getattr(self.runtime, "state_dict", None))
        ):
            # Keep module-backed legacy results meaningful through `.model`.
            self.model = self.runtime

    def summary(self) -> str:
        return f"backend={self.backend}; validation={self.validation}; converted={len(self.compatibility.supported)}"


class OptimizationError(RuntimeError):
    """Actionable failure with machine-readable, stable diagnostic context."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "unsupported_operation",
        backend: str | None = None,
        stage: str | None = None,
        module_path: str | None = None,
        requested: Any = None,
        remedies: tuple[str, ...] | list[str] = (),
    ):
        super().__init__(message)
        self.code, self.backend, self.stage = code, backend, stage
        self.module_path, self.requested = module_path, requested
        self.remedies = tuple(remedies)
