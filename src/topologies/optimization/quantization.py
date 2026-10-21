"""Explicit quantization facade with lazy TorchAO and isolated legacy dynamic INT8."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

from .contracts import CompatibilityReport, ModelCall, OptimizationError, OptimizationResult, Target


@dataclass(frozen=True)
class ModuleRule:
    """Module selector; exact path outranks glob path, which outranks module type."""

    path: str | None = None
    action: str = "quantize"
    pattern: str | None = None
    module_type: type | str | None = None

    def __post_init__(self):
        selectors = sum(value is not None for value in (self.path, self.pattern, self.module_type))
        if selectors != 1 or self.action not in ("quantize", "keep_float"):
            raise ValueError(
                "ModuleRule requires exactly one of path, pattern, module_type and action quantize|keep_float"
            )


@dataclass(frozen=True)
class QuantizationConfig:
    preset: str = "int8-weight-only"
    modules: tuple[str, ...] = ()
    rules: tuple[ModuleRule, ...] = ()
    backend: str | None = None
    shared_weights: str = "error"

    def __post_init__(self):
        if not isinstance(self.modules, tuple):
            object.__setattr__(self, "modules", tuple(self.modules))
        if not isinstance(self.rules, tuple):
            object.__setattr__(self, "rules", tuple(self.rules))
        signatures = [(r.path, r.pattern, str(r.module_type), r.action) for r in self.rules]
        if len(signatures) != len(set(signatures)):
            raise ValueError("Duplicate module rules are ambiguous")
        if self.shared_weights != "error":
            raise ValueError(
                "This implementation only supports shared_weights='error'; use a tying-aware handler"
            )


_BACKENDS: dict[str, Any] = {}


class BackendSession:
    """Guarded lifecycle adapter for plugins that expose a prepared session."""

    def __init__(self, backend: Any, session: Any):
        self.backend, self.session, self.state = backend, session, "prepared"

    @property
    def model(self):
        return self.session.model

    def calibrate(self, *args: Any, **kwargs: Any):
        if self.state != "prepared":
            raise OptimizationError(
                f"Calibration is invalid in session state {self.state!r}",
                code="invalid_stage_order",
                stage="calibrate",
            )
        method = getattr(self.session, "calibrate", None)
        if not callable(method):
            raise OptimizationError(
                "Backend session does not support calibration",
                code="invalid_stage_order",
                stage="calibrate",
            )
        value = method(*args, **kwargs)
        self.state = "calibrated"
        return value

    def convert(self, *args: Any, **kwargs: Any):
        if self.state not in ("prepared", "calibrated", "trained"):
            raise OptimizationError(
                f"Conversion is invalid in session state {self.state!r}",
                code="invalid_stage_order",
                stage="convert",
            )
        method = getattr(self.session, "convert", None)
        if not callable(method):
            raise OptimizationError(
                "Backend session does not support conversion",
                code="invalid_stage_order",
                stage="convert",
            )
        result = method(*args, **kwargs)
        self.state = "converted"
        return result


def register_quantization_backend(identifier: str, backend: Any) -> None:
    """Register an explicit backend implementing inspect and quantize methods."""
    if not identifier or identifier in _BACKENDS:
        raise ValueError(f"backend identifier empty or already registered: {identifier!r}")
    if not callable(getattr(backend, "inspect", None)) or not callable(
        getattr(backend, "quantize", None)
    ):
        raise TypeError(
            "backend must implement inspect(model, config, target) and quantize(model, config, target, **kwargs)"
        )
    _BACKENDS[identifier] = backend


def _config(value: QuantizationConfig | str) -> QuantizationConfig:
    return value if isinstance(value, QuantizationConfig) else QuantizationConfig(preset=value)


def _selected_rules(model: Any, cfg: QuantizationConfig):
    named = list(model.named_modules(remove_duplicate=False))
    paths = {name for name, _ in named}
    missing = set(cfg.modules) - paths
    for path in sorted(missing):
        raise OptimizationError(
            f"Selected module path {path!r} does not exist",
            code="unsupported_operation",
            stage="inspect",
            module_path=path,
            requested=cfg,
            remedies=("Correct the exact module path or remove the rule",),
        )
    matched_rules: dict[int, list[tuple[int, ModuleRule]]] = {i: [] for i in range(len(named))}
    for rule in cfg.rules:
        matched = False
        for index, (path, module) in enumerate(named):
            is_match = (
                path == rule.path
                if rule.path is not None
                else fnmatchcase(path, rule.pattern)
                if rule.pattern is not None
                else (
                    isinstance(module, rule.module_type)
                    if isinstance(rule.module_type, type)
                    else type(module).__name__ == rule.module_type
                    or f"{type(module).__module__}.{type(module).__qualname__}" == rule.module_type
                )
            )
            if is_match:
                matched = True
                priority = 3 if rule.path is not None else 2 if rule.pattern is not None else 1
                matched_rules[index].append((priority, rule))
        if not matched:
            selector = rule.path or rule.pattern or str(rule.module_type)
            raise OptimizationError(
                f"Module rule selector {selector!r} matched no modules",
                code="unmatched_selector",
                stage="inspect",
                requested=cfg,
            )
    chosen: dict[str, str] = {}
    selected_by_rules = any(r.action == "quantize" for r in cfg.rules)
    for index, (path, module) in enumerate(named):
        matches = matched_rules[index]
        if matches:
            priority = max(p for p, _ in matches)
            strongest = [r for p, r in matches if p == priority]
            actions = {r.action for r in strongest}
            if len(actions) > 1:
                raise OptimizationError(
                    f"Equally specific conflicting rules for module {path or '<root>'}",
                    code="invalid_configuration",
                    stage="inspect",
                    module_path=path,
                )
            chosen[path] = strongest[0].action
        elif path in cfg.modules:
            chosen[path] = "quantize"
        elif selected_by_rules:
            chosen[path] = "keep_float"
    if selected_by_rules or cfg.modules:
        included = {path for path, action in chosen.items() if action == "quantize"} | set(
            cfg.modules
        )
        excluded = {path for path, action in chosen.items() if action == "keep_float"}
    else:
        included, excluded = (
            set(),
            {path for path, action in chosen.items() if action == "keep_float"},
        )
    for path in cfg.modules:
        if chosen.get(path) == "keep_float":
            raise OptimizationError(
                f"Explicit module selection conflicts with keep_float rule for {path!r}",
                code="invalid_configuration",
                stage="inspect",
                module_path=path,
            )
    return named, included, excluded, chosen


def _validate_source_model(model: Any) -> None:
    import torch.nn.parameter as parameter
    import torch.nn.utils.parametrize as parametrize

    for path, module in model.named_modules(remove_duplicate=False):
        if parametrize.is_parametrized(module):
            raise OptimizationError(
                f"Parametrized module {path or '<root>'} is not supported by the initial quantization copier",
                code="unsupported_operation",
                stage="inspect",
                module_path=path,
            )
        for tensor_name, value in list(module.named_parameters(recurse=False)) + list(
            module.named_buffers(recurse=False)
        ):
            if isinstance(value, parameter.UninitializedParameter):
                raise OptimizationError(
                    f"Lazy tensor {path}.{tensor_name} must be materialized before quantization",
                    code="unsupported_operation",
                    stage="inspect",
                    module_path=path,
                )
            if value.device.type == "meta":
                raise OptimizationError(
                    f"Meta tensor {path}.{tensor_name} must be materialized before quantization",
                    code="unsupported_operation",
                    stage="inspect",
                    module_path=path,
                )


def _has_shared_parameters(model: Any, selected: set[str]) -> str | None:
    owners: dict[tuple[str, int], list[str]] = {}
    for module_path, module in model.named_modules(remove_duplicate=False):
        for name, parameter in module._parameters.items():
            if parameter is None or parameter.numel() == 0:
                continue
            try:
                key = (str(parameter.device), parameter.untyped_storage().data_ptr())
            except (RuntimeError, NotImplementedError):
                key = (str(parameter.device), id(parameter))
            owners.setdefault(key, []).append(f"{module_path}.{name}".strip("."))
    for paths in owners.values():
        module_paths = {path.rpartition(".")[0] for path in paths}
        if len(paths) > 1 and module_paths & selected:
            return ", ".join(sorted(paths))
    return None


def _backend_name(cfg: QuantizationConfig) -> str:
    if cfg.backend:
        return cfg.backend
    if cfg.preset == "dynamic-int8":
        return "pytorch-dynamic-legacy"
    return "torchao-qat" if cfg.preset == "int8-qat" else "torchao"


_BUILTIN_BACKENDS = {"torchao", "pytorch-dynamic-legacy", "torchao-pt2e"}


def _resolve_target(
    cfg: QuantizationConfig, target: Target | str
) -> tuple[Target, QuantizationConfig]:
    """Normalize the string convenience and honor explicit backend/workload constraints."""
    resolved = target if isinstance(target, Target) else Target(device=str(target))
    expected = (
        "pytorch-dynamic-legacy"
        if cfg.preset == "dynamic-int8"
        else "torchao-pt2e"
        if cfg.preset == "int8-static"
        else "torchao"
        if cfg.preset in ("int8-weight-only", "int8-dynamic", "int8-qat")
        else None
    )
    selected = cfg.backend
    if resolved.backend:
        if selected and selected != resolved.backend:
            raise OptimizationError(
                f"Target backend {resolved.backend!r} conflicts with configuration backend {selected!r}",
                code="invalid_configuration",
                backend=selected,
                stage="inspect",
                requested=resolved,
            )
        selected = resolved.backend
        if selected in _BUILTIN_BACKENDS and selected != expected:
            raise OptimizationError(
                f"Target backend {selected!r} conflicts with preset {cfg.preset!r} (requires {expected!r})",
                code="unsupported_target",
                backend=selected,
                stage="inspect",
                requested=resolved,
            )
        if selected not in _BUILTIN_BACKENDS and selected not in _BACKENDS:
            raise OptimizationError(
                f"Target backend {selected!r} is not registered",
                code="missing_backend",
                backend=selected,
                stage="inspect",
                requested=resolved,
            )
    elif selected and selected in _BUILTIN_BACKENDS and expected and selected != expected:
        raise OptimizationError(
            f"Backend {selected!r} conflicts with preset {cfg.preset!r}",
            code="invalid_configuration",
            backend=selected,
            stage="inspect",
            requested=resolved,
        )
    if selected != cfg.backend:
        cfg = QuantizationConfig(cfg.preset, cfg.modules, cfg.rules, selected, cfg.shared_weights)
    return resolved, cfg


def _validate_builtin_target_constraints(target: Target, backend: str) -> None:
    if target.workload_constraints:
        raise OptimizationError(
            "Built-in quantization does not implement workload/shape constraint validation",
            code="unsupported_target",
            backend=backend,
            stage="inspect",
            requested=target,
            remedies=("Use a registered provider backend that validates these constraints",),
        )


def _torchao_api():
    try:
        import torchao
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
    except Exception as exc:
        missing = isinstance(exc, (ImportError, AttributeError))
        detail = (
            (
                "The requested TorchAO optimization requires optional TorchAO; install the pinned compatible extra "
                "with `pip install 'topologies[optimization]'`. "
                if missing
                else f"TorchAO failed to initialize against this PyTorch build: {type(exc).__name__}: {exc}. "
            )
            + "The explicit `dynamic-int8` legacy preset is separate and uses PyTorch's eager backend."
        )
        raise OptimizationError(
            detail,
            code="missing_backend" if missing else "unsupported_version",
            backend="torchao",
            stage="prepare",
            requested="int8-weight-only",
            remedies=(
                "Install compatible TorchAO",
                "Use preset='dynamic-int8' only if activation-dynamic quantization is intended",
            ),
        ) from exc
    import torch

    torchao_version = str(getattr(torchao, "__version__", "unknown"))
    torch_version = str(torch.__version__).split("+", 1)[0]
    parse = lambda value: tuple(
        int(part) for part in re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value).groups() if part
    )
    try:
        tv, av = parse(torch_version), parse(torchao_version)
    except (AttributeError, TypeError, ValueError):
        raise OptimizationError(
            f"Cannot determine supported Torch/TorchAO versions (torch={torch_version}, torchao={torchao_version})",
            code="unsupported_version",
            backend="torchao",
            stage="prepare",
        )
    if tv < (2, 12) or tv >= (2, 13) or av != (0, 18, 0):
        raise OptimizationError(
            f"This tested weight-only backend matrix requires PyTorch 2.12.x and TorchAO 0.18.0; "
            f"found torch {torch_version}, torchao {torchao_version}.",
            code="unsupported_version",
            backend="torchao",
            stage="prepare",
            remedies=("Use the verified PyTorch 2.12.x + TorchAO 0.18.0 combination",),
        )
    return torchao, quantize_, Int8WeightOnlyConfig


def inspect_quantization(
    model: Any,
    config: QuantizationConfig | str = "int8-weight-only",
    *,
    target: Target | str = "cpu",
) -> CompatibilityReport:
    import torch.nn as nn

    provider_target = target
    cfg = _config(config)
    target, cfg = _resolve_target(cfg, target)
    if cfg.backend and cfg.backend not in ("torchao", "pytorch-dynamic-legacy", "torchao-pt2e"):
        plugin = _BACKENDS.get(cfg.backend)
        if plugin is None:
            missing = CompatibilityReport()
            missing.unsupported["backend"] = f"Backend {cfg.backend!r} is not registered"
            return missing
        return plugin.inspect(model, cfg, provider_target)
    tgt = target.device if isinstance(target, Target) else str(target)
    _validate_builtin_target_constraints(target, _backend_name(cfg))
    report = CompatibilityReport()
    if cfg.preset not in (
        "int8-weight-only",
        "dynamic-int8",
        "int8-dynamic",
        "int8-static",
        "int8-qat",
    ):
        report.unsupported["configuration"] = (
            f"Unsupported preset {cfg.preset!r}; available: int8-weight-only, int8-dynamic, int8-static, int8-qat, dynamic-int8 (legacy)."
        )
        return report
    if cfg.preset == "int8-qat":
        report.unsupported["lifecycle"] = (
            "int8-qat requires prepare_quantization(...), Trainer attachment, and explicit fit before convert"
        )
        return report
    expected_backend = (
        "torchao"
        if cfg.preset in ("int8-weight-only", "int8-dynamic")
        else "pytorch-dynamic-legacy"
        if cfg.preset == "dynamic-int8"
        else "torchao-pt2e"
        if cfg.preset == "int8-static"
        else "torchao"
    )
    if (
        cfg.backend in ("torchao", "pytorch-dynamic-legacy", "torchao-pt2e")
        and cfg.backend != expected_backend
    ):
        report.unsupported["backend"] = (
            f"Preset {cfg.preset!r} requires backend {expected_backend!r}, not {cfg.backend!r}."
        )
        return report
    if tgt != "cpu":
        report.unsupported["target"] = (
            f"{_backend_name(cfg)} currently supports CPU only, not {tgt!r}."
        )
        report.remedies["target"] = (
            "Select target='cpu' or register a backend tested for the requested device."
        )
        return report
    if any(
        tensor.device.type != "cpu" for tensor in list(model.parameters()) + list(model.buffers())
    ):
        report.unsupported["model_device"] = (
            "This initial backend is CPU-only and never migrates source tensors implicitly."
        )
        report.remedies["model_device"] = (
            "Reconstruct/copy the source model on CPU before quantization."
        )
        return report
    if cfg.preset in ("int8-weight-only", "int8-dynamic", "int8-static") and cfg.backend in (
        None,
        "torchao",
        "torchao-pt2e",
    ):
        try:
            _torchao_api()
        except OptimizationError as exc:
            report.unsupported["backend"] = str(exc)
            report.remedies["backend"] = (
                "Install the verified PyTorch 2.12.x + TorchAO 0.18.0 optional backend."
            )
    try:
        _validate_source_model(model)
        named, included, excluded, _chosen = _selected_rules(model, cfg)
    except OptimizationError as exc:
        report.unsupported[exc.module_path or "selection"] = str(exc)
        return report
    by_path = dict(named)
    eligible = (
        (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
        if cfg.preset == "int8-static"
        else (nn.Linear, nn.LSTM)
        if cfg.preset in ("dynamic-int8", "int8-dynamic")
        else nn.Linear
    )
    has_includes = bool(included)
    candidate_paths = included if has_includes else {n for n, m in named if isinstance(m, eligible)}
    candidate_paths -= excluded
    if cfg.preset == "int8-weight-only" and not has_includes:
        for path, module in named:
            if path not in excluded and isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                report.unsupported[path or "<root>"] = (
                    "Conv weight-only conversion is unsupported by the initial TorchAO preset; "
                    "select supported Linear scopes explicitly or use another backend."
                )
    for path in sorted(excluded):
        report.excluded[path or "<root>"] = "Explicit keep_float rule"
    if cfg.modules or cfg.rules:
        for path, module in named:
            if path in candidate_paths or path in excluded:
                continue
            if isinstance(module, (*eligible, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                report.excluded[path or "<root>"] = "Outside the explicit requested module scope"
            elif any(True for _ in module.parameters(recurse=False)):
                report.unknown.append(path or "<root>")
    for path in sorted(candidate_paths):
        module = by_path[path]
        if isinstance(module, eligible):
            report.supported.append(path or "<root>")
        elif cfg.preset == "int8-weight-only" and isinstance(
            module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)
        ):
            report.unsupported[path or "<root>"] = (
                "Conv weight-only conversion is unsupported by the initial TorchAO preset; "
                "select an explicitly supported Linear scope or use another backend."
            )
        else:
            report.unsupported[path or "<root>"] = (
                "Conv weight-only conversion is unsupported: TorchAO int8 weight-only currently selects Linear modules only."
                if cfg.preset == "int8-weight-only"
                else "Legacy dynamic INT8 supports Linear and LSTM modules only."
                if cfg.preset == "dynamic-int8"
                else "The selected TorchAO mode does not support this module scope."
            )
    if not report.supported and not report.unsupported:
        report.unsupported["model"] = (
            "No selected supported modules; refusing a no-op optimization."
        )
    shared = _has_shared_parameters(model, {p for p in candidate_paths if p in by_path})
    if shared:
        report.unsupported["shared_weights"] = (
            f"Selected parameter is shared across module paths: {shared}."
        )
        report.remedies["shared_weights"] = (
            "Apply a consistent supported policy to all aliases or provide a model factory/handler that preserves tying."
        )
    return report


def _alias_groups(model: Any):
    groups: dict[int, list[str]] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(param), []).append(name)
    return sorted(sorted(names) for names in groups.values() if len(names) > 1)


def _storage_alias_groups(model: Any):
    groups: dict[tuple[str, int], list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if parameter.numel() == 0:
            continue
        try:
            key = (str(parameter.device), parameter.untyped_storage().data_ptr())
        except (RuntimeError, NotImplementedError):
            key = (str(parameter.device), id(parameter))
        groups.setdefault(key, []).append(name)
    return sorted(sorted(names) for names in groups.values() if len(names) > 1)


def _clone(model: Any, inplace: bool, model_factory: Any):
    import torch

    if inplace:
        return model
    try:
        _validate_source_model(model)
        candidate = model_factory() if model_factory is not None else copy.deepcopy(model)
        source_modules = dict(model.named_modules(remove_duplicate=False))
        candidate_modules = dict(candidate.named_modules(remove_duplicate=False))
        source_shape = [(name, type(module)) for name, module in source_modules.items()]
        candidate_shape = [(name, type(module)) for name, module in candidate_modules.items()]
        if source_shape != candidate_shape:
            raise ValueError("factory produced a different named module topology")
        source_state = model.state_dict()
        candidate_state = candidate.state_dict()
        if set(source_state) != set(candidate_state):
            raise ValueError("factory state_dict keys differ from source")
        for key in source_state:
            src, dst = source_state[key], candidate_state[key]
            if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor):
                raise TypeError(f"non-tensor state at {key!r} requires a custom cloning factory")
            if src.shape != dst.shape or src.dtype != dst.dtype or src.device != dst.device:
                raise ValueError(f"factory state shape/dtype/device mismatch at {key!r}")
        if _alias_groups(model) != _alias_groups(candidate) or _storage_alias_groups(
            model
        ) != _storage_alias_groups(candidate):
            raise ValueError("factory does not preserve source tied-parameter topology")
        candidate.load_state_dict(source_state, strict=True)
        for name, src_module in source_modules.items():
            candidate_modules[name].training = src_module.training
        if any(
            candidate_modules[name].training != module.training
            for name, module in source_modules.items()
        ):
            raise ValueError("candidate mode flags differ from source")
        candidate_params = dict(candidate.named_parameters(remove_duplicate=False))
        for name, src_param in model.named_parameters(remove_duplicate=False):
            candidate_param = candidate_params[name]
            candidate_param.requires_grad_(src_param.requires_grad)
        if candidate is model or any(
            candidate_modules[name] is module for name, module in source_modules.items()
        ):
            raise ValueError("candidate shares module identity with source")
        source_params = {id(p) for p in model.parameters()}
        if any(id(p) in source_params for p in candidate.parameters()):
            raise ValueError("candidate aliases a source Parameter")
        source_storages = set()
        for tensor in list(model.parameters()) + list(model.buffers()):
            try:
                source_storages.add((tensor.device.type, tensor.untyped_storage().data_ptr()))
            except (RuntimeError, NotImplementedError):
                pass
        for tensor in list(candidate.parameters()) + list(candidate.buffers()):
            try:
                key = (tensor.device.type, tensor.untyped_storage().data_ptr())
                if key in source_storages:
                    raise ValueError("candidate tensor storage aliases source storage")
            except (RuntimeError, NotImplementedError):
                pass
        return candidate
    except Exception as exc:
        raise OptimizationError(
            f"Could not safely clone model: {exc}",
            code="unsafe_clone",
            stage="prepare",
            remedies=(
                "Supply model_factory=...",
                "Opt in to inplace=True only with exclusive ownership",
            ),
        ) from exc


def _quantize_torchao(model: Any, paths: list[str], config: QuantizationConfig):
    torchao, quantize_, config_type = _torchao_api()
    qconfig = config_type()
    try:
        modules = dict(model.named_modules())
        for path in paths:
            quantize_(modules[path], qconfig)
    except Exception as exc:
        raise OptimizationError(
            f"TorchAO int8 weight-only conversion failed: {exc}",
            code="unsupported_operation",
            backend="torchao",
            stage="convert",
            module_path=path if paths else None,
            requested=config,
            remedies=("Inspect backend/kernel support for this module and shape",),
        ) from exc
    return f"torchao-{getattr(torchao, '__version__', 'unknown')}"


def _quantize_torchao_dynamic(model: Any, paths: list[str], config: QuantizationConfig):
    _torchao_api()
    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

        mapping = dict(model.named_modules())
        for path in paths:
            quantize_(mapping[path], Int8DynamicActivationInt8WeightConfig())
    except Exception as exc:
        raise OptimizationError(
            f"TorchAO dynamic activation INT8 failed: {exc}",
            code="unsupported_operation",
            backend="torchao",
            stage="convert",
            requested=config,
            remedies=("Use eligible Linear modules on verified CPU backend",),
        ) from exc
    import torchao

    changed = any(
        type(getattr(module, "weight", None)).__module__.startswith("torchao")
        for module in model.modules()
    )
    if not changed:
        raise OptimizationError(
            "TorchAO dynamic backend changed no selected weights; refusing no-op success",
            code="unsupported_operation",
            backend="torchao",
            stage="convert",
        )
    return f"torchao-{torchao.__version__}"


def _quantize_legacy(model: Any, paths: list[str], config: QuantizationConfig):
    import torch
    import torch.nn as nn

    from . import _torch_ao

    _torch_ao.eager_api()
    engines = list(torch.backends.quantized.supported_engines)
    if not engines:
        raise OptimizationError(
            "This PyTorch build has no quantized CPU engine for dynamic INT8.",
            code="unsupported_target",
            backend="pytorch-dynamic-legacy",
            stage="prepare",
        )
    if paths == [""]:
        wrapper = nn.Sequential(model)
        _torch_ao.quantize_dynamic(
            wrapper, qconfig_spec={nn.Linear, nn.LSTM}, dtype=torch.qint8, inplace=True
        )
        converted = wrapper[0]
        if "quantized" not in type(converted).__module__:
            raise OptimizationError(
                "Legacy backend did not quantize root module",
                backend="pytorch-dynamic-legacy",
                stage="convert",
            )
        return converted
    mapping: Any = {nn.Linear, nn.LSTM} if not config.modules and not config.rules else set(paths)
    prior = torch.backends.quantized.engine
    engine = prior if prior in engines else engines[0]
    try:
        torch.backends.quantized.engine = engine
        _torch_ao.quantize_dynamic(model, qconfig_spec=mapping, dtype=torch.qint8, inplace=True)
    except Exception as exc:
        raise OptimizationError(
            f"Legacy dynamic INT8 conversion failed: {exc}",
            backend="pytorch-dynamic-legacy",
            stage="convert",
            requested=config,
        ) from exc
    finally:
        if prior in engines:
            torch.backends.quantized.engine = prior
    if not any("quantized" in type(m).__module__ for m in model.modules()):
        raise OptimizationError(
            "Legacy backend converted no modules; refusing no-op success.",
            backend="pytorch-dynamic-legacy",
        )
    return model


def quantize(
    model: Any,
    preset: QuantizationConfig | str = "int8-weight-only",
    *,
    target: Target | str = "cpu",
    examples: list[ModelCall | Any] | None = None,
    inplace: bool = False,
    model_factory: Any = None,
    backend: Any = None,
    modules: tuple[str, ...] = (),
    calibration: Any = None,
    calibration_adapter: Any = None,
    max_calibration_batches: int = 128,
    valid_mask: Any = None,
    calibration_id: str | None = None,
    capture_adapter: Any = None,
) -> OptimizationResult:
    """Transform a supported model; weight-only is TorchAO, dynamic INT8 is explicit legacy."""
    import torch

    provider_target = target
    cfg = _config(preset)
    if cfg.preset == "int8-qat":
        raise OptimizationError(
            "int8-qat is a prepared training lifecycle, not one-shot quantize(); "
            "use prepare_quantization(model, 'int8-qat', examples=...)",
            code="invalid_stage_order",
            backend="torchao-qat",
            stage="prepare",
        )
    if modules:
        cfg = QuantizationConfig(
            cfg.preset, tuple(modules), cfg.rules, cfg.backend, cfg.shared_weights
        )
    target, cfg = _resolve_target(cfg, target)
    tgt = target.device
    backend_id = cfg.backend or ("custom" if backend is not None else _backend_name(cfg))
    if backend is None and cfg.backend not in (
        None,
        "torchao",
        "pytorch-dynamic-legacy",
        "torchao-pt2e",
    ):
        backend = _BACKENDS.get(cfg.backend)
        if backend is None:
            raise OptimizationError(
                f"Backend {cfg.backend!r} is not registered",
                code="missing_backend",
                backend=cfg.backend,
                stage="inspect",
            )
    if backend is not None:
        backend_target = provider_target if isinstance(provider_target, Target) else tgt
        report = backend.inspect(model, cfg, backend_target)
        if not report.compatible:
            raise OptimizationError(
                f"Backend {backend_id!r} rejected model: {report.unsupported}",
                backend=backend_id,
                stage="inspect",
            )
        candidate = _clone(model, inplace, model_factory)
        result = backend.quantize(candidate, cfg, backend_target, examples=examples)
        if isinstance(result, OptimizationResult) and result.target is None:
            result.target = target
        return result
    if cfg.preset == "int8-static":
        if cfg.modules or cfg.rules:
            raise OptimizationError(
                "Initial PT2E preset does not yet apply per-module selectors; "
                "supply a quantizer that annotates the requested graph scopes explicitly",
                code="unsupported_operation",
                backend="torchao-pt2e",
                stage="inspect",
            )
        if calibration is None:
            raise OptimizationError(
                "int8-static requires explicit calibration=...; no data is inferred",
                code="invalid_calibration_data",
                backend="torchao-pt2e",
                stage="calibrate",
            )
        from .torchao_pt2e import StaticPTQConfig

        session = prepare_quantization(
            model,
            StaticPTQConfig(
                max_calibration_batches=max_calibration_batches,
                capture_adapter=capture_adapter,
                calibration_id=calibration_id,
                backend_id=cfg.backend,
            ),
            target=target,
            examples=examples or [],
            model_factory=model_factory,
        )
        session.calibrate(
            calibration,
            adapter=calibration_adapter,
            max_batches=max_calibration_batches,
            valid_mask=valid_mask,
            source_id=calibration_id,
        )
        result = session.convert()
        if examples:
            with torch.inference_mode():
                for example in examples:
                    call = example if isinstance(example, ModelCall) else ModelCall((example,))
                    result.model(*call.args, **call.kwargs)
            result.validation = f"passed ({len(examples)} calls)"
        return result
    if tgt != "cpu":
        raise OptimizationError(
            f"{backend_id} currently supports CPU only, not {tgt!r}",
            code="unsupported_target",
            backend=backend_id,
            stage="inspect",
            requested=cfg,
            remedies=("Select target='cpu'",),
        )
    _validate_builtin_target_constraints(target, backend_id)
    report = inspect_quantization(model, cfg, target=target)
    if not report.compatible:
        path = next(
            (name for name in report.unsupported if name not in ("backend", "target")),
            next(iter(report.unsupported)),
        )
        reason = report.unsupported[path]
        if path == "backend" and cfg.preset in ("int8-weight-only", "int8-dynamic", "int8-static"):
            _torchao_api()
        raise OptimizationError(
            f"{path}: {reason}",
            code="unsupported_operation",
            backend=backend_id,
            stage="inspect",
            module_path=path,
            requested=cfg,
            remedies=(
                report.remedies.get(path, "Check preset, scope, target, and backend support"),
            ),
        )
    candidate = _clone(model, inplace, model_factory)
    paths = ["" if p == "<root>" else p for p in report.supported]
    if cfg.preset == "int8-weight-only":
        backend_id = _quantize_torchao(candidate, paths, cfg)
    elif cfg.preset == "int8-dynamic":
        backend_id = _quantize_torchao_dynamic(candidate, paths, cfg)
    else:
        candidate = _quantize_legacy(candidate, paths, cfg)
        backend_id = "pytorch-dynamic-legacy"
    validation = "not_run"
    if examples is not None:
        if not examples:
            raise OptimizationError(
                "examples was supplied but empty", code="missing_examples", stage="validate"
            )
        from topologies._execution import model_execution

        try:
            with model_execution(candidate):
                for example in examples:
                    call = example if isinstance(example, ModelCall) else ModelCall((example,))
                    candidate(*call.args, **call.kwargs)
            validation = f"passed ({len(examples)} calls)"
        except Exception as exc:
            raise OptimizationError(
                f"Representative execution validation failed: {exc}",
                code="unsupported_operation",
                backend=backend_id,
                stage="validate",
                requested=cfg,
                remedies=("Provide representative inputs matching the supported module shapes",),
            ) from exc
    report.supported = list(paths)
    normalized_rules = [
        {
            "path": r.path,
            "pattern": r.pattern,
            "module_type": (
                f"{r.module_type.__module__}.{r.module_type.__qualname__}"
                if isinstance(r.module_type, type)
                else r.module_type
            ),
            "action": r.action,
        }
        for r in cfg.rules
    ]
    resolved = {
        "preset": cfg.preset,
        "target": tgt,
        "modules": paths,
        "rules": normalized_rules,
        "backend": backend_id,
        "shared_weights": cfg.shared_weights,
    }
    return OptimizationResult(
        candidate,
        resolved,
        report,
        backend_id,
        validation,
        codec=f"{backend_id}-module-v1",
        storage=_storage_metadata(candidate),
        trainable=any(p.requires_grad for p in candidate.parameters()),
        movement="CPU backend; device movement unsupported",
        load_targets=(tgt,),
        target=target,
    )


def _storage_metadata(model: Any) -> dict[str, Any]:
    """Report tensor backing storage exactly where the runtime exposes it."""
    import torch

    seen: set[tuple[str, int]] = set()
    total, unknown = 0, []
    backing_by_state: dict[str, int | None] = {}
    for root_path, root_value in model.state_dict().items():
        pending = [(root_path, root_value)]
        local: set[tuple[str, int]] = set()
        local_bytes, local_unknown = 0, []
        while pending:
            path, value = pending.pop()
            try:
                storage = value.untyped_storage()
                pointer = storage.data_ptr()
                key = (str(value.device), pointer)
                if key not in local:
                    local.add(key)
                    local_bytes += storage.nbytes()
                if key not in seen:
                    seen.add(key)
                    total += storage.nbytes()
            except (AttributeError, RuntimeError, NotImplementedError) as exc:
                # TorchAO subclasses expose actual backing tensors through flattening fields.
                children = [
                    (f"{path}.{name}", child)
                    for name, child in getattr(value, "__dict__", {}).items()
                    if isinstance(child, torch.Tensor)
                ]
                if children:
                    pending.extend(children)
                else:
                    local_unknown.append(f"{path}: {type(exc).__name__}")
        backing_by_state[root_path] = None if local_unknown else local_bytes
        unknown.extend(local_unknown)
    return {
        "tensor_backing_bytes": total if not unknown else None,
        "known": not unknown,
        "unknown_state": unknown,
        "backing_bytes_by_state": backing_by_state,
        "resident_process_bytes": None,
        "serialized_bytes": None,
    }


@dataclass
class QuantizationPlan:
    """Static, side-effect-free compatibility plan with explicit call validation."""

    model: Any
    config: QuantizationConfig
    target: Target | str
    compatibility: CompatibilityReport
    backend: str

    @property
    def resolved_config(self) -> dict[str, Any]:
        target = self.target.device if isinstance(self.target, Target) else str(self.target)
        rules = [
            {
                "path": r.path,
                "pattern": r.pattern,
                "module_type": (
                    f"{r.module_type.__module__}.{r.module_type.__qualname__}"
                    if isinstance(r.module_type, type)
                    else r.module_type
                ),
                "action": r.action,
            }
            for r in self.config.rules
        ]
        return {
            "preset": self.config.preset,
            "backend": self.backend,
            "target": target,
            "selected_modules": list(self.compatibility.supported),
            "excluded": dict(self.compatibility.excluded),
            "rules": rules,
            "shared_weights": self.config.shared_weights,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.resolved_config,
            "target": self.target.device if isinstance(self.target, Target) else str(self.target),
            "backend": self.backend,
            "supported": list(self.compatibility.supported),
            "excluded": dict(self.compatibility.excluded),
            "unsupported": dict(self.compatibility.unsupported),
            "unknown": list(self.compatibility.unknown),
            "validation": "requires explicit calls",
            "trainability": "not changed by inspection",
            "movement": "backend target capability; transformation not performed",
        }

    def validate(self, examples: list[ModelCall | Any]) -> dict[str, Any]:
        if not examples:
            raise OptimizationError(
                "Plan validation requires at least one representative call",
                code="missing_examples",
                stage="validate",
            )
        from topologies._execution import model_execution

        try:
            with model_execution(self.model):
                for example in examples:
                    call = example if isinstance(example, ModelCall) else ModelCall((example,))
                    self.model(*call.args, **call.kwargs)
        except Exception as exc:
            raise OptimizationError(
                f"Plan execution validation failed: {exc}",
                code="unsupported_operation",
                backend=self.backend,
                stage="validate",
                requested=self.config,
            ) from exc
        return {
            "status": "passed",
            "calls": len(examples),
            "claim": "observed calls only; unseen branches and shapes remain unvalidated",
        }


def plan_quantization(
    model: Any,
    config: QuantizationConfig | str = "int8-weight-only",
    *,
    target: Target | str = "cpu",
) -> QuantizationPlan:
    """Inspect compatibility without calling model.forward or importing backend kernels."""
    cfg = _config(config)
    report = inspect_quantization(model, cfg, target=target)
    return QuantizationPlan(model, cfg, target, report, cfg.backend or _backend_name(cfg))


def output_diagnostics(
    baseline: Any, candidate: Any, examples: list[ModelCall | Any]
) -> dict[str, Any]:
    """Explicit paired-call numerical distance; does not infer task quality."""
    import torch

    if not examples:
        raise ValueError("examples must contain at least one call")
    from topologies._execution import model_execution

    left, right = [], []
    with model_execution(baseline), model_execution(candidate):
        for example in examples:
            call = example if isinstance(example, ModelCall) else ModelCall((example,))
            a, b = baseline(*call.args, **call.kwargs), candidate(*call.args, **call.kwargs)
            if (
                not isinstance(a, torch.Tensor)
                or not isinstance(b, torch.Tensor)
                or a.shape != b.shape
            ):
                raise OptimizationError(
                    "Output diagnostic requires matching tensor outputs/shapes",
                    code="metric_input_mismatch",
                    stage="compare",
                )
            left.append(a.detach().float().reshape(-1))
            right.append(b.detach().float().reshape(-1))
    a, b = torch.cat(left), torch.cat(right)
    delta = (a - b).abs()
    denominator = torch.linalg.vector_norm(a).item()
    return {
        "calls": len(examples),
        "elements": a.numel(),
        "max_absolute_error": delta.max().item(),
        "mean_absolute_error": delta.mean().item(),
        "relative_l2_error": torch.linalg.vector_norm(a - b).item() / denominator
        if denominator
        else None,
        "task_quality": "not measured",
    }


def prepare_quantization(
    model: Any,
    config: Any = None,
    *,
    target: Any = "cpu",
    backend: Any = None,
    backend_id: str | None = None,
    examples: list[ModelCall | Any] | None = None,
    model_factory: Any = None,
    **kwargs: Any,
) -> BackendSession:
    """Prepare explicit PTQ/QAT topology before optimizer construction, or delegate to a plugin."""
    import torch

    from .torchao_pt2e import QATConfig, StaticPTQConfig, prepare_pt2e_session

    mode = None
    if isinstance(config, QATConfig):
        mode = "qat"
    elif isinstance(config, StaticPTQConfig):
        mode = "ptq"
    elif isinstance(config, QuantizationConfig) and config.preset == "int8-static":
        mode = "ptq"
        if config.modules or config.rules:
            raise OptimizationError(
                "Initial PT2E preset does not yet apply per-module selectors; "
                "supply a quantizer that annotates the requested graph scopes explicitly",
                code="unsupported_operation",
                backend="torchao-pt2e",
                stage="inspect",
            )
        config = StaticPTQConfig(capture_adapter=kwargs.get("capture_adapter"))
    elif isinstance(config, QuantizationConfig) and config.preset == "int8-qat":
        if config.modules or config.rules:
            raise OptimizationError(
                "Native int8-qat is Linear-scoped by its verified Intx QAT handler; "
                "custom per-module PT2E QAT requires a fake-quant quantizer",
                code="unsupported_operation",
                backend="torchao-qat",
                stage="inspect",
            )
        _torchao_api()
        from torchao.quantization import IntxWeightOnlyConfig

        mode, config = (
            "qat",
            QATConfig(
                base_config=IntxWeightOnlyConfig(weight_dtype=torch.int8), schedule_id="int8-qat-v1"
            ),
        )
    elif config == "int8-qat":
        _torchao_api()
        from torchao.quantization import IntxWeightOnlyConfig

        base_config = kwargs.pop("base_config", None)
        if base_config is None:
            base_config = IntxWeightOnlyConfig(weight_dtype=torch.int8)
        mode, config = (
            "qat",
            QATConfig(
                base_config=base_config, schedule_id=kwargs.pop("schedule_id", "int8-qat-v1")
            ),
        )
    elif config == "qat":
        mode, config = "qat", QATConfig()
    elif config == "int8-static":
        mode, config = "ptq", StaticPTQConfig()
    if mode is not None:
        tgt = target.device if isinstance(target, Target) else str(target)
        # Do not silently conflate TorchAO's eager Int8WeightOnlyConfig with a PT2E QAT graph.
        if (
            mode == "qat"
            and config.base_config is not None
            and type(config.base_config).__name__ == "Int8WeightOnlyConfig"
        ):
            raise OptimizationError(
                "Int8WeightOnlyConfig has no supported QAT mapping in this PT2E backend; "
                "use QATConfig(quantizer=...) with a tested PT2E quantizer",
                code="unsupported_operation",
                backend="torchao-pt2e",
                stage="prepare",
            )
        cfg = QuantizationConfig(preset="int8-static")
        report = inspect_quantization(model, cfg, target=target)
        if not report.compatible:
            path, reason = next(iter(report.unsupported.items()))
            raise OptimizationError(f"{path}: {reason}", stage="inspect", backend="torchao-pt2e")
        if not examples:
            raise OptimizationError(
                "PT2E prepare requires representative example calls for explicit graph capture",
                code="missing_examples",
                backend="torchao-pt2e",
                stage="prepare",
            )
        normalized = [x if isinstance(x, ModelCall) else ModelCall((x,)) for x in examples]
        candidate = _clone(model, False, model_factory)
        return prepare_pt2e_session(candidate, config, target=tgt, examples=normalized, mode=mode)
    selected = backend
    identifier = backend_id or "custom"
    if selected is None and backend_id is not None:
        selected = _BACKENDS.get(backend_id)
    if selected is None or not callable(getattr(selected, "prepare", None)):
        raise OptimizationError(
            "No registered backend preparation/session hook is available; "
            "QAT is not part of the built-in weight-only slice.",
            code="unsupported_operation",
            backend=identifier,
            stage="prepare",
        )
    report = selected.inspect(model, config, target)
    if not report.compatible:
        raise OptimizationError(
            f"Backend {identifier!r} rejected model: {report.unsupported}",
            backend=identifier,
            stage="inspect",
        )
    try:
        session = selected.prepare(model, config, target, **kwargs)
    except Exception as exc:
        raise OptimizationError(
            f"Backend {identifier!r} preparation failed: {exc}",
            backend=identifier,
            stage="prepare",
            requested=config,
        ) from exc
    return BackendSession(selected, session)


def quantize_with_backend(
    identifier: str, model: Any, config: Any, *, target: Any = "cpu", **kwargs: Any
) -> Any:
    try:
        backend = _BACKENDS[identifier]
    except KeyError as exc:
        raise OptimizationError(
            f"No local quantization backend {identifier!r} is registered",
            backend=identifier,
            code="missing_backend",
        ) from exc
    cfg = _config(config)
    cfg = QuantizationConfig(cfg.preset, cfg.modules, cfg.rules, identifier, cfg.shared_weights)
    return quantize(model, cfg, target=target, backend=backend, **kwargs)
