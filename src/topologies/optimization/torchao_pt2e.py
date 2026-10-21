"""Narrow, explicitly target-scoped TorchAO PT2E PTQ/QAT sessions."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from topologies._execution import ModelCall, execution_rng

from .contracts import CompatibilityReport, OptimizationError, OptimizationResult


@dataclass(frozen=True)
class StaticPTQConfig:
    preset: str = "int8-static"
    quantizer: Any = None
    capture_adapter: Callable[[nn.Module, ModelCall], Any] | None = None
    max_calibration_batches: int = 128
    min_calibration_batches: int = 1
    calibration_id: str | None = None
    preprocessing_id: str | None = None
    backend_id: str | None = None

    def __post_init__(self):
        if self.preset != "int8-static":
            raise ValueError("StaticPTQConfig.preset must be 'int8-static'")
        if (
            self.max_calibration_batches < 1
            or not 1 <= self.min_calibration_batches <= self.max_calibration_batches
        ):
            raise ValueError(
                "calibration bounds require 1 <= min_calibration_batches <= max_calibration_batches"
            )


@dataclass(frozen=True)
class QATConfig:
    """TorchAO native QAT base policy or a custom fake-quant-capable PT2E quantizer."""

    base_config: Any = None
    quantizer: Any = None
    capture_adapter: Callable[[nn.Module, ModelCall], Any] | None = None
    schedule_id: str = "caller-scheduled-v1"

    def __post_init__(self):
        if self.base_config is not None and (
            self.quantizer is not None or self.capture_adapter is not None
        ):
            raise ValueError(
                "QATConfig base_config selects native TorchAO QAT; do not combine it with PT2E quantizer/capture_adapter"
            )


def _quantizer(supplied: Any):
    if supplied is not None:
        return supplied
    try:
        from torchao.quantization.pt2e.quantizer.arm_inductor_quantizer import (
            ArmInductorQuantizer,
            get_default_arm_inductor_quantization_config,
        )

        return ArmInductorQuantizer().set_global(get_default_arm_inductor_quantization_config())
    except Exception as exc:
        raise OptimizationError(
            f"No PT2E quantizer available: {exc}; supply quantizer=... for a tested target",
            code="missing_backend",
            backend="torchao-pt2e",
            stage="prepare",
        ) from exc


def _capture(model: nn.Module, call: ModelCall, adapter: Any):
    if adapter is not None:
        graph = adapter(model, call)
        if not isinstance(graph, nn.Module):
            raise TypeError("capture_adapter must return a capture-compatible nn.Module")
        return graph
    return torch.export.export(model, call.args, kwargs=call.kwargs).module()


def _tensor_tree(value: Any, path: str = "input"):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _tensor_tree(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _tensor_tree(child, f"{path}[{index}]")


def _capture_descriptor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "__tensor__": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "device": "cpu",
        }
    if isinstance(value, dict):
        return {str(k): _capture_descriptor(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [_capture_descriptor(v) for v in value]}
    if isinstance(value, list):
        return [_capture_descriptor(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Capture argument {type(value).__name__} needs a registered artifact codec")


def _signature(model: nn.Module) -> str:
    body = [
        (name, type(mod).__module__ + "." + type(mod).__qualname__)
        for name, mod in model.named_modules()
    ]
    body += [
        (name, tuple(param.shape), str(param.dtype), bool(param.requires_grad))
        for name, param in model.named_parameters()
    ]
    return hashlib.sha256(json.dumps(body, default=str).encode()).hexdigest()


def _packed_storage(model: nn.Module) -> dict[str, Any]:
    from .quantization import _storage_metadata

    return _storage_metadata(model)


class TorchAOQuantizationSession:
    def __init__(
        self,
        model: nn.Module,
        graph: nn.Module,
        *,
        mode: str,
        config: Any,
        quantizer: Any,
        target: str,
        report: CompatibilityReport,
        calibration_example: ModelCall | None = None,
    ):
        self.model, self.source_model = graph, model
        self.mode, self.config, self.quantizer, self.target = mode, config, quantizer, target
        self.compatibility, self.state = report, "prepared"
        self.calibration_example = calibration_example
        self.calibration_history: dict[str, Any] = {"batches": 0, "shapes": {}, "source": None}
        self.training_steps = 0
        self.observer_enabled, self.fake_quant_enabled = True, True
        self._hook_handles: list[Any] = []
        self._observer_gates: list[tuple[Any, dict[str, bool]]] = []
        self._schedule_id = getattr(config, "schedule_id", "static-ptq-v1")
        self._observer_schedule = None
        self._fake_quant_schedule = None
        self._signature = _signature(graph)
        self.model.__topologies_optimization_session__ = self

    def calibrate(
        self,
        data: Any,
        *,
        adapter: Callable[[Any], ModelCall] | None = None,
        max_batches: int | None = None,
        min_batches: int | None = None,
        valid_mask: Callable[[Any], Any] | None = None,
        sample_counter: Callable[[Any], int] | None = None,
        replay_factory: Callable[[], Any] | None = None,
        source_id: str | None = None,
        preprocessing_id: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if self.mode != "ptq" or self.state != "prepared":
            raise OptimizationError(
                f"Calibration is invalid in {self.mode}/{self.state}",
                code="invalid_stage_order",
                backend="torchao-pt2e",
                stage="calibrate",
            )
        limit = (
            getattr(self.config, "max_calibration_batches", 128)
            if max_batches is None
            else max_batches
        )
        minimum = (
            getattr(self.config, "min_calibration_batches", 1)
            if min_batches is None
            else min_batches
        )
        if limit < 1 or minimum < 1 or minimum > limit:
            raise ValueError("calibration bounds require 1 <= min_batches <= max_batches")
        source_id = source_id or getattr(self.config, "calibration_id", None)
        preprocessing_id = preprocessing_id or getattr(self.config, "preprocessing_id", None)
        before = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        flags = [(module, module.training) for module in self.model.modules()]
        batch_count = samples = 0
        shapes: dict[str, int] = {}
        created_from_factory = False
        try:
            with execution_rng(seed=seed), torch.inference_mode():
                if callable(data) and replay_factory is None:
                    replay_factory, data = data, data()
                    created_from_factory = True
                iterator = iter(
                    data
                    if created_from_factory
                    else replay_factory()
                    if replay_factory is not None
                    else data
                )
                while batch_count < limit:
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                    call = (
                        batch
                        if isinstance(batch, ModelCall)
                        else adapter(batch)
                        if adapter
                        else None
                    )
                    if not isinstance(call, ModelCall):
                        raise OptimizationError(
                            "Calibration batches require a ModelCall or explicit adapter",
                            code="invalid_calibration_data",
                            stage="calibrate",
                        )
                    for path, tensor in _tensor_tree((call.args, call.kwargs)):
                        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                            raise OptimizationError(
                                f"Nonfinite calibration tensor at {path}",
                                code="invalid_calibration_data",
                                stage="calibrate",
                            )
                        shape_key = f"{path}:{tuple(tensor.shape)}:{tensor.dtype}"
                        shapes[shape_key] = shapes.get(shape_key, 0) + 1
                    if valid_mask is not None:
                        mask = valid_mask(batch)
                        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
                            raise OptimizationError(
                                "valid_mask must return a boolean tensor",
                                code="invalid_calibration_data",
                                stage="calibrate",
                            )
                        if not bool(mask.any()):
                            raise OptimizationError(
                                "Calibration batch has no valid masked values",
                                code="invalid_calibration_data",
                                stage="calibrate",
                            )
                    self.model(*call.args, **call.kwargs)
                    batch_count += 1
                    if sample_counter is not None:
                        value = sample_counter(batch)
                        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                            raise ValueError("sample_counter must return a nonnegative integer")
                        samples += value
            if batch_count < minimum:
                raise OptimizationError(
                    f"Calibration consumed {batch_count} usable batches; need {minimum}",
                    code="invalid_calibration_data",
                    stage="calibrate",
                )
        except Exception:
            self.model.load_state_dict(before, strict=True)
            raise
        finally:
            for module, flag in flags:
                module.training = flag
        self.calibration_history = {
            "batches": batch_count,
            "samples": samples if sample_counter else None,
            "shapes": shapes,
            "source": source_id,
            "preprocessing": preprocessing_id,
            "seed": seed,
            "replayable": replay_factory is not None,
            "one_shot_consumed": replay_factory is None,
        }
        self.state = "calibrated"
        return dict(self.calibration_history)

    def attach(
        self,
        trainer: Any,
        *,
        observer_schedule: Callable[[int], bool] | None = None,
        fake_quant_schedule: Callable[[int], bool] | None = None,
    ) -> None:
        if self.mode not in ("qat", "qat-native") or self.state != "prepared":
            raise OptimizationError(
                "Only a prepared QAT session can attach to Trainer",
                code="invalid_stage_order",
                stage="attach",
            )
        if (
            getattr(getattr(trainer, "dist", None), "world_size", 1) > 1
            or getattr(trainer, "strategy", None) == "ddp"
        ):
            raise OptimizationError(
                "QAT with DDP is not verified; run single-process or provide a tested observer-sync backend",
                code="unsupported_target",
                backend="torchao-pt2e",
                stage="attach",
            )
        if trainer.model is not self.model:
            raise OptimizationError(
                "Trainer must own the prepared QAT model before optimizer construction",
                code="invalid_stage_order",
                stage="attach",
            )
        recipe_level = getattr(getattr(trainer, "recipe", None), "level", "loss")
        if recipe_level not in ("loss", "step"):
            raise OptimizationError(
                f"Trainer recipe level {recipe_level!r} has no verified QAT step-hook integration; "
                "use a loss/step recipe or own observer/fake-quant scheduling",
                code="unsupported_operation",
                stage="attach",
            )
        for module in self.model.modules():
            if "observer" in type(module).__name__.lower():
                gate = {"enabled": self.observer_enabled}
                handle = module.register_forward_pre_hook(
                    lambda _module, args, gate=gate: None if gate["enabled"] else args
                )
                self._hook_handles.append(handle)
                self._observer_gates.append((module, gate))
            if hasattr(module, "enabled") and "fakequant" in type(module).__name__.lower():
                module.enabled = self.fake_quant_enabled
        fake_quantizers = [
            module
            for module in self.model.modules()
            if hasattr(module, "enabled") and "fakequant" in type(module).__name__.lower()
        ]
        if observer_schedule is not None and not self._observer_gates:
            raise OptimizationError(
                "Selected QAT backend exposes no observer schedule controls; "
                "supply an observer-aware quantizer or omit observer_schedule",
                code="unsupported_operation",
                stage="attach",
            )
        if fake_quant_schedule is not None and not fake_quantizers:
            raise OptimizationError(
                "Selected QAT backend exposes no fake-quant schedule controls",
                code="unsupported_operation",
                stage="attach",
            )

        def on_step(step: int):
            self.training_steps = max(self.training_steps, int(step) + 1)
            self.state = "trained"
            if observer_schedule is not None:
                enabled = bool(observer_schedule(step))
                self.observer_enabled = enabled
                for _module, gate in self._observer_gates:
                    gate["enabled"] = enabled
            if fake_quant_schedule is not None:
                enabled = bool(fake_quant_schedule(step))
                self.fake_quant_enabled = enabled
                for module in self.model.modules():
                    if hasattr(module, "enabled") and "fakequant" in type(module).__name__.lower():
                        module.enabled = enabled
                    if enabled and callable(getattr(module, "enable_fake_quant", None)):
                        module.enable_fake_quant()
                    elif not enabled and callable(getattr(module, "disable_fake_quant", None)):
                        module.disable_fake_quant()

        add_hook = getattr(trainer, "add_hook", None)
        register_hook = getattr(trainer, "register_optimization_step_hook", None)
        if callable(add_hook):
            add_hook("before_optimizer_step", on_step)
        elif callable(register_hook):  # older/duck-typed trainers
            register_hook(on_step)
        else:
            raise OptimizationError(
                "Trainer lacks transformation step hooks; upgrade Trainer integration",
                code="unsupported_operation",
                stage="attach",
            )
        self._observer_schedule = observer_schedule
        self._fake_quant_schedule = fake_quant_schedule
        self._attached = True

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "topologies-quantization-session",
            "version": 1,
            "mode": self.mode,
            "state": self.state,
            "signature": self._signature,
            "config": self._config_fingerprint(),
            "history": self.calibration_history,
            "training_steps": self.training_steps,
            "observer_enabled": self.observer_enabled,
            "fake_quant_enabled": self.fake_quant_enabled,
            "schedule_id": self._schedule_id,
        }

    def validate_state_dict(self, state: dict[str, Any]) -> None:
        if (
            self.mode not in ("qat", "qat-native")
            or state.get("mode") != self.mode
            or state.get("state") == "converted"
        ):
            raise OptimizationError(
                "Checkpoint is not resumable QAT state; converted inference artifacts cannot resume training",
                code="incompatible_resume",
                stage="resume",
            )
        if (
            state.get("signature") != self._signature
            or state.get("config") != self._config_fingerprint()
        ):
            raise OptimizationError(
                "Prepared QAT topology/config differs from checkpoint",
                code="incompatible_resume",
                stage="resume",
            )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.validate_state_dict(state)
        self.state = state.get("state", "prepared")
        self.calibration_history = dict(state.get("history", {}))
        self.training_steps = int(state.get("training_steps", 0))
        self.observer_enabled = bool(state.get("observer_enabled", True))
        self.fake_quant_enabled = bool(state.get("fake_quant_enabled", True))

    def _config_fingerprint(self):
        fields = {
            k: str(v)
            for k, v in vars(self.config).items()
            if k not in ("quantizer", "capture_adapter")
        }
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()

    def convert(self) -> OptimizationResult:
        allowed = (
            self.state == "calibrated"
            if self.mode == "ptq"
            else self.state in ("prepared", "trained")
        )
        if not allowed:
            raise OptimizationError(
                f"Conversion is invalid in session state {self.state!r}",
                code="invalid_stage_order",
                backend="torchao-pt2e",
                stage="convert",
            )
        try:
            if self.mode == "qat-native":
                from torchao.quantization import quantize_
                from torchao.quantization.qat import QATConfig as TorchAOQATConfig

                converted = copy.deepcopy(self.model)
                quantize_(converted, TorchAOQATConfig(self.config.base_config, step="convert"))
            else:
                from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e

                converted = convert_pt2e(self.model)
                if self.mode == "qat":
                    from torchao.quantization.pt2e import move_exported_model_to_eval

                    converted = move_exported_model_to_eval(converted)
            if self.mode == "qat-native":
                converted.eval()
        except Exception as exc:
            if (
                self.mode == "qat-native"
                and isinstance(exc, ImportError)
                and "mslk" in str(exc).lower()
            ):
                message = (
                    "Int4 QAT conversion needs platform-compatible mslk>=1.0.0; "
                    "verified int8 QAT uses IntxWeightOnlyConfig without MSLK. "
                    f"Backend error: {exc}"
                )
            else:
                message = (
                    f"TorchAO QAT conversion failed: {exc}"
                    if self.mode == "qat-native"
                    else f"PT2E conversion failed: {exc}"
                )
            raise OptimizationError(
                message,
                code="unsupported_operation",
                backend="torchao-qat" if self.mode == "qat-native" else "torchao-pt2e",
                stage="convert",
            ) from exc
        self.state = "converted"
        converted.__topologies_capture_call__ = self.calibration_example
        for handle in self._hook_handles:
            handle.remove()
        try:
            capture_spec = {
                "args": _capture_descriptor(self.calibration_example.args),
                "kwargs": _capture_descriptor(self.calibration_example.kwargs),
            }
            portable_capture = (
                getattr(self.config, "capture_adapter", None) is None
                and getattr(self.config, "quantizer", None) is None
            )
        except Exception:  # noqa: BLE001 - capture metadata is optional; record as non-portable
            capture_spec, portable_capture = None, False
        representation = "reference-qdq" if self.mode == "ptq" else "qat-observer-reference"
        if self.mode == "qat-native":
            representation = "torchao-int8-unpacked-weight-only"
        resolved = {
            "preset": "int8-static" if self.mode == "ptq" else "int8-qat",
            "backend": "torchao-pt2e-arm-cpu",
            "target": self.target,
            "representation": representation,
            "calibration": self.calibration_history,
            "trainable": False,
            "movement": "cpu-only",
            "capture_spec": capture_spec,
            "portable_capture": portable_capture,
            "session_mode": self.mode,
            "quantizer_id": "arm-inductor-default-v1" if self.config.quantizer is None else None,
            "schedule_id": self._schedule_id,
            "fake_quant_enabled": self.fake_quant_enabled,
            "qat_numerics": "int8-weight-per-axis-0-unpacked"
            if self.mode == "qat-native"
            else None,
        }
        if self.mode == "qat-native":
            resolved["qat_base_config"] = {
                "class": type(self.config.base_config).__name__,
                "weight_dtype": str(self.config.base_config.weight_dtype),
                "granularity": str(self.config.base_config.granularity),
                "mapping_type": str(self.config.base_config.mapping_type),
                "scale_dtype": (
                    str(self.config.base_config.scale_dtype)
                    if self.config.base_config.scale_dtype is not None
                    else None
                ),
                "packing": str(self.config.base_config.intx_packing_format),
                "version": int(self.config.base_config.version),
            }
        backend_id = (
            "torchao-native-qat-int8-linear-cpu"
            if self.mode == "qat-native"
            else getattr(self.config, "backend_id", None)
            or (
                "torchao-pt2e-arm-cpu"
                if self.config.quantizer is None
                else "torchao-pt2e-custom-cpu"
            )
        )
        resolved["backend"] = backend_id
        resolved["operator_mapping"] = dict(self.compatibility.operator_mapping)
        resolved["boundaries"] = list(self.compatibility.boundaries)
        return OptimizationResult(
            converted,
            resolved,
            self.compatibility,
            backend_id,
            "not_run",
            history=[dict(self.calibration_history)],
            codec=(
                "torchao-qat-int8-unpacked-v1"
                if self.mode == "qat-native"
                else "torchao-pt2e-reference-v1"
            ),
            storage=(
                _packed_storage(converted)
                if self.mode == "qat-native"
                else {
                    "tensor_backing_bytes": None,
                    "known": False,
                    "note": "reference Q/DQ graph; lowered packed-kernel storage is not asserted",
                }
            ),
            trainable=any(p.requires_grad for p in converted.parameters()),
            movement="CPU-only; compile/lowering is explicit",
            load_targets=(self.target,),
        )


def prepare_pt2e_session(
    model: nn.Module,
    config: StaticPTQConfig | QATConfig,
    *,
    target: str,
    examples: list[ModelCall],
    mode: str,
    model_factory: Any = None,
) -> TorchAOQuantizationSession:
    if target != "cpu":
        raise OptimizationError(
            "Initial PT2E quantizer is verified for CPU only; supply a tested target quantizer",
            code="unsupported_target",
            backend="torchao-pt2e",
            stage="inspect",
        )
    if not examples:
        raise OptimizationError(
            "PT2E capture requires explicit example calls", code="missing_examples", stage="prepare"
        )
    if any(
        tensor.device.type != "cpu" for tensor in list(model.parameters()) + list(model.buffers())
    ):
        raise OptimizationError(
            "Initial PT2E path does not move model parameters; reconstruct the source model on CPU",
            code="unsupported_target",
            backend="torchao-pt2e",
            stage="prepare",
        )
    from .quantization import _torchao_api

    _torchao_api()
    if mode == "qat" and config.base_config is not None:
        if type(config.base_config).__name__ == "Int8WeightOnlyConfig":
            raise OptimizationError(
                "Int8WeightOnlyConfig is PTQ-only; use IntxWeightOnlyConfig(weight_dtype=torch.int8) for QAT",
                code="unsupported_operation",
                backend="torchao-qat",
                stage="prepare",
            )
        if type(config.base_config).__name__ == "Int4WeightOnlyConfig":
            raise OptimizationError(
                "Native Int4 QAT conversion requires platform-compatible mslk>=1.0.0; "
                "use IntxWeightOnlyConfig(weight_dtype=torch.int8) for verified QAT",
                code="unsupported_operation",
                backend="torchao-qat",
                stage="prepare",
            )
        try:
            from torchao.quantization import quantize_
            from torchao.quantization.qat import QATConfig as TorchAOQATConfig

            if type(config.base_config).__name__ != "IntxWeightOnlyConfig":
                raise ValueError(
                    "Native QAT supports IntxWeightOnlyConfig(weight_dtype=torch.int8) only"
                )
            if config.base_config.weight_dtype != torch.int8:
                raise ValueError("This native QAT preset only supports int8 weights")
            if str(config.base_config.granularity) != "PerAxis(axis=0)":
                raise ValueError(
                    "This native QAT path only supports per-output-channel granularity on axis 0"
                )
            if str(config.base_config.intx_packing_format) != "IntxPackingFormat.UNPACKED_TO_INT8":
                raise ValueError(
                    "This tested native QAT path requires unpacked int8 backing storage"
                )
            parameterized = [
                (name, module)
                for name, module in model.named_modules()
                if any(True for _ in module.parameters(recurse=False))
            ]
            if not parameterized or any(
                not isinstance(module, nn.Linear) for _, module in parameterized
            ):
                raise ValueError("Native TorchAO QAT handler supports Linear modules only")
            rng_state = torch.random.get_rng_state()
            module_modes = {name: module.training for name, module in model.named_modules()}
            try:
                quantize_(model, TorchAOQATConfig(config.base_config, step="prepare"))
                converted_modules = dict(model.named_modules())
                for name, training in module_modes.items():
                    if name in converted_modules:
                        converted_modules[name].training = training
            finally:
                torch.random.set_rng_state(rng_state)
            report = CompatibilityReport(supported=[name or "<root>" for name, _ in parameterized])
            return TorchAOQuantizationSession(
                model,
                model,
                mode="qat-native",
                config=config,
                quantizer=None,
                target=target,
                report=report,
                calibration_example=examples[0],
            )
        except OptimizationError:
            raise
        except Exception as exc:
            raise OptimizationError(
                f"TorchAO native QAT prepare failed: {exc}",
                code="unsupported_operation",
                backend="torchao-qat",
                stage="prepare",
            ) from exc
    source = model
    # A factory is handled by the facade clone helper to preserve source topology and weights.
    capture_adapter = config.capture_adapter
    call = examples[0]
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        quantizer = _quantizer(config.quantizer)
        flags = [(module, module.training) for module in source.modules()]
        if mode == "ptq":
            source.eval()
        with torch.random.fork_rng(devices=[]):
            candidate = _capture(source, call, capture_adapter)
        for module, flag in flags:
            module.training = flag
        if mode == "ptq":
            from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e

            prepared = prepare_pt2e(candidate, quantizer)
            from torchao.quantization.pt2e import move_exported_model_to_eval

            prepared = move_exported_model_to_eval(prepared)
        else:
            from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e

            prepared = prepare_qat_pt2e(candidate, quantizer)
            fake_quant_present = any(
                "fakequant" in type(module).__name__.lower() for module in prepared.modules()
            ) or any("fake_quant" in str(node.target).lower() for node in prepared.graph.nodes)
            if not fake_quant_present:
                raise OptimizationError(
                    "Selected PT2E quantizer inserts observers but no fake-quant operations; "
                    "use a fake-quant-capable quantizer or native TorchAO QAT config",
                    code="unsupported_operation",
                    backend="torchao-pt2e",
                    stage="prepare",
                )
            from torchao.quantization.pt2e import move_exported_model_to_train

            prepared = move_exported_model_to_train(prepared)
    except OptimizationError:
        raise
    except Exception as exc:
        raise OptimizationError(
            f"PT2E graph capture/preparation failed: {exc}; provide a capture_adapter for this model",
            code="capture_failed",
            backend="torchao-pt2e",
            stage="prepare",
            remedies=(
                "Provide capture_adapter(model, ModelCall)",
                "Use an export-compatible model wrapper",
            ),
        ) from exc
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    compatible = CompatibilityReport(
        supported=[
            n or "<root>"
            for n, m in source.named_modules()
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
        ]
    )
    if not compatible.supported:
        raise OptimizationError(
            "PT2E requires at least one Linear/Conv operation in the initial handler set",
            code="unsupported_operation",
            backend="torchao-pt2e",
            stage="inspect",
        )
    compatible.operator_mapping = {
        node.name: str(node.target)
        for node in candidate.graph.nodes
        if node.op in ("call_function", "call_method", "call_module")
    }
    compatible.boundaries = [
        {"node": node.name, "target": str(node.target), "kind": "observed_float_or_unclassified"}
        for node in candidate.graph.nodes
        if node.op in ("call_function", "call_method")
    ]
    return TorchAOQuantizationSession(
        source,
        prepared,
        mode=mode,
        config=config,
        quantizer=quantizer,
        target=target,
        report=compatible,
        calibration_example=call,
    )
