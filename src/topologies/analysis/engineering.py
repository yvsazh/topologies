"""Static engineering reports for PyTorch modules, without executing ``forward``."""

from __future__ import annotations

import math
import platform
from collections.abc import Callable, Mapping
from typing import Any

from topologies.analysis.core import AnalysisContext, AnalysisResult, analyze


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("engineering analyzers require PyTorch") from exc
    return torch


def _module(value: Any):
    if not isinstance(value, _torch().nn.Module):
        raise TypeError("data must be a torch.nn.Module")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be a finite non-negative number")
    return float(value)


def _storage_allocations(tensor: Any, active=None) -> dict | None:
    """Inspect public backing/flatten protocols without importing a backend."""
    active = set() if active is None else active
    if id(tensor) in active:
        return None
    try:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
        return {key: key[2]}
    except (AttributeError, RuntimeError, NotImplementedError):
        flatten = getattr(tensor, "__tensor_flatten__", None)
        if not callable(flatten):
            return None
        active.add(id(tensor))
        try:
            names, _ = flatten()
            if not names:
                return None
            allocations = {}
            for name in names:
                child = _storage_allocations(getattr(tensor, name), active)
                if child is None:
                    return None
                allocations.update(child)
            return allocations
        except (AttributeError, RuntimeError, NotImplementedError, TypeError, ValueError):
            return None
        finally:
            active.remove(id(tensor))


def _backing_bytes(tensor: Any) -> int | None:
    allocations = _storage_allocations(tensor)
    return sum(allocations.values()) if allocations is not None else None


def _storage_total(tensors):
    allocations, unknown, logical = {}, 0, 0
    for tensor in tensors:
        backing = _storage_allocations(tensor)
        if backing is None:
            unknown += 1
            logical += int(tensor.numel() * tensor.element_size())
        else:
            allocations.update(backing)
    return sum(allocations.values()), unknown, logical


def _bytes(tensor: Any) -> int:
    """Return backing bytes where inspectable; otherwise a separately qualified logical size."""
    visible = _backing_bytes(tensor)
    return visible if visible is not None else int(tensor.numel() * tensor.element_size())


def _storage_key(tensor: Any):
    try:
        storage = tensor.untyped_storage()
        return (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
    except (AttributeError, RuntimeError, NotImplementedError):
        return None


def _unique(named):
    seen = set()
    result = []
    for name, value in named:
        if id(value) not in seen:
            seen.add(id(value))
            result.append((name, value))
    return result


class ModelStructure:
    """Count unique parameters/storage and list module, parameter, and buffer rows."""

    name = "model_structure"

    def run(self, data: Any, context: AnalysisContext) -> AnalysisResult:
        model = _module(data)
        named = list(model.named_parameters(remove_duplicate=False))
        seen: dict[int, str] = {}
        parameter_rows = []
        module_parameters: dict[str, dict[str, int]] = {}
        for name, param in named:
            alias = seen.get(id(param))
            if alias is None:
                seen[id(param)] = name
            module_name = name.rpartition(".")[0]
            item = module_parameters.setdefault(
                module_name, {"unique_parameters": 0, "alias_parameters": 0}
            )
            item["alias_parameters" if alias is not None else "unique_parameters"] += param.numel()
            parameter_rows.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                    "bytes": _bytes(param),
                    "trainable": param.requires_grad,
                    "alias_of": alias,
                }
            )
        unique_params = _unique(named)
        buffers = list(model.named_buffers(remove_duplicate=False))
        unique_buffers = _unique(buffers)
        seen_buffers: dict[int, str] = {}
        buffer_rows = []
        for name, buffer in buffers:
            alias = seen_buffers.get(id(buffer))
            if alias is None:
                seen_buffers[id(buffer)] = name
            buffer_rows.append(
                {
                    "name": name,
                    "shape": list(buffer.shape),
                    "dtype": str(buffer.dtype),
                    "bytes": _bytes(buffer),
                    "alias_of": alias,
                }
            )
        storage_bytes = {}
        unavailable = 0
        for _, tensor in [*unique_params, *unique_buffers]:
            allocations = _storage_allocations(tensor)
            if allocations is None:
                unavailable += 1
            else:
                storage_bytes.update(allocations)
        payload = {
            "parameters": {
                "total": sum(p.numel() for _, p in unique_params),
                "trainable": sum(p.numel() for _, p in unique_params if p.requires_grad),
                "frozen": sum(p.numel() for _, p in unique_params if not p.requires_grad),
                "bytes": sum(_bytes(p) for _, p in unique_params),
            },
            "modules": [
                {"name": n, "type": type(m).__qualname__} for n, m in model.named_modules()
            ],
            "module_parameters": [
                {"module": name, **counts} for name, counts in module_parameters.items()
            ],
            "parameter_rows": parameter_rows,
            "buffers": buffer_rows,
            "buffer_bytes": sum(_bytes(b) for _, b in unique_buffers),
            "storage": {
                "parameter_and_buffer_backing_bytes": sum(storage_bytes.values()),
                "kind": "computed" if unavailable == 0 else "partial",
                "unique_backing_allocations": len(storage_bytes),
                "unavailable_tensors": unavailable,
                "note": "backend-private non-tensor packing and serialized artifact overhead are not included",
            },
            "storage_overlap_limitation": "logical tensor rows may share a backing allocation; storage total deduplicates visible allocations",
        }
        return AnalysisResult(payload, kind="computed", name=self.name)


_PRECISION_BYTES = {
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "fp32": 4,
    "float32": 4,
    "fp64": 8,
    "float64": 8,
}


class ModelMemory:
    """Actual tensor storage and separately labeled caller-controlled estimates."""

    name = "model_memory"

    def run(self, data: Any, context: AnalysisContext) -> AnalysisResult:
        model = _module(data)
        params = [p for _, p in _unique(model.named_parameters())]
        buffers = [b for _, b in _unique(model.named_buffers())]
        options = context.parameters
        precision = options.get("precision")
        target_width = None
        if precision is not None:
            key = str(precision).lower()
            if key not in _PRECISION_BYTES:
                raise ValueError(f"unsupported precision: {precision}")
            target_width = _PRECISION_BYTES[key]
        trainable = [p for p in params if p.requires_grad]
        optimizer = str(options.get("optimizer", "unknown")).lower()
        optimizer_bytes = options.get("optimizer_state_bytes_per_parameter")
        if optimizer_bytes is not None:
            optimizer_bytes = _nonnegative(optimizer_bytes, "optimizer_state_bytes_per_parameter")
        elif optimizer in ("adam", "adamw"):
            state_width = _PRECISION_BYTES.get(
                str(options.get("optimizer_state_precision", "fp32")).lower()
            )
            if state_width is None:
                raise ValueError("unsupported optimizer_state_precision")
            optimizer_bytes = state_width * 2
        elif optimizer == "sgd":
            optimizer_bytes = (target_width or 4) if options.get("momentum", False) else 0
        elif optimizer in ("none", "disabled"):
            optimizer_bytes = 0
        elif optimizer != "unknown":
            raise ValueError(f"unsupported optimizer: {optimizer}")
        master_copy = options.get("master_weight_copy", False)
        if not isinstance(master_copy, bool):
            raise TypeError("master_weight_copy must be boolean")
        parameter_backing, parameter_unknown, parameter_logical = _storage_total(params)
        buffer_backing, buffer_unknown, buffer_logical = _storage_total(buffers)
        checkpoint_backing, checkpoint_unknown, _ = _storage_total([*params, *buffers])
        items: dict[str, Any] = {
            "parameter_storage": {
                "value": parameter_backing if not parameter_unknown else None,
                "unit": "bytes",
                "visible_backing_bytes": parameter_backing,
                "unavailable_tensors": parameter_unknown,
                "logical_unavailable_bytes": parameter_logical,
                "kind": "computed" if not parameter_unknown else "unknown",
                "assumptions": [
                    "deduplicated visible backing allocations, including tensor-subclass scales; opaque storage remains unknown and backend-private non-tensor allocations are excluded"
                ],
            },
            "buffer_storage": {
                "value": buffer_backing if not buffer_unknown else None,
                "unit": "bytes",
                "visible_backing_bytes": buffer_backing,
                "unavailable_tensors": buffer_unknown,
                "logical_unavailable_bytes": buffer_logical,
                "kind": "computed" if not buffer_unknown else "unknown",
                "assumptions": [
                    "deduplicated visible buffer allocations; opaque storage remains unknown"
                ],
            },
            "gradient_storage": {
                "value": sum(int(p.numel() * p.element_size()) for p in trainable),
                "unit": "bytes",
                "kind": "estimated",
                "assumptions": ["one gradient per trainable parameter; actual parameter dtype"],
            },
            "checkpoint_tensor_bytes_lower_bound": {
                "value": checkpoint_backing,
                "unit": "bytes",
                "kind": "partial" if checkpoint_unknown else "computed",
                "assumptions": [
                    "visible backing allocations only; excludes opaque storage and checkpoint container/serialization overhead"
                ],
            },
        }
        if precision is not None:
            items["hypothetical_parameter_storage"] = {
                "value": sum(p.numel() for p in params) * target_width,
                "unit": "bytes",
                "kind": "estimated",
                "assumptions": [
                    f"hypothetical stored dtype {precision}; autocast alone does not convert parameter storage"
                ],
            }
            items["hypothetical_gradient_storage"] = {
                "value": sum(p.numel() for p in trainable) * target_width,
                "unit": "bytes",
                "kind": "estimated",
                "assumptions": [
                    f"one gradient per trainable parameter at hypothetical stored dtype {precision}"
                ],
            }
        if optimizer_bytes is not None:
            items["optimizer_state"] = {
                "value": sum(p.numel() for p in trainable) * optimizer_bytes,
                "unit": "bytes",
                "kind": "estimated",
                "assumptions": [
                    (
                        f"{optimizer} state; {optimizer_bytes} bytes per trainable parameter"
                        if options.get("optimizer_state_bytes_per_parameter") is None
                        else "caller override bytes per trainable parameter"
                    ),
                    f"optimizer state precision: {options.get('optimizer_state_precision', 'fp32')}",
                ],
            }
        if master_copy:
            items["master_weight_copy"] = {
                "value": sum(p.numel() for p in trainable) * 4,
                "unit": "bytes",
                "kind": "estimated",
                "assumptions": ["one fp32 master copy per trainable parameter"],
            }
        for key in ("activation_bytes", "workspace_bytes"):
            if key in options:
                items[key] = {
                    "value": _nonnegative(options[key], key),
                    "unit": "bytes",
                    "kind": "estimated",
                    "assumptions": ["caller-supplied estimate"],
                }
            else:
                items[key] = {
                    "value": None,
                    "unit": "bytes",
                    "kind": "unknown",
                    "assumptions": ["not supplied or measured"],
                }
        return AnalysisResult(items, kind="engineering_report", name=self.name)


class KVCacheSpec:
    """Explicit generic key/value cache dimensions; no architecture inference."""

    def __init__(
        self,
        *,
        layers: int,
        kv_heads: int,
        head_dim: int,
        batch_size: int,
        sequence_length: int,
        dtype_bytes: int,
    ):
        self.layers = _positive_int(layers, "layers")
        self.kv_heads = _positive_int(kv_heads, "kv_heads")
        self.head_dim = _positive_int(head_dim, "head_dim")
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.sequence_length = _positive_int(sequence_length, "sequence_length")
        self.dtype_bytes = _positive_int(dtype_bytes, "dtype_bytes")

    @property
    def bytes(self) -> int:
        return (
            2
            * self.layers
            * self.kv_heads
            * self.head_dim
            * self.batch_size
            * self.sequence_length
            * self.dtype_bytes
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.bytes,
            "unit": "bytes",
            "kind": "estimated",
            "assumptions": ["K and V each: layers*batch*sequence*KV_heads*head_dim*dtype_bytes"],
            "dimensions": {
                "layers": self.layers,
                "kv_heads": self.kv_heads,
                "head_dim": self.head_dim,
                "batch_size": self.batch_size,
                "sequence_length": self.sequence_length,
                "dtype_bytes": self.dtype_bytes,
            },
        }


class ComputeProviderRegistry:
    """Register handler(module, input_shape) -> (MACs, output_shape) or None."""

    def __init__(self):
        self.handlers: dict[type, Callable] = {}

    def register(self, module_type: type, handler: Callable) -> None:
        if not isinstance(module_type, type) or not callable(handler):
            raise TypeError("register requires a module type and callable handler")
        self.handlers[module_type] = handler

    def handler(self, module: Any):
        return next((fn for typ, fn in self.handlers.items() if isinstance(module, typ)), None)


def _default_compute_registry() -> ComputeProviderRegistry:
    torch = _torch()
    registry = ComputeProviderRegistry()

    def linear(module, shape):
        if len(shape) < 1 or shape[-1] != module.in_features:
            return None
        return math.prod(shape[:-1] or (1,)) * module.in_features * module.out_features, (
            *shape[:-1],
            module.out_features,
        )

    def conv(module, shape):
        dimensions = len(module.kernel_size)
        if len(shape) != dimensions + 2 or shape[1] != module.in_channels:
            return None
        spatial = []
        for size, kernel, stride, padding, dilation in zip(
            shape[2:], module.kernel_size, module.stride, module.padding, module.dilation
        ):
            output = (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
            if output <= 0:
                return None
            spatial.append(output)
        macs = math.prod(spatial) * shape[0] * module.out_channels
        macs *= (module.in_channels // module.groups) * math.prod(module.kernel_size)
        return macs, (shape[0], module.out_channels, *spatial)

    registry.register(torch.nn.Linear, linear)
    registry.register(torch.nn.Conv1d, conv)
    registry.register(torch.nn.Conv2d, conv)
    return registry


class ModelCompute:
    """Static MACs for Sequential paths and registered handlers, without graph execution."""

    name = "model_compute"

    def __init__(self, registry: ComputeProviderRegistry | None = None):
        self.registry = registry or _default_compute_registry()

    def run(self, data: Any, context: AnalysisContext) -> AnalysisResult:
        model = _module(data)
        raw_shape = context.parameters.get("input_shape")
        shape = None
        if raw_shape is not None:
            if not isinstance(raw_shape, (tuple, list)) or not raw_shape:
                raise ValueError("input_shape must contain positive integer dimensions")
            shape = tuple(_positive_int(v, "input_shape dimension") for v in raw_shape)
        torch = _torch()
        rows = []
        total = 0
        complete = shape is not None
        if isinstance(model, torch.nn.Sequential):
            operations = [(str(i), child) for i, child in enumerate(model)]
        elif self.registry.handler(model):
            operations = [("", model)]
        else:
            operations = [(name, child) for name, child in model.named_modules() if name]
            complete = False
        for name, module in operations:
            input_shape = shape
            handler = self.registry.handler(module)
            outcome = (
                handler(module, input_shape)
                if handler is not None and input_shape is not None
                else None
            )
            valid = (
                isinstance(outcome, tuple)
                and len(outcome) == 2
                and isinstance(outcome[0], int)
                and not isinstance(outcome[0], bool)
                and outcome[0] >= 0
                and isinstance(outcome[1], (tuple, list))
                and all(isinstance(x, int) and x > 0 for x in outcome[1])
            )
            if not valid:
                complete = False
                shape = None
                macs = None
                output_shape = None
            else:
                macs, output_shape = outcome
                shape = tuple(output_shape)
                total += macs
            bias_adds = None
            if valid and isinstance(module, torch.nn.Linear) and module.bias is not None:
                bias_adds = math.prod(output_shape[:-1] or (1,)) * module.out_features
            elif (
                valid
                and isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d))
                and module.bias is not None
            ):
                bias_adds = math.prod(output_shape[2:]) * output_shape[0] * module.out_channels
            rows.append(
                {
                    "module": name or type(module).__qualname__,
                    "input_shape": list(input_shape) if input_shape else None,
                    "output_shape": list(output_shape) if valid else None,
                    "macs": macs,
                    "bias_adds": bias_adds,
                    "flops_multiply_add_as_two": macs * 2 if valid else None,
                    "coverage": "known" if valid else "unknown",
                }
            )
        multiplier = _nonnegative(
            context.parameters.get("backward_multiplier", 2.0), "backward_multiplier"
        )
        return AnalysisResult(
            {
                "modules": rows,
                "macs": total if complete else None,
                "flops_multiply_add_as_two": total * 2 if complete else None,
                "coverage": "complete" if complete else "partial_or_unknown",
                "backward_multiplier": {
                    "value": multiplier,
                    "kind": "heuristic",
                    "assumptions": [
                        "caller-configurable multiplier, not included in forward total"
                    ],
                },
            },
            kind="computed",
            name=self.name,
            warnings=[]
            if complete
            else ["Unsupported or unknown graph paths are not included in a total"],
        )


def _cpu_ram_bytes() -> int | None:
    try:
        import os

        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    if platform.system() == "Darwin":
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            return int(result.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    return None


class HardwareInfo:
    """Query available hardware without allocating tensors."""

    name = "hardware_info"

    def run(self, data: Any = None, context: AnalysisContext | None = None) -> AnalysisResult:
        torch = _torch()
        warnings = []
        devices: dict[str, Any] = {
            "cpu": {"platform": platform.platform(), "ram_bytes": _cpu_ram_bytes()}
        }
        try:
            gpus = []
            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    properties = torch.cuda.get_device_properties(index)
                    free, total = torch.cuda.mem_get_info(index)
                    try:
                        bf16 = bool(torch.cuda.is_bf16_supported(index))
                    except (RuntimeError, TypeError):
                        bf16 = None
                    gpus.append(
                        {
                            "index": index,
                            "name": properties.name,
                            "total_bytes": total,
                            "free_bytes": free,
                            "capability": list(torch.cuda.get_device_capability(index)),
                            "precision": {"fp16": True, "bf16": bf16},
                        }
                    )
            devices["cuda"] = {"available": bool(gpus), "gpus": gpus}
        except (RuntimeError, OSError, AttributeError, NotImplementedError) as exc:
            warnings.append(f"CUDA query unavailable: {type(exc).__name__}: {exc}")
            devices["cuda"] = {"available": None, "gpus": []}
        try:
            devices["mps"] = {"available": bool(torch.backends.mps.is_available())}
        except (RuntimeError, OSError, AttributeError, NotImplementedError) as exc:
            warnings.append(f"MPS query unavailable: {type(exc).__name__}: {exc}")
            devices["mps"] = {"available": None}
        return AnalysisResult(devices, kind="measured", name=self.name, warnings=warnings)


_COMMON_OPTIONS = {
    "batch_size",
    "sequence_length",
    "seq_length",
    "input_shape",
    "precision",
    "optimizer",
    "optimizer_state_precision",
    "optimizer_state_bytes_per_parameter",
    "master_weight_copy",
    "momentum",
    "activation_bytes",
    "workspace_bytes",
    "kv_cache",
    "kv_cache_spec",
    "hardware",
    "device",
    "device_index",
    "budget_bytes",
    "capacity_bytes",
    "capacity_policy",
    "backward_multiplier",
    "storage_policy",
}


class _Feasibility:
    training = False
    name = "feasibility"

    def run(self, data: Any, context: AnalysisContext) -> AnalysisResult:
        model = _module(data)
        params = dict(context.parameters)
        unknown = set(params) - _COMMON_OPTIONS
        if unknown:
            raise ValueError(f"unsupported feasibility options: {', '.join(sorted(unknown))}")
        if "seq_length" in params and "sequence_length" in params:
            raise ValueError("use only sequence_length, not both sequence_length and seq_length")
        if "seq_length" in params:
            params["sequence_length"] = params.pop("seq_length")
        storage_policy = params.get("storage_policy", "actual")
        if storage_policy not in ("actual", "hypothetical"):
            raise ValueError("storage_policy must be 'actual' or 'hypothetical'")
        if storage_policy == "hypothetical" and params.get("precision") is None:
            raise ValueError("storage_policy='hypothetical' requires precision")
        for key in ("batch_size", "sequence_length"):
            if key in params:
                _positive_int(params[key], key)
        child_ctx = AnalysisContext(parameters=params, metadata=context.metadata)
        children: list[AnalysisResult] = []

        def invoke(analyzer):
            child = analyze(model, analyzer, context=child_ctx)
            children.append(child)
            return child.payload

        structure = invoke(ModelStructure())
        memory = invoke(ModelMemory())
        compute = invoke(ModelCompute()) if params.get("input_shape") is not None else None
        hardware = params.get("hardware")
        if hardware is None:
            hardware = invoke(HardwareInfo())
        elif not isinstance(hardware, Mapping):
            raise ValueError("hardware must be a mapping")
        cuda_gpus = hardware.get("cuda", {}).get("gpus", [])
        mps_info = hardware.get("mps", {})
        if "device" in params:
            device = params["device"]
        elif cuda_gpus:
            device = "cuda"
        elif mps_info.get("available") and mps_info.get("free_bytes") is not None:
            device = "mps"
        else:
            device = "cpu"
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be cpu, cuda, or mps")
        explicit_capacity = params.get("budget_bytes", params.get("capacity_bytes"))
        if explicit_capacity is not None:
            capacity = _nonnegative(explicit_capacity, "budget_bytes")
            capacity_source = "explicit budget"
        else:
            capacity, capacity_source = None, None
            if device == "cpu":
                capacity = hardware.get("cpu", {}).get("ram_bytes")
                capacity_source = "CPU total RAM"
            elif device == "mps":
                capacity = hardware.get("mps", {}).get("free_bytes")
                capacity_source = "MPS free bytes"
            else:
                gpus = hardware.get("cuda", {}).get("gpus", [])
                if gpus:
                    index = params.get("device_index", 0)
                    if (
                        not isinstance(index, int)
                        or isinstance(index, bool)
                        or not 0 <= index < len(gpus)
                    ):
                        raise ValueError("device_index must select an available CUDA GPU")
                    gpu = gpus[index]
                    policy = params.get("capacity_policy", "free")
                    if policy not in ("free", "total"):
                        raise ValueError("capacity_policy must be 'free' or 'total'")
                    capacity = gpu.get("free_bytes" if policy == "free" else "total_bytes")
                    capacity_source = f"CUDA GPU {index} {policy} bytes"
                elif "cuda" in hardware and "free_bytes" in hardware["cuda"]:
                    capacity = hardware["cuda"].get("free_bytes")
                    capacity_source = "CUDA free bytes"
        if capacity is not None:
            capacity = _nonnegative(capacity, "hardware capacity")
        lower = memory["parameter_storage"]["value"] + memory["buffer_storage"]["value"]
        component_names = ["parameter_storage", "buffer_storage"]
        unknown_components = []
        if storage_policy == "hypothetical":
            lower += (
                memory["hypothetical_parameter_storage"]["value"]
                - memory["parameter_storage"]["value"]
            )
            component_names[0] = "hypothetical_parameter_storage"
        if self.training:
            gradient_key = (
                "hypothetical_gradient_storage"
                if storage_policy == "hypothetical"
                else "gradient_storage"
            )
            lower += memory[gradient_key]["value"]
            component_names.append(gradient_key)
            if "optimizer_state" in memory:
                lower += memory["optimizer_state"]["value"]
                component_names.append("optimizer_state")
            else:
                unknown_components.append("optimizer_state")
            if "master_weight_copy" in memory:
                lower += memory["master_weight_copy"]["value"]
                component_names.append("master_weight_copy")
        for item in ("activation_bytes", "workspace_bytes"):
            if memory[item]["value"] is not None:
                lower += memory[item]["value"]
                component_names.append(item)
            else:
                unknown_components.append(item)
        kv = params.get("kv_cache_spec", params.get("kv_cache"))
        if kv is not None:
            if isinstance(kv, KVCacheSpec):
                kv_report = kv.as_dict()
            elif isinstance(kv, Mapping):
                kv_report = KVCacheSpec(**kv).as_dict()
            else:
                raise ValueError("kv_cache must be KVCacheSpec or a dimension mapping")
            lower += kv_report["value"]
            component_names.append("kv_cache")
        else:
            kv_report = None
            if not self.training and params.get("sequence_length") is not None:
                unknown_components.append("kv_cache")
        if capacity is None:
            unknown_components.append("capacity")
        if capacity is not None and lower > capacity:
            status = "insufficient"
        else:
            status = "indeterminate" if unknown_components else "within_budget_estimates"
        payload = {
            "status": status,
            "known_lower_bound_bytes": lower,
            "capacity_bytes": capacity,
            "unknown_components": unknown_components,
            "capacity_source": capacity_source,
            "capacity_policy": params.get("capacity_policy", "free"),
            "headroom_bytes": capacity - lower if capacity is not None else None,
            "components_in_lower_bound": component_names,
            "structure": structure,
            "memory": memory,
            "compute": compute,
            "hardware": hardware,
            "kv_cache": kv_report,
            "dimensions": {
                k: params[k]
                for k in ("batch_size", "sequence_length", "input_shape", "precision")
                if k in params
            },
        }
        warnings = [warning for child in children for warning in child.warnings]
        assumptions = [assumption for child in children for assumption in child.assumptions]
        if status == "indeterminate":
            warnings.append(
                "Unknown required components prevent a confident fit claim: "
                + ", ".join(unknown_components)
            )
        return AnalysisResult(
            payload,
            kind="engineering_report",
            name=self.name,
            warnings=list(dict.fromkeys(warnings)),
            assumptions=list(dict.fromkeys(assumptions)),
            stages=children,
            provenance={"analyzer": self.name, "children": [c.provenance for c in children]},
        )


class TrainingFeasibility(_Feasibility):
    """Memory lower-bound assessment for training with explicit optimizer assumptions."""

    training = True
    name = "training_feasibility"


class InferenceFeasibility(_Feasibility):
    """Memory lower-bound assessment for inference; KV cache is never inferred."""

    name = "inference_feasibility"


class ModelAnalysis:
    """Default static model report; accepts shape/precision from context parameters."""

    name = "model_analysis"

    def run(self, model: Any, context: AnalysisContext) -> AnalysisResult:
        options = {
            key: context.parameters[key]
            for key in ("input_shape", "precision")
            if key in context.parameters
        }
        return analyze_model(model, **options)


def analyze_model(model: Any, *, input_shape=None, precision=None) -> AnalysisResult:
    """Build structure and memory report; include static compute only with input_shape."""
    params = {
        k: v
        for k, v in {"input_shape": input_shape, "precision": precision}.items()
        if v is not None
    }
    ctx = AnalysisContext(parameters=params)
    stages = [
        analyze(model, ModelStructure(), context=ctx),
        analyze(model, ModelMemory(), context=ctx),
    ]
    if input_shape is not None:
        stages.append(analyze(model, ModelCompute(), context=ctx))
    return AnalysisResult(
        {
            "structure": stages[0].payload,
            "memory": stages[1].payload,
            **({"compute": stages[2].payload} if len(stages) > 2 else {}),
        },
        kind="engineering_report",
        name="model_analysis",
        stages=stages,
        warnings=list(dict.fromkeys(w for stage in stages for w in stage.warnings)),
        assumptions=list(dict.fromkeys(a for stage in stages for a in stage.assumptions)),
        provenance={"analyzer": "analyze_model", "children": [s.provenance for s in stages]},
    )


def _convenience_feasibility(model: Any, analyzer, kwargs) -> AnalysisResult:
    ctx = AnalysisContext(parameters=kwargs)
    return analyze(model, analyzer(), context=ctx)


def analyze_training(model: Any, **kwargs) -> AnalysisResult:
    return _convenience_feasibility(model, TrainingFeasibility, kwargs)


def analyze_inference(model: Any, **kwargs) -> AnalysisResult:
    return _convenience_feasibility(model, InferenceFeasibility, kwargs)


def analyze_hardware() -> AnalysisResult:
    return analyze(None, HardwareInfo())
