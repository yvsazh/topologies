"""Versioned, checksummed artifact codecs with manifest-last atomic publication."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

FORMAT = "topologies-optimization-artifact"
VERSION = 2
_CODECS: dict[str, Any] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_artifact_codec(identifier: str, codec: Any) -> None:
    """Register an explicit codec: save_state(runtime), load_state(factory_runtime, state, config)."""
    if not identifier or identifier in _CODECS:
        raise ValueError(f"codec identifier empty or already registered: {identifier!r}")
    if not callable(getattr(codec, "save_state", None)) or not callable(
        getattr(codec, "load_state", None)
    ):
        raise TypeError(
            "codec must implement save_state(model) and load_state(model, state, config)"
        )
    _CODECS[identifier] = codec


def _tensor_codec_state(model: Any) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError(
            "Default tensor codec refuses non-tensor backend state; register a representation-aware codec"
        )
    return {key: value.detach().cpu() for key, value in state.items()}


def _publish(root: Path, state: Any, manifest: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    weights_name = f"model-{uuid.uuid4().hex}.pt"
    manifest["weights"] = weights_name
    with tempfile.TemporaryDirectory(prefix=".topologies-artifact-", dir=root) as tmp:
        temp = Path(tmp)
        torch.save(state, temp / "model.pt")
        manifest["sha256"] = _sha256(temp / "model.pt")
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        # Immutable generation file + manifest-last replace means the previous published
        # artifact remains valid until the new manifest is atomically visible.
        os.replace(temp / "model.pt", root / weights_name)
        os.replace(temp / "manifest.json", root / "manifest.json")


def save_artifact(
    result: Any,
    path: str | Path,
    *,
    model_factory: Callable[[], Any],
    metadata: dict | None = None,
    codec: str | None = None,
    preprocessing: Any = None,
) -> Path:
    """Save a module or explicit codec-backed runtime; the factory supplies reconstruction."""
    root = Path(path)
    model = result.model
    runtime = getattr(result, "runtime", None)
    if runtime is None:
        runtime = model
    codec_id = codec or getattr(result, "codec", None) or "tensor-state-v1"
    if model is None and codec_id not in _CODECS:
        from .contracts import OptimizationError

        raise OptimizationError(
            f"Runtime-only result requires explicitly registered artifact codec {codec_id!r}",
            code="missing_artifact_codec",
            backend=getattr(result, "backend", None),
            stage="save",
            remedies=("Register a runtime codec and provide its matching reconstruction factory",),
        )
    if (
        codec_id == "torchao-pt2e-reference-v1"
        and getattr(result, "config", {}).get("portable_capture") is not True
    ):
        raise ValueError(
            "PT2E result used custom capture/quantizer; register a matching artifact codec"
        )
    backend_codec = _CODECS.get(codec_id)
    if backend_codec is None and codec_id != "tensor-state-v1":
        # TorchAO codec reconstructs the same topology/config before loading packed state.
        if codec_id == "torchao-pt2e-reference-v1":
            backend_codec = _TorchAOReferenceCodec()
        elif codec_id == "torchao-qat-int8-unpacked-v1":
            backend_codec = _TorchAOQATInt8Codec()
        elif codec_id.startswith("torchao-"):
            backend_codec = _TorchAOCodec()
        elif codec_id.startswith("pytorch-dynamic-legacy-"):
            backend_codec = _LegacyDynamicCodec()
        else:
            if model is None:
                from .contracts import OptimizationError

                raise OptimizationError(
                    f"Runtime-only result requires registered artifact codec {codec_id!r}",
                    code="missing_artifact_codec",
                    backend=getattr(result, "backend", None),
                    remedies=(
                        "Register the codec before saving; arbitrary runtime callables are never pickled",
                    ),
                )
            raise ValueError(f"No artifact codec {codec_id!r} is registered")
    runtime_only = model is None
    if runtime_only and backend_codec is None:
        from .contracts import OptimizationError

        raise OptimizationError(
            f"Runtime-only result requires an explicitly registered artifact codec; {codec_id!r} is not available",
            code="missing_artifact_codec",
            backend=getattr(result, "backend", None),
            remedies=(
                "Register the named codec with save_state(runtime) and load_state(factory_runtime, state, config)",
            ),
        )
    if (
        not runtime_only
        and not callable(getattr(model, "state_dict", None))
        and backend_codec is None
    ):
        raise TypeError(
            "Default tensor codec refuses runtime without state_dict; register an explicit artifact codec"
        )
    state = backend_codec.save_state(runtime) if backend_codec else _tensor_codec_state(model)
    prep_manifest = preprocessing.to_dict() if preprocessing is not None else None
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "backend": result.backend,
        "codec": codec_id,
        "config": result.config,
        "validation": result.validation,
        "representation": "backend-runtime" if runtime_only else "module-backend",
        "model_factory": f"{model_factory.__module__}.{model_factory.__qualname__}",
        "model_factory_required": True,
        "codec_version": 1,
        "load_targets": list(getattr(result, "load_targets", ())),
        "storage": getattr(result, "storage", {}),
        "target_constraints": (
            {
                "device": result.target.device,
                "backend": result.target.backend,
                "hardware_fingerprint": result.target.hardware_fingerprint,
                "workload_constraints": result.target.workload_constraints,
            }
            if getattr(result, "target", None) is not None
            else {}
        ),
        "module_modes": (
            {name: module.training for name, module in model.named_modules(remove_duplicate=False)}
            if callable(getattr(model, "named_modules", None))
            else {}
        ),
        "metadata": metadata or {},
        "preprocessing": prep_manifest,
    }
    _publish(root, state, manifest)
    return root


class _TorchAOCodec:
    def save_state(self, model: Any) -> Any:
        # TorchAO tensor subclasses may carry packed data; serialization is delegated to TorchAO.
        return model.state_dict()

    def prepare_model(self, model: Any, config: dict[str, Any], target: str):
        import torchao

        from .quantization import QuantizationConfig, _torchao_api, quantize

        expected = str(config.get("backend", ""))
        installed = f"torchao-{getattr(torchao, '__version__', 'unknown')}"
        if expected.startswith("torchao-") and expected != installed:
            raise RuntimeError(f"Artifact requires {expected}; installed backend is {installed}")
        _torchao_api()  # Validate versioned API before reading representation state.
        return quantize(
            model,
            QuantizationConfig(preset=config["preset"], modules=tuple(config.get("modules", ()))),
            target=target,
        ).model

    def load_state(self, model: Any, state: Any, config: dict[str, Any]):
        model.load_state_dict(state, strict=True)
        return model


class _LegacyDynamicCodec:
    """Rebuild eager dynamic modules before assigning their packed state dict."""

    def save_state(self, model: Any) -> Any:
        return model.state_dict()

    def prepare_model(self, model: Any, config: dict[str, Any], target: str):
        from .quantization import QuantizationConfig, quantize

        return quantize(
            model,
            QuantizationConfig(preset="dynamic-int8", modules=tuple(config.get("modules", ()))),
            target=target,
        ).model

    def load_state(self, model: Any, state: Any, config: dict[str, Any]):
        model.load_state_dict(state, strict=True)
        return model


class _TorchAOReferenceCodec:
    """Persist a safe PyTorch ExportedProgram with converted reference Q/DQ graph."""

    def save_state(self, model: Any) -> Any:
        call = getattr(model, "__topologies_capture_call__", None)
        if call is None:
            raise ValueError(
                "PT2E reference result is missing its capture call; refusing incomplete artifact"
            )
        import tempfile

        from topologies._execution import ModelCall

        if not isinstance(call, ModelCall):
            raise TypeError("PT2E artifact capture metadata is invalid")
        with tempfile.TemporaryDirectory(prefix="topologies-export-") as folder:
            exported = torch.export.export(model, call.args, kwargs=call.kwargs)
            path = Path(folder) / "model.pt2"
            torch.export.save(exported, path)
            return {"format": "torchao-exported-program-v1", "payload": path.read_bytes()}

    def prepare_model(self, model: Any, config: dict[str, Any], target: str):
        from .quantization import _torchao_api

        if target != "cpu" or config.get("portable_capture") is not True:
            raise ValueError(
                "PT2E artifact requires verified CPU target and portable export capture descriptor"
            )
        _torchao_api()
        return model

    def load_state(self, model: Any, state: Any, config: dict[str, Any]):
        if state.get("format") != "torchao-exported-program-v1" or not isinstance(
            state.get("payload"), bytes
        ):
            raise ValueError("Unsupported TorchAO ExportedProgram codec state")
        import tempfile

        with tempfile.TemporaryDirectory(prefix="topologies-export-load-") as folder:
            path = Path(folder) / "model.pt2"
            path.write_bytes(state["payload"])
            exported = torch.export.load(path)
            return exported.module()


class _TorchAOQATInt8Codec:
    """Recreate native Intx int8 QAT topology, convert to IntxUnpackedToInt8Tensor, load state."""

    def save_state(self, model: Any) -> Any:
        return model.state_dict()

    def prepare_model(self, model: Any, config: dict[str, Any], target: str):
        from .quantization import _torchao_api

        if target != "cpu":
            raise ValueError("Native Intx int8 QAT artifacts are CPU-only")
        _torchao_api()
        from torchao.quantization import IntxWeightOnlyConfig, quantize_
        from torchao.quantization.qat import QATConfig as TorchAOQATConfig

        policy = config.get("qat_base_config", {})
        if (
            policy.get("class") != "IntxWeightOnlyConfig"
            or policy.get("weight_dtype") != "torch.int8"
            or policy.get("granularity") != "PerAxis(axis=0)"
            or policy.get("mapping_type") != "MappingType.SYMMETRIC"
            or policy.get("scale_dtype") is not None
            or policy.get("packing") != "IntxPackingFormat.UNPACKED_TO_INT8"
            or policy.get("version") != 2
        ):
            raise ValueError("Unsupported native Intx QAT artifact numerical policy")
        base = IntxWeightOnlyConfig(weight_dtype=torch.int8)
        quantize_(model, TorchAOQATConfig(base, step="prepare"))
        quantize_(model, TorchAOQATConfig(base, step="convert"))
        model.eval()
        return model

    def load_state(self, model: Any, state: Any, config: dict[str, Any]):
        model.load_state_dict(state, strict=True)
        return model


def load_artifact(
    path: str | Path,
    *,
    model_factory: Callable[[], Any],
    map_location: str = "cpu",
    target: str | None = None,
    return_bundle: bool = False,
    expected_preprocessing: Any = None,
) -> Any:
    """Load v1 state-dict bundles and v2 codec-dispatched artifacts."""
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    version = manifest.get("version")
    if manifest.get("format") != FORMAT or version not in (1, VERSION):
        raise ValueError("Unsupported optimization artifact format/version")
    requested_target = target or map_location
    if version == VERSION:
        codec_id = manifest.get("codec")
        if not isinstance(codec_id, str):
            raise ValueError("Version-2 artifact is missing its codec identifier")
        if manifest.get("codec_version") != 1:
            raise ValueError(f"Unsupported codec version {manifest.get('codec_version')!r}")
        allowed_targets = manifest.get("load_targets") or [
            manifest.get("config", {}).get("target", "cpu")
        ]
        if requested_target not in allowed_targets:
            raise ValueError(
                f"Artifact is restricted to load targets {allowed_targets}; requested {requested_target!r}"
            )
        codec = _CODECS.get(codec_id)
        if codec is None and codec_id == "torchao-pt2e-reference-v1":
            codec = _TorchAOReferenceCodec()
        elif codec is None and codec_id == "torchao-qat-int8-unpacked-v1":
            codec = _TorchAOQATInt8Codec()
        elif codec is None and codec_id.startswith("torchao-"):
            codec = _TorchAOCodec()
        elif codec is None and codec_id.startswith("pytorch-dynamic-legacy-"):
            codec = _LegacyDynamicCodec()
        if codec is None and codec_id != "tensor-state-v1":
            raise ValueError(
                f"Missing artifact codec {codec_id!r}; refusing dense fallback or unsafe serialization"
            )
    weights_name = manifest.get("weights")
    if not isinstance(weights_name, str) or Path(weights_name).name != weights_name:
        raise ValueError("Artifact weights path must be a file name inside the artifact directory")
    weights = root / weights_name
    if "sha256" in manifest and _sha256(weights) != manifest["sha256"]:
        raise ValueError("Optimization artifact checksum mismatch")
    if version == VERSION:
        model = model_factory()
        prepare = getattr(codec, "prepare_model", None) if codec is not None else None
        if callable(prepare):
            model = prepare(model, manifest.get("config", {}), requested_target)
    state = torch.load(weights, map_location=map_location, weights_only=True)
    if version == 1:
        model = model_factory()
        model.load_state_dict(state)
        return model
    if codec is not None:
        model = codec.load_state(model, state, manifest.get("config", {}))
    else:
        model.load_state_dict(state, strict=True)
    modes = manifest.get("module_modes", {})
    if callable(getattr(model, "named_modules", None)):
        for name, module in model.named_modules(remove_duplicate=False):
            if name in modes:
                module.training = bool(modes[name])
    prep_payload = manifest.get("preprocessing")
    preprocessing = None
    if prep_payload is not None:
        from topologies.data.preprocessing import PreprocessingPipeline

        preprocessing = PreprocessingPipeline.from_dict(prep_payload)
    if expected_preprocessing is not None:
        expected_fp = expected_preprocessing.to_dict().get("fingerprint")
        found_fp = preprocessing.to_dict().get("fingerprint") if preprocessing is not None else None
        if expected_fp != found_fp:
            raise ValueError("Preprocessing schema/fingerprint mismatch")
    if return_bundle:
        from dataclasses import dataclass

        @dataclass
        class ArtifactBundle:
            model: Any
            preprocessing: Any
            manifest: dict[str, Any]

        return ArtifactBundle(model, preprocessing, manifest)
    return model
