"""Explicit, validated ordered optimization pipelines.

Validation is a planning operation: all stage transitions are checked before the
first stage can inspect, calibrate, or mutate a candidate.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .contracts import OptimizationError


def _stage_value(value: Any) -> Any:
    """Pass an OptimizationResult's executable representation into the next stage."""
    from .contracts import OptimizationResult as BackendResult

    if isinstance(value, BackendResult):
        if value.runtime is not None:
            return value.runtime
        return value.model
    return value


@dataclass
class OptimizationContext:
    target: Any = None
    seed: int = 0
    source_identity: str = "unknown"
    data_identity: str = "unknown"
    preprocessing_identity: str = "unknown"
    backend_versions: dict[str, str] = field(default_factory=dict)
    workload_identity: str = "unknown"
    cohort_identity: str = "unknown"
    artifact_store: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TransformationStage(Protocol):
    name: str
    prerequisites: frozenset[str]
    changes_topology: bool
    replaces_parameters: bool
    requires_calibration: bool
    requires_training: bool
    output_representation: str
    codec: Any
    factory_id: str

    def inspect(self, model: Any, context: OptimizationContext) -> Any: ...
    def apply(self, model: Any, context: OptimizationContext) -> Any: ...


@dataclass
class FunctionalStage:
    name: str
    apply_fn: Callable[[Any, OptimizationContext], Any]
    inspect_fn: Callable[[Any, OptimizationContext], Any] = lambda _m, _c: None
    prerequisites: frozenset[str] = frozenset({"dense"})
    changes_topology: bool = False
    replaces_parameters: bool = False
    requires_calibration: bool = False
    requires_training: bool = False
    output_representation: str = "dense"
    backend: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    calibration_fresh: bool = False
    sparse_capabilities: frozenset[str] = frozenset()
    codec: Any = None
    factory_id: str = ""
    calibration_replayable: bool = True

    def inspect(self, model: Any, context: OptimizationContext) -> Any:
        return self.inspect_fn(model, context)

    def apply(self, model: Any, context: OptimizationContext) -> Any:
        return self.apply_fn(model, context)


@dataclass
class StageRecord:
    name: str
    input_representation: str
    output_representation: str
    provenance: str
    artifact: Any = None
    report: Any = None


@dataclass
class OptimizationResult:
    model: Any
    history: list[StageRecord]
    provenance: str
    representation: str


def _stable_hash(value: Any) -> str:
    def default(obj: Any) -> Any:
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return repr(obj)

    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=default)
    return hashlib.sha256(raw.encode()).hexdigest()


class OptimizationPipeline:
    def __init__(self, stages: list[TransformationStage] | tuple[TransformationStage, ...]):
        self.stages = tuple(stages)

    def validate(self, *, initial_representation: str = "dense") -> tuple[str, ...]:
        rep = initial_representation
        history: list[str] = []
        fresh_calibration = False
        for stage in self.stages:
            if rep == "converted" and stage.requires_training:
                raise OptimizationError(
                    f"{stage.name} cannot train a converted inference model",
                    code="invalid_stage_order",
                    stage=stage.name,
                )
            if rep not in stage.prerequisites:
                raise OptimizationError(
                    f"{stage.name} requires {sorted(stage.prerequisites)}, got {rep}",
                    code="invalid_stage_order",
                    stage=stage.name,
                )
            if (
                rep == "structurally_pruned"
                and stage.requires_calibration
                and not fresh_calibration
            ):
                raise OptimizationError(
                    f"{stage.name} requires fresh calibration after structural pruning",
                    code="invalid_stage_order",
                    stage=stage.name,
                )
            if rep in {"masked", "sparse"} and stage.output_representation in {
                "sparse",
                "converted",
            }:
                capability = f"{rep}+{stage.output_representation}"
                if capability not in stage.sparse_capabilities:
                    raise OptimizationError(
                        f"{stage.name} backend does not declare {capability} support",
                        code="invalid_stage_order",
                        stage=stage.name,
                    )
            rep = stage.output_representation
            fresh_calibration = bool(stage.calibration_fresh)
            history.append(rep)
        return tuple(history)

    def run(
        self, model: Any, context: OptimizationContext, *, initial_representation: str = "dense"
    ) -> OptimizationResult:
        outputs = self.validate(initial_representation=initial_representation)
        identity = _stable_hash(
            {
                "stages": [
                    {
                        "name": s.name,
                        "config": s.config,
                        "backend": s.backend,
                        "output": s.output_representation,
                    }
                    for s in self.stages
                ],
                "source": context.source_identity,
                "data": context.data_identity,
                "preprocessing": context.preprocessing_identity,
                "cohort": context.cohort_identity,
                "backend_versions": context.backend_versions,
                "target": context.target,
                "workload": context.workload_identity,
            }
        )
        import random

        random.seed(context.seed)
        try:
            import numpy as np

            np.random.seed(context.seed)
        except ImportError:
            pass
        records: list[StageRecord] = []
        current = model
        rep = initial_representation
        for stage, out_rep in zip(self.stages, outputs):
            report = stage.inspect(current, context)
            changed = stage.apply(current, context)
            artifact = None
            if isinstance(changed, tuple) and len(changed) == 2:
                current, artifact = changed
            else:
                current = changed
            current = _stage_value(current)
            records.append(StageRecord(stage.name, rep, out_rep, identity, artifact, report))
            rep = out_rep
        return OptimizationResult(current, records, identity, rep)

    def _signature(self, context: OptimizationContext, initial_representation: str) -> str:
        self.validate(initial_representation=initial_representation)
        return _stable_hash(
            {
                "stages": [
                    {
                        "name": s.name,
                        "factory_id": getattr(s, "factory_id", ""),
                        "config": s.config,
                        "backend": s.backend,
                        "output": s.output_representation,
                        "topology": s.changes_topology,
                        "parameters": s.replaces_parameters,
                        "calibration": s.requires_calibration,
                    }
                    for s in self.stages
                ],
                "source": context.source_identity,
                "data": context.data_identity,
                "preprocessing": context.preprocessing_identity,
                "cohort": context.cohort_identity,
                "backend_versions": context.backend_versions,
                "target": context.target,
                "workload": context.workload_identity,
                "initial": initial_representation,
            }
        )

    def run_durable(
        self,
        model: Any,
        context: OptimizationContext,
        directory: str | Path,
        *,
        initial_representation: str = "dense",
    ) -> OptimizationResult:
        """Run stages, publishing each completed boundary atomically via stage codecs.

        A codec implements ``save(model, path)`` and ``load(path)``. The manifest
        is committed last; incomplete payloads are therefore never resumable.
        """
        outputs = self.validate(initial_representation=initial_representation)
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        signature = self._signature(context, initial_representation)
        history: list[StageRecord] = []
        entries: list[dict[str, Any]] = []
        current = model
        rep = initial_representation
        for index, (stage, out_rep) in enumerate(zip(self.stages, outputs)):
            report = stage.inspect(current, context)
            changed = stage.apply(current, context)
            artifact = None
            if isinstance(changed, tuple) and len(changed) == 2:
                current, artifact = changed
            else:
                current = changed
            current = _stage_value(current)
            codec = getattr(stage, "codec", None)
            if codec is None:
                raise OptimizationError(
                    f"stage {stage.name} needs a codec for durable boundaries",
                    code="missing_artifact_codec",
                    stage=stage.name,
                )
            filename = f"stage-{index:04d}.payload"
            tmp = root / (filename + ".tmp")
            codec.save(current, tmp)
            with tmp.open("rb") as stream:
                os.fsync(stream.fileno())
            digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
            os.replace(tmp, root / filename)
            entry = {
                "index": index,
                "name": stage.name,
                "representation": out_rep,
                "file": filename,
                "sha256": digest,
                "codec": getattr(codec, "codec_id", "unknown"),
                "factory_id": getattr(stage, "factory_id", ""),
            }
            history.append(StageRecord(stage.name, rep, out_rep, signature, artifact, report))
            manifest = {
                "format": "topologies-optimization-pipeline",
                "version": 1,
                "signature": signature,
                "completed": index + 1,
                "boundaries": [*entries, entry],
            }
            entries.append(entry)
            temp_manifest = root / "manifest.json.tmp"
            with temp_manifest.open("w", encoding="utf8") as stream:
                json.dump(manifest, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_manifest, root / "manifest.json")
            rep = out_rep
        return OptimizationResult(current, history, signature, rep)

    def resume(
        self,
        context: OptimizationContext,
        directory: str | Path,
        *,
        source_factory: Callable[[], Any],
        initial_representation: str = "dense",
    ) -> OptimizationResult:
        """Restore the last verified completed boundary, validating identities first."""
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
        expected = self._signature(context, initial_representation)
        if (
            manifest.get("format") != "topologies-optimization-pipeline"
            or manifest.get("version") != 1
            or manifest.get("signature") != expected
        ):
            raise OptimizationError("pipeline resume identity mismatch", code="incompatible_resume")
        completed = int(manifest.get("completed", 0))
        boundaries = manifest.get("boundaries", [])
        if completed < 1 or completed > len(self.stages) or len(boundaries) != completed:
            raise OptimizationError("invalid pipeline boundary count", code="incompatible_resume")
        stage = self.stages[completed - 1]
        if getattr(stage, "requires_calibration", False) and not getattr(
            stage, "calibration_replayable", True
        ):
            raise OptimizationError(
                "cannot resume one-shot calibration without explicit replay support",
                code="incompatible_resume",
                stage=stage.name,
            )
        entry = boundaries[-1]
        if entry.get("index") != completed - 1 or entry.get("name") != stage.name:
            raise OptimizationError("pipeline stage boundary mismatch", code="incompatible_resume")
        payload = root / entry["file"]
        digest = hashlib.sha256(payload.read_bytes()).hexdigest()
        if digest != entry.get("sha256"):
            raise OptimizationError(
                "pipeline checkpoint checksum mismatch", code="incompatible_resume"
            )
        codec = getattr(stage, "codec", None)
        if codec is None or getattr(codec, "codec_id", "unknown") != entry.get("codec"):
            raise OptimizationError(
                "pipeline checkpoint codec mismatch", code="incompatible_resume"
            )
        # Identity and checksum checks precede both factory invocation and decoding.
        source_factory()  # invoked only after the identity checks above (resume contract)
        restored = codec.load(payload)
        records = [
            StageRecord(
                s.name,
                "unknown",
                s.output_representation,
                expected,
                artifact=boundaries[i].get("file"),
            )
            for i, s in enumerate(self.stages[:completed])
        ]
        return OptimizationResult(restored, records, expected, stage.output_representation)
