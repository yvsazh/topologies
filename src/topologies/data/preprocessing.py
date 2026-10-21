"""Small explicit-fit preprocessing pipeline foundation."""
import hashlib
import importlib
import json
import random
import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@contextmanager
def _rng_policy(policy):
    if policy == "stateful":
        yield
        return
    if policy != "preserve":
        raise ValueError("rng_policy must be 'preserve' or explicitly 'stateful'")
    import torch
    python_state = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)


@dataclass
class PreprocessingPipeline:
    transforms: list[Callable[[Any], Any]] = field(default_factory=list)
    fitted: bool = False
    identifiers: list[str] = field(default_factory=list)
    mode: str = "inference"
    input_schema: Any = None
    feature_order: list[str] | None = None
    max_fit_items: int = 10000
    rng_policy: str = "preserve"
    cache_config: dict[str, Any] = field(default_factory=dict)

    def fit(self, data: Any, *, mode: str = "training", max_items: int | None = None) -> "PreprocessingPipeline":
        if mode != "training":
            raise ValueError("Preprocessing may only be fitted with mode='training'")
        limit = self.max_fit_items if max_items is None else max_items
        if limit < 1: raise ValueError("max_items must be positive")
        sample = data
        # Preserve array/tensor fit input; bound potentially unending streams.
        if (not isinstance(data, (str, bytes, dict)) and hasattr(data, "__iter__")
                and not hasattr(data, "shape")):
            from itertools import islice
            sample = list(islice(iter(data), limit))
        with _rng_policy(self.rng_policy):
            for index, transform in enumerate(self.transforms):
                applicable = getattr(transform, "modes", None)
                if applicable is not None and "training" not in applicable:
                    continue
                fit = getattr(transform, "fit", None)
                if fit is not None:
                    fit(sample)
                # Each fitted stage observes the output of its predecessors,
                # rather than all stages fitting the original raw feature space.
                if any(callable(getattr(stage, "fit", None)) for stage in self.transforms[index + 1:]):
                    sample = transform.transform(sample) if hasattr(transform, "transform") else transform(sample)
        self.fitted = True
        return self

    def transform(self, value: Any, *, mode: str | None = None, schema: Any = None) -> Any:
        if mode is not None and mode not in {"training", "evaluation", "calibration", "inference"}:
            raise ValueError("mode must be training, evaluation, calibration, or inference")
        if not self.fitted and any(callable(getattr(t, "fit", None)) for t in self.transforms):
            raise RuntimeError("Fitted preprocessing state is required; call fit(training_data) explicitly")
        if schema is not None and self.input_schema is not None and schema != self.input_schema:
            raise ValueError("Preprocessing input schema mismatch")
        active_mode = mode or self.mode
        with _rng_policy(self.rng_policy):
            for transform in self.transforms:
                applicable = getattr(transform, "modes", None)
                if applicable is not None and active_mode not in applicable:
                    continue
                value = transform.transform(value) if hasattr(transform, "transform") else transform(value)
        return value

    def __call__(self, value: Any) -> Any:
        return self.transform(value)

    def to_dict(self) -> dict[str, Any]:
        if len(self.identifiers) != len(self.transforms):
            raise TypeError("Missing preprocessing codec: provide registered IDs or importable function descriptors; lambdas/closures require a custom codec.")
        payload = []
        for identifier, transform in zip(self.identifiers, self.transforms):
            if identifier.startswith("import:"):
                state = _function_descriptor(identifier, transform)
            else:
                state = transform.to_dict() if hasattr(transform, "to_dict") else None
                if state is None: raise TypeError(f"Transform {identifier!r} has no to_dict() serializer; register a codec.")
            payload.append({"id": identifier, "state": state})
        result = {"format": "topologies-preprocessing", "version": 2, "fitted": self.fitted,
                  "mode": self.mode, "input_schema": self.input_schema, "feature_order": self.feature_order,
                  "rng_policy": self.rng_policy, "cache_config": self.cache_config,
                  "transforms": payload}
        result["fingerprint"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreprocessingPipeline":
        if payload.get("format") != "topologies-preprocessing" or payload.get("version") not in (1, 2):
            raise ValueError("Unsupported preprocessing manifest format/version")
        if payload.get("version") == 2:
            expected = payload.get("fingerprint")
            check = dict(payload); check.pop("fingerprint", None)
            actual = hashlib.sha256(json.dumps(check, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if expected != actual: raise ValueError("Preprocessing fingerprint mismatch")
        transforms = []
        identifiers = []
        for item in payload["transforms"]:
            identifier = item["id"]
            if identifier.startswith("import:"):
                transforms.append(_resolve_import(identifier))
                identifiers.append(identifier)
                continue
            if identifier not in _TRANSFORMS:
                raise ValueError(f"Unregistered preprocessing transform {identifier!r}")
            transforms.append(_TRANSFORMS[identifier].from_dict(item["state"]))
            identifiers.append(identifier)
        return cls(transforms, bool(payload.get("fitted")), identifiers,
                   payload.get("mode", "inference"), payload.get("input_schema"), payload.get("feature_order"),
                   rng_policy=payload.get("rng_policy", "preserve"), cache_config=payload.get("cache_config", {}))

    def cache_key(self, *, raw_data_identity: str, sample_identity: str = "all",
                  stochastic_policy: str | None = None) -> str:
        if not raw_data_identity: raise ValueError("raw_data_identity must be explicit")
        manifest=self.to_dict()
        identity={"raw_data_identity": raw_data_identity, "sample_identity": sample_identity,
                  "pipeline_fingerprint": manifest["fingerprint"],
                  "rng_policy": stochastic_policy or self.rng_policy, "cache_config": self.cache_config}
        return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


_TRANSFORMS: dict[str, Any] = {}


def importable_identifier(function: Callable[..., Any]) -> str:
    """Return a stable import-path descriptor; reject closures/lambdas/non-importable callables."""
    module=getattr(function, "__module__", None); qualname=getattr(function, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname or "<lambda>" in qualname:
        raise TypeError("Function is not importable; register a transform codec or use an importable top-level function")
    identifier=f"import:{module}:{qualname}"
    if _resolve_import(identifier) is not function:
        raise TypeError("Function import path does not resolve to the supplied callable")
    return identifier


def _resolve_import(identifier: str) -> Callable[..., Any]:
    _, module_name, qualname = identifier.split(":", 2)
    value=importlib.import_module(module_name)
    for part in qualname.split("."): value=getattr(value, part)
    if not callable(value): raise TypeError(f"Imported preprocessing object is not callable: {identifier}")
    return value


def _function_descriptor(identifier: str, function: Any) -> dict[str, str]:
    if _resolve_import(identifier) is not function:
        raise TypeError(f"Import descriptor does not match transform {identifier!r}")
    return {"import": identifier}


def register_transform(identifier: str, transform_type: Any) -> None:
    if not identifier or identifier in _TRANSFORMS:
        raise ValueError(f"transform identifier empty or already registered: {identifier!r}")
    if not callable(getattr(transform_type, "from_dict", None)):
        raise TypeError("Registered transforms must implement classmethod from_dict(state)")
    _TRANSFORMS[identifier] = transform_type


def save_preprocessing(pipeline: PreprocessingPipeline, path: str) -> str:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(pipeline.to_dict(), stream, indent=2)
    return path


def load_preprocessing(path: str) -> PreprocessingPipeline:
    with open(path, encoding="utf-8") as stream:
        return PreprocessingPipeline.from_dict(json.load(stream))
