"""Payload base class shared by every technique result, plus the result factory.

A payload is a dataclass that implements ``views(**opts) -> list[ViewSpec]`` and
``to_records() -> list[dict]``. JSON serialization goes through
``analysis.register_serializer``. Payloads that contain only summaries
serialize; payloads that hold raw tensors raise ``TypeError`` with the documented
remedy (``trace.save`` or ``payload.to_records()``).
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import torch

RAW_TENSOR_ERROR = ("introspection payload {name} contains raw tensors ({path}); JSON keeps "
                    "summaries only: export rows with payload.to_records(), or save traces with "
                    "trace.save(path)")


class Payload:
    """Mixin for technique payload dataclasses."""

    def views(self, **options: Any) -> list[Any]:
        return []

    def to_records(self) -> list[dict[str, Any]]:
        return []

    def to_json_dict(self) -> dict[str, Any]:
        name = type(self).__name__
        out = {"type": name}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            out[f.name] = _plain(getattr(self, f.name), f"{name}.{f.name}", name)
        return out


def _plain(value: Any, path: str, name: str) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            item = float(value)
            return item if math.isfinite(item) else str(item)
        raise TypeError(RAW_TENSOR_ERROR.format(name=name, path=path))
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v, f"{path}.{k}", name) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v, f"{path}[{i}]", name) for i, v in enumerate(value)]
    if isinstance(value, Payload):
        return value.to_json_dict()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name), f"{path}.{f.name}", name)
                for f in dataclasses.fields(value)}
    if hasattr(value, "value") and type(value).__module__.startswith("topologies.introspection"):
        return str(value.value)
    return str(value)


_REGISTERED = False


def register_payload_serializer() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from topologies.analysis.core import register_serializer

    register_serializer(Payload, "introspection-payload", lambda p: p.to_json_dict(),
                        lambda data: dict(data))
    _REGISTERED = True


def make_result(technique: Any, payload: Any, subject: Any = None, *, kind: str | None = None,
                name: str | None = None, assumptions: list[str] | None = None,
                warnings: list[str] | None = None, provenance: dict[str, Any] | None = None,
                stages: list[Any] | None = None) -> Any:
    """``AnalysisResult`` with the standard introspection provenance block."""
    from topologies.analysis.core import AnalysisResult

    register_payload_serializer()
    prov = {"technique": getattr(technique, "name", str(technique)),
            "tier": getattr(technique, "tier", None)}
    if subject is not None:
        prov["subject"] = subject.describe()
    prov.update(provenance or {})
    base_assumptions = list(getattr(subject, "adapter", None).notes()) if subject is not None and \
        subject.adapter is not None and getattr(subject.adapter, "name", "") != "generic" else []
    return AnalysisResult(payload, kind=kind or getattr(technique, "result_kind", "introspection"),
                          name=name or getattr(technique, "name", None), provenance=prov,
                          warnings=list(dict.fromkeys(warnings or [])),
                          assumptions=list(dict.fromkeys([*base_assumptions, *(assumptions or [])])),
                          stages=list(stages or []))


def input_fingerprint(inputs: Any) -> Any:
    from topologies.introspection import _tree

    if callable(inputs) and not isinstance(inputs, torch.nn.Module):
        return "callable"
    if isinstance(inputs, str):
        return {"text_chars": len(inputs)}
    if isinstance(inputs, (list, tuple)) and inputs and all(isinstance(t, str) for t in inputs):
        return {"texts": len(inputs)}
    try:
        return _tree.describe(inputs)
    except Exception:  # noqa: BLE001
        return type(inputs).__name__


__all__ = ["RAW_TENSOR_ERROR", "Payload", "input_fingerprint", "make_result",
           "register_payload_serializer"]
