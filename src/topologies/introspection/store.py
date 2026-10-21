"""Trace persistence: ``manifest.json`` + ``tensors/*.pt`` shards (format ``topologies-trace`` v1).

Shards are written with ``torch.save`` and read with ``weights_only=True``; no
dependency beyond torch. A safetensors backend may be added behind the same
functions later.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from topologies.introspection import _tree
from topologies.introspection.semantics import Axes
from topologies.introspection.sites import SiteKey
from topologies.introspection.trace import CallRecord, CallTree, SiteRecord, Trace, _jsonable

FORMAT = "topologies-trace"
VERSION = 1


def model_fingerprint(model: nn.Module) -> dict[str, Any]:
    """Class path + parameter count + sha256 of the first 1 MiB of state-dict bytes."""
    cls = type(model)
    digest = hashlib.sha256()
    remaining = 2**20
    for name, tensor in model.state_dict().items():
        if remaining <= 0:
            break
        digest.update(name.encode())
        data = tensor.detach().cpu().contiguous().reshape(-1)
        size = min(data.numel() * data.element_size(), remaining)
        chunk = ctypes.string_at(data.data_ptr(), size) if size else b""
        digest.update(chunk)
        remaining -= len(chunk)
    return {"class": f"{cls.__module__}.{cls.__qualname__}",
            "parameters": sum(p.numel() for p in model.parameters()),
            "sha256_first_mib": digest.hexdigest()}


def _flatten_output(output: Any) -> dict[str, torch.Tensor]:
    return {path or "": t.detach().cpu() for path, t in _tree.leaves(output)}


def save_trace(trace: Trace, path: str | Path, *, model: nn.Module | None = None,
               format: str = "torch") -> Path:
    if format not in ("torch", "safetensors"):
        raise ValueError("format must be 'torch' or 'safetensors'")
    out = Path(path)
    (out / "tensors").mkdir(parents=True, exist_ok=True)
    records = []
    bundle: dict[str, torch.Tensor] = {}
    for i, (key, record) in enumerate(trace.records.items()):
        entry = {"key": [key.path, key.port, key.occurrence, key.op, key.op_index],
                 **record.to_dict()}
        has_value = record.value is not None or record.loader is not None
        if has_value:
            tensor = record.tensor.detach().cpu().contiguous()
            if format == "safetensors":
                entry["tensor_key"] = f"{i:06d}"
                bundle[entry["tensor_key"]] = tensor
            else:
                shard = f"tensors/{i:06d}.pt"
                torch.save({"value": tensor}, out / shard)
                entry["shard"] = shard
        records.append(entry)
    if format == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError("format='safetensors' needs the optional safetensors package") from exc
        save_file(bundle, str(out / "tensors.safetensors"))
    leaves = _flatten_output(trace.output)
    if leaves:
        torch.save(leaves, out / "tensors" / "output.pt")
    fingerprint = model_fingerprint(model) if model is not None else trace.provenance.get(
        "fingerprint")
    manifest = {"format": FORMAT, "version": VERSION, "storage": format, "records": records,
                "calls": trace.calls.to_dict(), "policy": _policy_dict(trace.policy),
                "provenance": _jsonable(trace.provenance), "fingerprint": fingerprint,
                "warnings": trace.warnings, "assumptions": trace.assumptions,
                "metadata": _jsonable(trace.metadata),
                "output_structure": _jsonable(_tree.describe(trace.output))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=True))
    return out


def _policy_dict(policy: Any) -> Any:
    return policy.to_dict() if hasattr(policy, "to_dict") else _jsonable(policy)


def load_trace(path: str | Path) -> Trace:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{root} is not a {FORMAT} directory")
    if int(manifest.get("version", 0)) > VERSION:
        raise ValueError(f"trace format version {manifest['version']} is newer than {VERSION}")
    records: dict[SiteKey, SiteRecord] = {}
    bundle: dict[str, torch.Tensor] = {}
    if manifest.get("storage") == "safetensors":
        from safetensors.torch import load_file

        bundle = load_file(str(root / "tensors.safetensors"))
    for entry in manifest["records"]:
        key = SiteKey(*entry["key"])
        value = None
        if "shard" in entry:
            value = torch.load(root / entry["shard"], weights_only=True)["value"]
        elif "tensor_key" in entry:
            value = bundle[entry["tensor_key"]]
        axes = Axes.from_dict(entry["axes"]) if entry.get("axes") else None
        records[key] = SiteRecord(entry["site"], entry["port"], entry["occurrence"],
                                  tuple(entry["shape"]), entry["dtype"], entry["device"], axes,
                                  value, entry.get("summary"), None, key, dict(entry.get("meta", {})))
    calls = CallTree([CallRecord(c["index"], "" if c["path"] == "<root>" else c["path"],
                                 c["occurrence"], c["parent"], c["depth"], c["type"],
                                 c.get("inputs"), c.get("outputs"), tuple(c.get("dtypes", ())),
                                 tuple(c.get("devices", ())), c.get("duration"), c.get("error", False))
                      for c in manifest["calls"]["calls"]],
                     tracked=tuple(manifest["calls"].get("tracked", ())))
    output = None
    output_path = root / "tensors" / "output.pt"
    if output_path.exists():
        leaves = torch.load(output_path, weights_only=True)
        output = leaves.get("") if set(leaves) == {""} else leaves
    from topologies.introspection.capture import CapturePolicy

    policy_values = manifest.get("policy", {})
    try:
        policy = CapturePolicy.build(None, **{k: (tuple(v) if isinstance(v, list) else v)
                                               for k, v in policy_values.items()
                                               if k in ("keep", "ports", "occurrences", "store_device",
                                                        "max_bytes", "clone", "grads", "timing",
                                                        "training", "seed", "max_sites")})
    except (TypeError, ValueError):
        policy = policy_values
    provenance = dict(manifest.get("provenance", {}))
    provenance["fingerprint"] = manifest.get("fingerprint")
    provenance["loaded_from"] = str(root)
    return Trace(output, calls, records, policy, warnings=manifest.get("warnings", []),
                 assumptions=manifest.get("assumptions", []), metadata=manifest.get("metadata", {}),
                 provenance=provenance)
