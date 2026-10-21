"""Explicit, packaged offline constructor/input fixtures for catalog gates.

These fixtures describe test geometry; they are never public default model or
dataset entries. Decoding is deliberately declarative and does not evaluate code.
"""
from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def catalog_specs(*, include_inactive: bool = False) -> dict[str, dict[str, Any]]:
    specifications = json.loads(files("topologies.validation").joinpath("catalog_specs.json").read_text())["entries"]
    if include_inactive:
        return specifications
    from topologies._catalog import entries

    active = {entry.id for kind in ("model", "layer") for entry in entries(kind).values()}
    return {ident: spec for ident, spec in specifications.items() if ident in active}


def decode(value: Any, *, device: str = "cpu", seed: int = 41) -> Any:
    import torch

    if isinstance(value, list):
        return [decode(item, device=device, seed=seed) for item in value]
    if not isinstance(value, dict):
        return value
    if "tensor" not in value:
        return {key: decode(item, device=device, seed=seed) for key, item in value.items()}
    dtype = getattr(torch, value.get("dtype", "float32"))
    kind = value["tensor"]
    if kind == "values":
        return torch.tensor(value["values"], dtype=dtype, device=device)
    shape = tuple(value["shape"])
    if kind == "full":
        return torch.full(shape, value["value"], dtype=dtype, device=device)
    if device == "meta":
        return torch.empty(shape, dtype=torch.long if kind == "integers" else dtype, device=device)
    generator = torch.Generator().manual_seed(seed)
    if kind == "normal":
        return torch.randn(shape, generator=generator, dtype=dtype).to(device)
    if kind == "uniform":
        return torch.rand(shape, generator=generator, dtype=dtype).to(device)
    if kind == "integers":
        return torch.randint(value.get("low", 0), value["high"], shape, generator=generator).to(device)
    raise ValueError(f"unknown tensor fixture: {kind!r}")


def fixture_inputs(ident: str, *, device: str = "cpu") -> tuple[list[Any], dict[str, Any]]:
    specification = catalog_specs()[ident]
    return decode(specification.get("args", []), device=device), decode(specification.get("kwargs", {}), device=device)


def tensor_leaves(output: Any) -> list[Any]:
    import torch

    if torch.is_tensor(output):
        return [output]
    if isinstance(output, dict):
        return [tensor for value in output.values() for tensor in tensor_leaves(value)]
    if isinstance(output, (tuple, list)):
        return [tensor for value in output for tensor in tensor_leaves(value)]
    return []
