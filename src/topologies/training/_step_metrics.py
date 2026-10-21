"""Step-metric helpers: on-device accumulation, single host copy, batch utilities."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch

_NOT_AVERAGED = {"loss", "lr", "learning_rate", "grad_norm"}

def _acc_dtype(device: torch.device) -> torch.dtype:
    """float64 for exact-ish sums; MPS has no float64, so float32 there."""
    return torch.float32 if device.type == "mps" else torch.float64

def _floats(values: Mapping[str, Any]) -> dict[str, float]:
    """Python floats for metric values; all tensors cross to the host in **one** copy."""
    out: dict[str, float] = {}
    tensors: list[tuple[str, torch.Tensor]] = []
    for k, v in values.items():
        if isinstance(v, torch.Tensor):
            tensors.append((k, v))
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
    if tensors:
        device = tensors[0][1].device
        stacked = torch.stack([v.detach().to(device=device, dtype=_acc_dtype(device)).reshape(())
                               for _, v in tensors])
        out.update(zip((k for k, _ in tensors), stacked.tolist()))
    return {k: out[k] for k in values if k in out}

class _StepMetrics(MutableMapping):
    """Step metrics that stay on the device until something reads them.

    ``raw`` holds the recipe's values (tensors or floats); the first read
    converts them all with a single device->host copy (cached). Callbacks may
    add or replace entries (``metrics["x"] = 1.0``, ``metrics.update(...)``);
    they flow into the epoch averages and the step events like recipe metrics.
    """

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self._values: dict[str, float] | None = None

    def _floats(self) -> dict[str, float]:
        if self._values is None:
            self._values = _floats(self.raw)
        return self._values

    def __getitem__(self, key: str) -> float:
        return self._floats()[key]

    def __iter__(self):
        return iter(self._floats())

    def __len__(self) -> int:
        return len(self._floats())

    def __setitem__(self, key: str, value: Any) -> None:
        self.raw[key] = value
        self._values = None

    def __delitem__(self, key: str) -> None:
        del self.raw[key]
        self._values = None

def _raw(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    return metrics.raw if isinstance(metrics, _StepMetrics) else metrics

def _loss_value(metrics: Mapping[str, Any]) -> Any:
    """The step loss for epoch sums, kept on the device when it is a tensor."""
    value = _raw(metrics).get("loss", 0.0)
    if isinstance(value, torch.Tensor):
        return value.detach().to(_acc_dtype(value.device)).reshape(())
    return float(value)

def _accumulate(sums: dict[str, Any], metrics: Mapping[str, Any]) -> None:
    """Add step metrics into epoch sums; tensors are summed on the device (no sync)."""
    for k, v in _raw(metrics).items():
        if k in _NOT_AVERAGED:
            continue
        if isinstance(v, torch.Tensor):
            v = v.detach().to(_acc_dtype(v.device)).reshape(())
        else:
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
        sums[k] = sums[k] + v if k in sums else v

def _train_averages(sums: dict[str, Any], n: int) -> dict[str, float]:
    """Epoch means of recipe step metrics, e.g. ``acc`` -> ``train_acc``."""
    return {(k if k.startswith("train_") else f"train_{k}"): v / max(1, n)
            for k, v in _floats(sums).items()}

def _batch_size_of(batch: dict) -> int:
    for v in batch.values():
        if isinstance(v, torch.Tensor) and v.dim() > 0:
            return int(v.shape[0])
    return 1

def _concat_outputs(chunks: list) -> Any:
    """Concatenate per-batch outputs on dim 0 (tensors, tuples, dicts)."""
    if not chunks:
        return None
    first = chunks[0]
    if isinstance(first, torch.Tensor):
        try:
            return torch.cat(chunks, dim=0)
        except RuntimeError:
            return chunks  # ragged (e.g. variable-length sequences)
    if isinstance(first, (tuple, list)):
        return type(first)(_concat_outputs([c[i] for c in chunks]) for i in range(len(first)))
    if isinstance(first, dict):
        return {k: _concat_outputs([c[k] for c in chunks]) for k in first}
    return chunks

def _to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, (tuple, list)):
        return type(obj)(_to_cpu(o) for o in obj)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    return obj
