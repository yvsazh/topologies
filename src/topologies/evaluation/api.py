"""Task-agnostic, stateful evaluation primitives.

Metrics receive records verbatim: this module deliberately does not infer task semantics.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from topologies._execution import ModelCall, model_execution


@dataclass
class EvaluationRecord:
    predictions: Any
    targets: Any = None
    weights: Any = None
    mask: Any = None
    sample_ids: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricValue:
    value: Any = None
    status: str = "ok"
    support: float = 0
    unit: str | None = None
    direction: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    records: int
    slices: dict[str, EvaluationResult] = field(default_factory=dict)
    groups: dict[str, EvaluationResult] = field(default_factory=dict)
    identities: dict[str, str] = field(default_factory=dict)
    cohort_identity: str | None = None
    batches: int = 0


_FACTORIES: dict[str, Callable[..., Any]] = {}
_METADATA: dict[str, dict[str, Any]] = {}


class EvaluationError(ValueError, TypeError):
    # Both bases on purpose: evaluation errors historically surfaced as either
    # ValueError or TypeError, and callers catch both. Kept for compatibility.
    """Actionable evaluation failure with a stable machine-readable code."""

    def __init__(self, message: str, *, code: str, stage: str = "evaluation"):
        super().__init__(message)
        self.code, self.stage = code, stage


@dataclass(frozen=True)
class MetricFactory:
    """Explicit factory wrapper; unlike callable reducers this always creates state."""

    create: Callable[[], Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __call__(self):
        return self.create()


class EvaluationAdapter:
    """Separates model inputs from labels/metadata and selects arbitrary outputs.

    ``input_selector`` returns ``ModelCall``; target/weight/mask/id selectors are
    independent. The optional selector runs once on the model output.
    """

    def __init__(
        self,
        input_selector: Callable[[Any], ModelCall],
        *,
        target_selector: Callable[[Any], Any] | None = None,
        metadata_selector: Callable[[Any], dict[str, Any]] | None = None,
        weights_selector: Callable[[Any], Any] | None = None,
        mask_selector: Callable[[Any], Any] | None = None,
        sample_ids_selector: Callable[[Any], Any] | None = None,
        output_selector: Callable[[Any], Any] | None = None,
    ):
        self.input_selector = input_selector
        self.target_selector = target_selector
        self.metadata_selector = metadata_selector
        self.weights_selector = weights_selector
        self.mask_selector = mask_selector
        self.sample_ids_selector = sample_ids_selector
        self.output_selector = output_selector

    def prepare(self, batch):
        call = self.input_selector(batch)
        metadata = self.metadata_selector(batch) if self.metadata_selector else {}
        metadata = dict(metadata or {})
        if self.weights_selector:
            metadata["weights"] = self.weights_selector(batch)
        if self.mask_selector:
            metadata["mask"] = self.mask_selector(batch)
        if self.sample_ids_selector:
            metadata["sample_ids"] = self.sample_ids_selector(batch)
        target = self.target_selector(batch) if self.target_selector else None
        return call, target, metadata


class ModelAdapterBridge(EvaluationAdapter):
    """Bridge training ``ModelAdapter`` semantics without forwarding labels.

    ``input_keys`` selects raw batch fields before the adapter renames them; its
    input_map/forward_map/output_map is therefore applied exactly once.
    """

    def __init__(self, model_adapter, *, input_keys, target_selector=None, **kwargs):
        from topologies.training.adapters import call_with_adapter

        self.model_adapter = model_adapter
        self.input_keys = tuple(input_keys)
        super().__init__(lambda batch: ModelCall(), target_selector=target_selector, **kwargs)
        self._call_with_adapter = call_with_adapter

    def invoke(self, model, batch, call):
        inputs = {key: batch[key] for key in self.input_keys}
        primary = self.input_keys[0] if len(self.input_keys) == 1 else None
        return self._call_with_adapter(
            self.model_adapter, model, inputs, list(self.input_keys), primary=primary
        )


@dataclass
class _ReducerTotals:
    """Constant-memory sufficient statistics for explicitly declared reducers."""

    total: float = 0.0
    weighted_total: float = 0.0
    weight: float = 0.0
    updates: int = 0

    def append(self, observation):
        value, weight = observation
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Callable reducer weights must be finite and nonnegative")
        if weight == 0:
            return
        self.total += value
        self.weighted_total += value * weight
        self.weight += weight
        self.updates += 1


def register_metric(
    name: str, factory: Callable[..., Any], *, metadata: dict[str, Any] | None = None
) -> None:
    if not name or name in _FACTORIES:
        raise ValueError(f"metric name is empty or already registered: {name!r}")
    _FACTORIES[name], _METADATA[name] = factory, dict(metadata or {})


def metric(name: str, **kwargs: Any) -> Any:
    try:
        value = _FACTORIES[name](**kwargs)
        try:
            value._evaluation_metadata = dict(_METADATA.get(name, {}))
        except (AttributeError, TypeError):
            pass
        return value
    except KeyError as exc:
        raise KeyError(f"Unknown metric {name!r}; registered: {sorted(_FACTORIES)}") from exc


def _state(spec: Any) -> Any:
    if isinstance(spec, MetricFactory):
        return spec.create()
    if isinstance(spec, str):
        return metric(spec)
    if isinstance(spec, type):
        return spec()
    try:
        state = copy.deepcopy(spec)
    except Exception as exc:
        raise TypeError("Metric instances must be deepcopyable or supplied as a factory") from exc
    reset = getattr(state, "reset", None)
    if callable(reset):
        reset()
    return state


def fresh_metric_states(metrics: dict[str, Any]) -> dict[str, Any]:
    return {name: _state(spec) for name, spec in metrics.items()}


def _metric_metadata(name: str, spec: Any) -> dict[str, Any]:
    metadata = dict(_METADATA.get(name, {})) if isinstance(spec, str) else {}
    if isinstance(spec, MetricFactory):
        metadata.update(spec.metadata)
    metadata.update(getattr(spec, "_evaluation_metadata", {}))
    declared = getattr(spec, "metadata", None)
    if isinstance(declared, dict):
        metadata.update(declared)
    return metadata


def _decorate_metric(
    name: str, spec: Any, value: Any, record: EvaluationRecord | None = None
) -> Any:
    metadata = _metric_metadata(name, spec)
    required = set(metadata.get("required_fields", ()))
    if record is not None:
        missing = []
        for field_name in required:
            if field_name.startswith("metadata."):
                found = field_name.split(".", 1)[1] in record.metadata
            else:
                found = (
                    field_name
                    in {"predictions", "targets", "weights", "mask", "sample_ids", "metadata"}
                    and getattr(record, field_name, None) is not None
                )
            if not found:
                missing.append(field_name)
        if missing:
            raise EvaluationError(
                f"Metric {name!r} requires record fields {sorted(missing)}",
                code="metric_input_mismatch",
            )
    if isinstance(value, MetricValue):
        value.unit = metadata.get("unit", value.unit)
        value.direction = metadata.get("direction", value.direction)
        return value
    if metadata.get("unit") or metadata.get("direction"):
        return MetricValue(
            value, "ok", unit=metadata.get("unit"), direction=metadata.get("direction")
        )
    return value


def update_metric(state: Any, record: EvaluationRecord) -> None:
    state.update(record)


def merge_metric(state: Any, distributed: Any) -> None:
    """Merge distributed state using a metric-declared associative protocol."""
    if getattr(distributed, "world_size", 1) <= 1:
        return
    if callable(getattr(state, "merge_state", None)):
        state.merge_state()
    elif callable(getattr(state, "all_reduce", None)):
        state.all_reduce()
    else:
        raise TypeError(f"Metric {type(state).__name__} has no distributed merge protocol")


def _value(state: Any, count: int, *, min_support: int = 1) -> Any:
    if count < min_support:
        return MetricValue(None, "insufficient_support", count)
    result = state.compute()
    if isinstance(result, MetricValue):
        if result.support == 0:
            result.support = count
        return result
    return result


def evaluate(
    model: Any,
    data: Any,
    *,
    metrics: dict[str, Any],
    adapter: Callable[[Any], Any] | None = None,
    slices: dict[str, Callable[[EvaluationRecord], Any]] | None = None,
    group_by: str | list[str] | None = None,
    max_groups: int = 1000,
    min_support: int = 1,
    max_records: int | None = None,
    identities: dict[str, str] | None = None,
    nonpadding: bool = False,
    max_sample_ids: int = 1_000_000,
    _slice_capture: dict[str, list[list[bool]]] | None = None,
    _slice_capture_count: dict[str, int] | None = None,
    _slice_override: dict[str, list[list[bool]]] | None = None,
    _call_callback: Callable[[ModelCall, Any, Any], None] | None = None,
    _record_callback: Callable[[EvaluationRecord], None] | None = None,
) -> EvaluationResult:
    """Evaluate in one pass. Adapter returns ``(ModelCall, target[, metadata])``.

    A callable batch metric must declare ``reducer`` (``mean`` or ``sum``) and a
    record weight/granularity contract; nonlinear scores cannot be averaged implicitly.
    """
    if adapter is None:
        adapter = lambda batch: (ModelCall((batch[0],)), batch[1])
    states = fresh_metric_states(metrics)
    slice_states: dict[str, dict[str, Any]] = {
        name: fresh_metric_states(metrics) for name in (slices or {})
    }
    slice_counts: dict[str, int] = {name: 0 for name in (slices or {})}
    slice_batches = {name: 0 for name in (slices or {})}
    slice_values: dict[str, dict[str, _ReducerTotals]] = {name: {} for name in (slices or {})}
    group_states: dict[str, dict[str, Any]] = {}
    group_counts: dict[str, int] = {}
    group_values: dict[str, dict[str, _ReducerTotals]] = {}
    callable_values = {
        k: _ReducerTotals()
        for k, v in states.items()
        if callable(v) and not callable(getattr(v, "update", None))
    }
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        if not nonpadding:
            raise EvaluationError(
                "distributed evaluation requires a declared non-padding partition (nonpadding=True); sampler padding cannot be safely inferred",
                code="unsafe_distributed_evaluation",
            )
        if group_by or slices:
            raise EvaluationError(
                "distributed standalone evaluation currently requires no dynamic groups/slices",
                code="not_mergeable",
            )
        unsupported = [
            name
            for name, state in states.items()
            if not callable(getattr(state, "all_reduce", None))
            and not callable(getattr(state, "merge_state", None))
        ]
        if unsupported:
            raise EvaluationError(
                f"distributed evaluation metrics lack merge protocol: {unsupported}",
                code="not_mergeable",
            )
        if callable_values:
            raise EvaluationError(
                "callable reducers do not support distributed evaluation; use mergeable metric state",
                code="not_mergeable",
            )
    count = 0
    batches_seen = 0
    seen_ids: set[Any] = set()
    if isinstance(group_by, str):
        group_by = [group_by]
    with model_execution(model):
        for batch in data:
            if max_records is not None and batches_seen >= max_records:
                raise EvaluationError(
                    f"evaluation exceeded max_records={max_records}", code="metric_memory_limit"
                )
            adapted = (
                adapter.prepare(batch) if isinstance(adapter, EvaluationAdapter) else adapter(batch)
            )
            if len(adapted) == 2:
                call, targets = adapted
                metadata = {}
            elif len(adapted) == 3:
                call, targets, metadata = adapted
            else:
                raise EvaluationError(
                    "adapter must return (ModelCall, target[, metadata])",
                    code="metric_input_mismatch",
                )
            if not isinstance(call, ModelCall):
                raise EvaluationError(
                    "adapter's first value must be ModelCall", code="metric_input_mismatch"
                )
            if _call_callback is not None:
                _call_callback(call, targets, metadata)
            prediction = (
                adapter.invoke(model, batch, call)
                if isinstance(adapter, EvaluationAdapter) and hasattr(adapter, "invoke")
                else model(*call.args, **call.kwargs)
            )
            if isinstance(adapter, EvaluationAdapter) and adapter.output_selector:
                prediction = adapter.output_selector(prediction)
            batches_seen += 1
            if isinstance(metadata, dict):
                sample_ids = metadata.get("sample_ids")
                weights, record_mask = metadata.get("weights"), metadata.get("mask")
            else:
                sample_ids, weights, record_mask = None, None, None
            record = EvaluationRecord(
                prediction,
                targets,
                weights=weights,
                mask=record_mask,
                sample_ids=sample_ids,
                metadata=metadata or {},
            )
            record = _deduplicate_record(record, seen_ids, max_sample_ids)
            if record is None:
                continue
            count += _record_support(record)
            if _record_callback is not None:
                _record_callback(record)
            for metric_name, spec in metrics.items():
                _decorate_metric(metric_name, spec, None, record)
            for name, state in states.items():
                if callable(getattr(state, "update", None)):
                    state.update(record)
                elif callable(state):
                    reducer = getattr(state, "reducer", None)
                    if reducer not in ("mean", "sum"):
                        raise EvaluationError(
                            f"Callable metric {name!r} must declare reducer='mean' or 'sum'",
                            code="metric_input_mismatch",
                        )
                    value = state(record)
                    weight = _metric_weight(record)
                    callable_values[name].append((float(value), weight))
                else:
                    raise EvaluationError(
                        f"Metric {name!r} needs update(record)/compute() or declared reducer",
                        code="metric_input_mismatch",
                    )
            for name, predicate in (slices or {}).items():
                batch_index = batches_seen - 1
                if _slice_override is not None:
                    if name not in _slice_override or batch_index >= len(_slice_override[name]):
                        raise EvaluationError(
                            f"missing baseline slice membership for {name!r}, batch {batch_index}",
                            code="cohort_mismatch",
                        )
                    chosen = torch.tensor(_slice_override[name][batch_index], dtype=torch.bool)
                else:
                    chosen = predicate(record)
                n = _sample_count(record)
                if not isinstance(chosen, (bool, int)):
                    member = torch.as_tensor(chosen, dtype=torch.bool)
                    if member.ndim != 1 or member.numel() != n:
                        raise EvaluationError(
                            f"slice {name!r} must return one boolean per sample for paired membership",
                            code="metric_input_mismatch",
                        )
                    chosen = member
                else:
                    chosen = [bool(chosen)] * n
                if _slice_capture is not None:
                    captured = _slice_capture.setdefault(name, [])
                    totals = _slice_capture_count if _slice_capture_count is not None else {}
                    if totals.get(name, 0) + len(chosen) > max_sample_ids:
                        raise EvaluationError(
                            f"slice membership exceeded max_sample_ids={max_sample_ids}",
                            code="metric_memory_limit",
                        )
                    captured.append([bool(x) for x in chosen])
                    totals[name] = totals.get(name, 0) + len(chosen)
                if (
                    bool(chosen)
                    if isinstance(chosen, (bool, int))
                    else bool(torch.as_tensor(chosen).any())
                ):
                    selected = torch.as_tensor(chosen, dtype=torch.bool)
                    sliced = _select_record_rows(
                        record, torch.nonzero(selected, as_tuple=False).flatten().tolist()
                    )
                    for key, state in slice_states[name].items():
                        if callable(getattr(state, "update", None)):
                            state.update(sliced)
                        elif callable(state):
                            fn = state
                            if getattr(fn, "reducer", None) not in ("mean", "sum"):
                                raise TypeError(
                                    f"Callable metric {key!r} must declare reducer='mean' or 'sum'"
                                )
                            wt = _metric_weight(sliced)
                            slice_values[name].setdefault(key, _ReducerTotals()).append(
                                (float(fn(sliced)), wt)
                            )
                    slice_counts[name] += _record_support(sliced)
                    slice_batches[name] += 1
            for field_name in group_by or ():
                if field_name not in record.metadata:
                    raise EvaluationError(
                        f"group_by field {field_name!r} missing from metadata",
                        code="metric_input_mismatch",
                    )
                n = _sample_count(record)
                categories = _group_values(record.metadata[field_name], n)
                try:
                    unique_categories = list(dict.fromkeys(categories))
                except TypeError as exc:
                    raise EvaluationError(
                        "group categories must be hashable scalar values",
                        code="metric_input_mismatch",
                    ) from exc
                for category in unique_categories:
                    indices = [i for i, item in enumerate(categories) if item == category]
                    grouped = _select_record_rows(record, indices)
                    key = f"{field_name}={category}"
                    if key not in group_states:
                        if len(group_states) >= max_groups:
                            raise EvaluationError(
                                f"group cardinality exceeded max_groups={max_groups}",
                                code="metric_memory_limit",
                            )
                        group_states[key] = fresh_metric_states(metrics)
                        group_counts[key] = 0
                        group_values[key] = {}
                    for metric_name, state in group_states[key].items():
                        if callable(getattr(state, "update", None)):
                            state.update(grouped)
                        elif callable(state):
                            if getattr(state, "reducer", None) not in ("mean", "sum"):
                                raise EvaluationError(
                                    f"Callable metric {metric_name!r} must declare reducer='mean' or 'sum'",
                                    code="metric_input_mismatch",
                                )
                            group_values[key].setdefault(metric_name, _ReducerTotals()).append(
                                (float(state(grouped)), _metric_weight(grouped))
                            )
                    group_counts[key] += _record_support(grouped)

    def result(
        states_: dict[str, Any], n: int, callable_map: dict[str, _ReducerTotals] | None = None
    ) -> dict[str, Any]:
        out = {}
        for k, state in states_.items():
            if callable(getattr(state, "update", None)):
                out[k] = _decorate_metric(k, metrics[k], _value(state, n, min_support=min_support))
            else:
                vals = (callable_map or {}).get(k, _ReducerTotals())
                reducer = getattr(state, "reducer", None)
                out[k] = (
                    vals.weighted_total / vals.weight
                    if reducer == "mean" and vals.weight and n >= min_support
                    else vals.total
                    if reducer == "sum" and vals.updates and n >= min_support
                    else MetricValue(None, "insufficient_support", 0)
                )
                out[k] = _decorate_metric(k, metrics[k], out[k])
        return out

    if distributed:
        for state in states.values():
            (
                state.all_reduce
                if callable(getattr(state, "all_reduce", None))
                else state.merge_state
            )()
        counts = torch.tensor([count], dtype=torch.long)
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        count = int(counts.item())
    return EvaluationResult(
        result(states, count, callable_values),
        count,
        {
            k: EvaluationResult(
                result(v, slice_counts[k], slice_values[k]),
                slice_counts[k],
                batches=slice_batches[k],
            )
            for k, v in slice_states.items()
        },
        {
            k: EvaluationResult(result(v, group_counts[k], group_values[k]), group_counts[k])
            for k, v in group_states.items()
        },
        identities=dict(identities or {}),
        cohort_identity=(identities or {}).get("cohort"),
        batches=batches_seen,
    )


def _record_support(record: EvaluationRecord) -> int:
    if record.mask is not None:
        return int(torch.as_tensor(record.mask).sum())
    if record.targets is not None:
        try:
            return int(record.targets.shape[0])
        except (AttributeError, IndexError):
            pass
    return 1


def _metric_weight(record: EvaluationRecord) -> float:
    if record.weights is None:
        return float(_record_support(record))
    weights = torch.as_tensor(record.weights).double()
    if record.mask is not None:
        mask = torch.as_tensor(record.mask).bool()
        if weights.ndim == 1 and mask.ndim > 1 and weights.shape[0] == mask.shape[0]:
            weights = weights.reshape((weights.shape[0],) + (1,) * (mask.ndim - 1))
        weights = torch.broadcast_to(weights, mask.shape) * mask
    return float(weights.sum())


def _sample_count(record: EvaluationRecord) -> int:
    for value in (record.sample_ids, record.targets, record.predictions):
        if isinstance(value, torch.Tensor) and value.ndim:
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    return 1


def _select_rows(value: Any, indices: list[int], count: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim and value.shape[0] == count:
            return value.index_select(
                0, torch.tensor(indices, dtype=torch.long, device=value.device)
            )
        return value
    if isinstance(value, dict):
        return type(value)((k, _select_rows(v, indices, count)) for k, v in value.items())
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_select_rows(v, indices, count) for v in value))
    if isinstance(value, tuple):
        return (
            tuple(value[i] for i in indices)
            if len(value) == count
            else tuple(_select_rows(v, indices, count) for v in value)
        )
    if isinstance(value, list):
        return (
            [value[i] for i in indices]
            if len(value) == count
            and all(not isinstance(v, (dict, list, tuple, torch.Tensor)) for v in value)
            else [_select_rows(v, indices, count) for v in value]
        )
    return value


def _select_record_rows(record: EvaluationRecord, indices: list[int]) -> EvaluationRecord:
    n = _sample_count(record)
    return EvaluationRecord(
        _select_rows(record.predictions, indices, n),
        _select_rows(record.targets, indices, n),
        _select_rows(record.weights, indices, n),
        _select_rows(record.mask, indices, n),
        _select_rows(record.sample_ids, indices, n),
        _select_rows(record.metadata, indices, n),
    )


def _group_values(value: Any, count: int) -> list[Any]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return [value.item()] * count
        if value.shape[0] != count or value.ndim != 1:
            raise EvaluationError(
                "group_by metadata tensor must be scalar or one category per sample",
                code="metric_input_mismatch",
            )
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == count:
            return list(value)
        raise EvaluationError(
            "group_by metadata sequence must have one category per sample",
            code="metric_input_mismatch",
        )
    return [value] * count


def _deduplicate_record(
    record: EvaluationRecord, seen: set[Any], max_ids: int = 1_000_000
) -> EvaluationRecord | None:
    """Remove sampler-padding duplicates when explicit sample IDs are supplied."""
    if record.sample_ids is None:
        return record
    ids = (
        record.sample_ids.reshape(-1).tolist()
        if isinstance(record.sample_ids, torch.Tensor)
        else list(record.sample_ids)
        if isinstance(record.sample_ids, (list, tuple))
        else [record.sample_ids]
    )
    keep = []
    for i, sample_id in enumerate(ids):
        try:
            if sample_id in seen:
                continue
            if len(seen) >= max_ids:
                raise ValueError(f"sample-ID deduplication exceeded max_sample_ids={max_ids}")
            seen.add(sample_id)
        except TypeError as exc:
            raise TypeError("sample IDs must be stable hashable scalar identities") from exc
        keep.append(i)
    if not keep:
        return None
    if len(keep) == len(ids):
        return record
    index = torch.tensor(keep, dtype=torch.long)

    def select(value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == len(ids):
            return value.index_select(0, index.to(value.device))
        if isinstance(value, dict):
            return type(value)((key, select(child)) for key, child in value.items())
        if isinstance(value, (list, tuple)):
            if len(value) == len(ids) and all(
                isinstance(child, (str, int, float, bool)) for child in value
            ):
                return type(value)(value[i] for i in keep)
            if isinstance(value, tuple) and hasattr(value, "_fields"):
                return type(value)(*(select(child) for child in value))
            return type(value)(select(child) for child in value)
        return value

    metadata = {}
    for key, value in record.metadata.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == len(ids):
            metadata[key] = select(value)
        elif isinstance(value, (list, tuple)) and len(value) == len(ids):
            metadata[key] = type(value)(value[i] for i in keep)
        else:
            metadata[key] = value
    return EvaluationRecord(
        select(record.predictions),
        select(record.targets),
        select(record.weights),
        select(record.mask),
        [ids[i] for i in keep],
        metadata,
    )


@dataclass(frozen=True)
class ComparisonPolicy:
    """Decision knobs of ``compare_models`` (grouped; the keyword API is unchanged)."""

    tolerances: dict[str, Any] = field(default_factory=dict)
    direction: dict[str, str] = field(default_factory=dict)
    relative: bool = False
    near_zero: float = 1e-12
    required: set[str] | None = None
    resample_units: str | None = None
    n_resamples: int = 1000
    seed: int = 0
    membership_policy: str = "baseline_fixed"
    unit_aggregation: str | None = None
    max_units: int = 1_000_000
    max_records: int = 100_000


class _PairedComparison:
    """State of one paired comparison: cohort hashes, per-unit scores, slice membership."""

    def __init__(
        self,
        policy: ComparisonPolicy,
        *,
        paired_unit_metrics: dict[str, Callable[[EvaluationRecord], float]] | None,
        call_fingerprint: Callable[[ModelCall, Any, Any], str | bytes] | None,
    ) -> None:
        self.policy = policy
        self.paired_unit_metrics = paired_unit_metrics
        self.call_fingerprint = call_fingerprint
        self.paired: dict[str, dict[str, dict[Any, float]]] = {"baseline": {}, "candidate": {}}
        self.unit_n: dict[str, dict[str, dict[Any, int]]] = {"baseline": {}, "candidate": {}}
        self.unit_total = {"baseline": 0, "candidate": 0}
        self.fixed_memberships: dict[str, list[list[bool]]] = {}
        self.membership_counts: dict[str, int] = {}
        self.cohort_hashes = {name: hashlib.sha256() for name in ("baseline", "candidate")}

    # -- collection -------------------------------------------------------------
    def _observe_call(self, which: str) -> Callable[[ModelCall, Any, Any], None]:
        def observe(call, targets, metadata):
            digest = self.cohort_hashes[which]
            digest.update(b"\x00BATCH\x00")
            if self.call_fingerprint is not None:
                identity = self.call_fingerprint(call, targets, metadata)
                if not isinstance(identity, (str, bytes)):
                    raise TypeError("call_fingerprint must return a stable string or bytes")
                _fingerprint_value(digest, identity)
            else:
                _fingerprint_value(digest, (call.args, call.kwargs, targets, metadata))

        return observe

    def _capture(self, which: str) -> Callable[[EvaluationRecord], None]:
        policy = self.policy

        def callback(record: EvaluationRecord):
            if not self.paired_unit_metrics:
                return
            if not policy.resample_units or policy.resample_units not in record.metadata:
                raise ValueError("resample unit field missing from record metadata")
            raw_units = record.metadata[policy.resample_units]
            units = (
                raw_units.reshape(-1).tolist()
                if isinstance(raw_units, torch.Tensor)
                else list(raw_units)
                if isinstance(raw_units, (list, tuple))
                else [raw_units]
            )
            for metric_name, fn in self.paired_unit_metrics.items():
                value = fn(record)
                if len(units) == 1 and isinstance(value, (int, float)):
                    values = [float(value)]
                elif isinstance(value, (list, tuple, torch.Tensor)) and len(value) == len(units):
                    values = [float(v) for v in value]
                else:
                    raise ValueError(
                        "paired_unit_metrics must return one scalar per declared independent unit"
                    )
                unit_map = self.paired[which].setdefault(metric_name, {})
                counts = self.unit_n[which].setdefault(metric_name, {})
                for unit, score in zip(units, values):
                    if unit in unit_map:
                        if policy.unit_aggregation is None:
                            raise EvaluationError(
                                f"resampling unit {unit!r} repeats; specify unit_aggregation='mean'",
                                code="duplicate_resample_unit",
                            )
                        old_n = counts[unit]
                        unit_map[unit] = (unit_map[unit] * old_n + score) / (old_n + 1)
                        counts[unit] = old_n + 1
                    else:
                        if self.unit_total[which] >= policy.max_units:
                            raise EvaluationError(
                                f"paired bootstrap exceeded max_units={policy.max_units}",
                                code="metric_memory_limit",
                            )
                        unit_map[unit] = score
                        counts[unit] = 1
                        self.unit_total[which] += 1

        return callback

    def _collect(
        self,
        which: str,
        model: Any,
        data: Any,
        *,
        metrics: dict[str, Any],
        adapter: Any,
        slices: Any,
        identities: dict[str, str] | None,
    ) -> EvaluationResult:
        policy = self.policy
        extra: dict[str, Any] = {}
        if which == "baseline":
            extra["_slice_capture"] = self.fixed_memberships if slices else None
            extra["_slice_capture_count"] = self.membership_counts if slices else None
        else:
            extra["_slice_override"] = (
                self.fixed_memberships
                if slices and policy.membership_policy == "baseline_fixed"
                else None
            )
        return evaluate(
            model,
            data,
            metrics=metrics,
            adapter=adapter,
            slices=slices,
            max_records=policy.max_records,
            identities=identities,
            _call_callback=self._observe_call(which),
            _record_callback=self._capture(which) if policy.resample_units else None,
            **extra,
        )

    def check_cohorts(
        self, baseline_result: EvaluationResult, candidate_result: EvaluationResult, slices: Any
    ) -> None:
        if baseline_result.records != candidate_result.records:
            raise ValueError("paired cohort record counts differ between baseline and candidate")
        if slices and self.policy.membership_policy == "baseline_fixed":
            mismatched = [
                name
                for name, members in self.fixed_memberships.items()
                if len(members) != candidate_result.batches
            ]
            if mismatched:
                raise EvaluationError(
                    f"baseline slice membership batch alignment differs for {mismatched}",
                    code="cohort_mismatch",
                )
        if self.cohort_hashes["baseline"].digest() != self.cohort_hashes["candidate"].digest():
            raise ValueError(
                "paired cohort inputs/targets/metadata differ; supply a fixed replayable cohort and deterministic adapter"
            )

    # -- decisions --------------------------------------------------------------
    def _verdict_for(self, name, base_raw, cand_raw, result, identities, *, scope=None):
        policy = self.policy
        budgets, directions, required = policy.tolerances, policy.direction, policy.required
        key = f"{scope}.{name}" if scope is not None else name
        base_status = base_raw.status if isinstance(base_raw, MetricValue) else "ok"
        cand_status = cand_raw.status if isinstance(cand_raw, MetricValue) else "ok"
        base = base_raw.value if isinstance(base_raw, MetricValue) else base_raw
        cand = cand_raw.value if isinstance(cand_raw, MetricValue) else cand_raw
        if (
            base_status != "ok"
            or cand_status != "ok"
            or not isinstance(base, (int, float))
            or not isinstance(cand, (int, float))
            or not math.isfinite(base)
            or not math.isfinite(cand)
        ):
            status, degradation, reason = (
                "indeterminate",
                None,
                f"metric status baseline={base_status}, candidate={cand_status} or value is non-finite",
            )
        else:
            declared_direction = getattr(base_raw, "direction", None) or "max"
            metric_direction = directions.get(key, directions.get(name, declared_direction))
            degradation = base - cand if metric_direction in ("max", "higher") else cand - base
            raw_budget = budgets.get(key, budgets.get(name))
            spec = (
                raw_budget
                if isinstance(raw_budget, dict)
                else {"value": raw_budget, "relative": policy.relative}
            )
            budget = spec.get("value")
            use_relative = bool(spec.get("relative", False))
            if budget is None:
                status, reason = "indeterminate", "no tolerance supplied"
            elif use_relative and abs(base) <= policy.near_zero:
                status, reason = "indeterminate", "baseline is near zero for relative tolerance"
            else:
                limit = float(budget) * (abs(base) if use_relative else 1)
                status, reason = (
                    ("pass", "within tolerance")
                    if degradation <= limit
                    else ("fail", "tolerance exceeded")
                )
        # Legacy custom scalar metrics describe batch records; rich metrics declare
        # their own sample/token/event denominator explicitly.
        support = cand_raw.support if isinstance(cand_raw, MetricValue) else result.batches
        return {
            "baseline": base,
            "candidate": cand,
            "degradation": degradation,
            "tolerance": budgets.get(key, budgets.get(name)),
            "support": support,
            "support_unit": "declared_observations"
            if isinstance(cand_raw, MetricValue)
            else "batch",
            "status": status,
            "reason": reason,
            "required": required is None
            or key in required
            or name in required
            or (scope is None and f"aggregate.{name}" in required),
            "identities": dict(identities or {}),
        }

    def _verdicts(
        self,
        baseline_result: EvaluationResult,
        candidate_result: EvaluationResult,
        identities: dict[str, str],
    ) -> tuple[dict, dict, str]:
        verdict = {
            name: self._verdict_for(
                name, value, candidate_result.metrics.get(name), candidate_result, identities
            )
            for name, value in baseline_result.metrics.items()
        }
        slice_verdicts: dict[str, dict[str, Any]] = {}
        for slice_name, base_slice in baseline_result.slices.items():
            cand_slice = candidate_result.slices.get(slice_name)
            if cand_slice is None:
                continue
            slice_verdicts[slice_name] = {
                name: self._verdict_for(
                    name,
                    value,
                    cand_slice.metrics.get(name),
                    cand_slice,
                    identities,
                    scope=slice_name,
                )
                for name, value in base_slice.metrics.items()
            }
        all_verdicts = list(verdict.values()) + [
            v for group in slice_verdicts.values() for v in group.values()
        ]
        required_statuses = [v["status"] for v in all_verdicts if v["required"]]
        combined = (
            "fail"
            if "fail" in required_statuses
            else "indeterminate"
            if "indeterminate" in required_statuses
            else "pass"
        )
        return verdict, slice_verdicts, combined

    def _bootstrap(self) -> dict[str, Any]:
        policy = self.policy
        intervals: dict[str, Any] = {}
        if policy.resample_units is None:
            return intervals
        if policy.n_resamples < 1:
            raise ValueError("n_resamples must be positive")
        rng = __import__("random").Random(policy.seed)
        for name in self.paired_unit_metrics or {}:
            left = self.paired["baseline"].get(name, {})
            right = self.paired["candidate"].get(name, {})
            units = sorted(set(left) & set(right), key=repr)
            if not units:
                intervals[name] = {
                    "status": "indeterminate",
                    "reason": "no matched independent units",
                }
                continue
            diffs = [
                (left[u] - right[u])
                if policy.direction.get(name, "max") in ("max", "higher")
                else (right[u] - left[u])
                for u in units
            ]
            means = []
            for _ in range(policy.n_resamples):
                means.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
            means.sort()
            intervals[name] = {
                "status": "ok",
                "unit": policy.resample_units,
                "units": len(units),
                "confidence": 0.95,
                "degradation_ci": [
                    means[int(0.025 * (len(means) - 1))],
                    means[int(0.975 * (len(means) - 1))],
                ],
                "decision_rule": "descriptive paired bootstrap; does not establish equivalence",
            }
        return intervals


def compare_models(
    baseline: Any,
    candidate: Any,
    data: Any,
    *,
    metrics: dict[str, Any],
    adapter: Callable[[Any], Any] | None = None,
    tolerances: dict[str, Any] | None = None,
    direction: dict[str, str] | None = None,
    slices: dict[str, Callable[[EvaluationRecord], Any]] | None = None,
    relative: bool = False,
    near_zero: float = 1e-12,
    identities: dict[str, str] | None = None,
    required: set[str] | None = None,
    resample_units: str | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
    reset: Callable[[Any], None] | None = None,
    paired_unit_metrics: dict[str, Callable[[EvaluationRecord], float]] | None = None,
    call_fingerprint: Callable[[ModelCall, Any, Any], str | bytes] | None = None,
    membership_policy: str = "baseline_fixed",
    unit_aggregation: str | None = None,
    max_units: int = 1_000_000,
    max_records: int = 100_000,
) -> dict[str, Any]:
    """Compare a bounded replayable cohort. Iterators must be supplied as factories.

    ``resample_units`` declares an application-provided independent unit in record metadata;
    bootstrap intervals are deliberately not inferred from token/frame rows.
    """
    if (
        getattr(baseline, "stateful", False) or getattr(candidate, "stateful", False)
    ) and reset is None:
        raise ValueError("stateful paired comparison requires explicit reset(model)")
    if membership_policy not in {"baseline_fixed", "candidate_recomputed"}:
        raise ValueError("membership_policy must be baseline_fixed or candidate_recomputed")
    if unit_aggregation not in {None, "mean"}:
        raise ValueError("unit_aggregation must be None or 'mean'")
    if callable(data):
        baseline_data, candidate_data = data(), data()
    else:
        if iter(data) is data:
            raise ValueError(
                "paired comparison needs replayable data or a data factory; iterator materialization is not implicit"
            )
        try:
            size = len(data)
        except TypeError:
            size = None
        if size is not None and size > max_records:
            raise ValueError(f"paired cohort exceeds max_records={max_records}")
        baseline_data = candidate_data = data
    policy = ComparisonPolicy(
        tolerances=dict(tolerances or {}),
        direction=dict(direction or {}),
        relative=relative,
        near_zero=near_zero,
        required=required,
        resample_units=resample_units,
        n_resamples=n_resamples,
        seed=seed,
        membership_policy=membership_policy,
        unit_aggregation=unit_aggregation,
        max_units=max_units,
        max_records=max_records,
    )
    comparison = _PairedComparison(
        policy, paired_unit_metrics=paired_unit_metrics, call_fingerprint=call_fingerprint
    )
    if resample_units is not None and not paired_unit_metrics:
        raise ValueError(
            "paired bootstrap requires paired_unit_metrics mapping each metric to a per-unit scalar"
        )
    if reset:
        reset(baseline)
    common = {"metrics": metrics, "adapter": adapter, "slices": slices, "identities": identities}
    baseline_result = comparison._collect("baseline", baseline, baseline_data, **common)
    if reset:
        reset(candidate)
    candidate_result = comparison._collect("candidate", candidate, candidate_data, **common)
    comparison.check_cohorts(baseline_result, candidate_result, slices)
    identities = dict(identities or {})
    identities["observed_cohort_sha256"] = comparison.cohort_hashes["baseline"].hexdigest()
    baseline_result.identities.update(identities)
    candidate_result.identities.update(identities)
    verdict, slice_verdicts, combined = comparison._verdicts(
        baseline_result, candidate_result, identities
    )
    intervals = comparison._bootstrap()
    return {
        "status": combined,
        "metrics": verdict,
        "slices": slice_verdicts,
        "baseline": baseline_result,
        "candidate": candidate_result,
        "identities": dict(identities or {}),
        "paired_intervals": intervals,
        "slice_membership_policy": membership_policy,
    }


def _fingerprint_value(digest, value):
    """Canonical cohort evidence without serializing arbitrary Python objects."""
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError("Non-dense cohort tensors require explicit call_fingerprint")
        _fingerprint_value(digest, (str(value.dtype), tuple(value.shape)))
        backing = value.detach().cpu().contiguous().clone()
        raw = bytes(backing.untyped_storage())
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    elif isinstance(value, dict):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda k: (type(k).__name__, repr(k))):
            _fingerprint_value(digest, key)
            _fingerprint_value(digest, value[key])
        digest.update(b"end")
    elif isinstance(value, (tuple, list)):
        digest.update(type(value).__name__.encode())
        digest.update(len(value).to_bytes(8, "big"))
        for child in value:
            _fingerprint_value(digest, child)
    elif value is None or isinstance(value, (str, int, float, bool, bytes)):
        raw = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
        digest.update(type(value).__name__.encode())
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    else:
        raise TypeError(
            f"Cohort field of type {type(value).__name__} requires explicit call_fingerprint"
        )
