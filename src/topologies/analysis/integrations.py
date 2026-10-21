"""Explicit read-only adapters for optional first-party integrations."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from topologies.analysis.core import RecordSet


def from_result_set(results: Any, *, include_history: bool = False,
                    include_series: bool = False) -> RecordSet:
    """Convert ResultSet/TrialRecord records without flattening colliding keys.

    Paths: ``config`` (raw configuration), ``labels`` (sweep labels and seed),
    ``metrics.final/best/extras/build/env``, ``goal``, ``cost``, ``status``,
    ``series.*`` (opt-in), ``history`` (opt-in), and ``source.*`` provenance.
    """
    rows = []
    goal = deepcopy(getattr(results, "goal", (None, None)))
    result_meta = deepcopy(getattr(results, "meta", {}) or {})
    if isinstance(goal, (list, tuple)):
        direction = {"name": goal[0] if goal else None,
                     "mode": goal[1] if len(goal) > 1 else None}
    else:
        direction = deepcopy(goal)
    for trial in results:
        raw = deepcopy(getattr(trial, "raw", {}) or {})
        labels = deepcopy(raw.get("labels") or getattr(trial, "config_labels", {}) or {})
        metrics = {key: deepcopy(raw.get(key) or {})
                   for key in ("final", "best", "extras", "build", "env")}
        row = {"source": {"experiment": raw.get("experiment", getattr(results, "name", None)),
                           "trial_id": raw.get("trial_id"),
                           "spec": result_meta.get("spec"),
                           "result_metadata": result_meta},
               "config": deepcopy(raw.get("config") or {}), "labels": labels,
               "metrics": metrics, "goal": deepcopy(raw.get("goal") or {}),
               "metric_direction": deepcopy(direction), "cost": deepcopy(raw.get("cost") or {}),
               "status": raw.get("status")}
        history = deepcopy(raw.get("history") or [])
        if include_series:
            series: dict[str, list] = {}
            for item in history:
                for key, value in item.items():
                    if key != "epoch":
                        series.setdefault(key, []).append((item.get("epoch"), deepcopy(value)))
            row["series"] = series
        if include_history:
            row["history"] = history
        rows.append(row)
    return RecordSet(rows, source_provenance={"name": getattr(results, "name", None),
                                              "goal": direction, "metadata": result_meta,
                                              "record_count": len(rows)})


def trainer_provenance(trainer: Any) -> dict[str, Any]:
    """Structured description of a Trainer for analysis provenance."""
    config = trainer.config.model_dump(mode="python")
    return {"source": "Trainer", "config": config, "device": str(trainer.device),
            "precision": str(trainer.precision),
            "recipe": getattr(trainer.recipe, "name", type(trainer.recipe).__name__),
            "optimizer_types": {k: type(v).__name__ for k, v in trainer.optimizers.items()}}


def from_trainer(trainer: Any, *, include_history: bool = True,
                 include_model: bool = False) -> Any:
    """Select a Trainer model or copied history records with structured metadata."""
    if include_model:
        return trainer.model
    return RecordSet(deepcopy(trainer.history) if include_history else [],
                     source_provenance=trainer_provenance(trainer))
