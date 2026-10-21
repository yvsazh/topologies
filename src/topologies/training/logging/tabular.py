"""CSV and TensorBoard metric loggers (rank-0 only, fail-safe via the reporter)."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from topologies.training.errors import TrainingError
from topologies.training.logging.base import BaseLogger


def _scalars(metrics: dict | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in (metrics or {}).items():
        try:
            f = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out[str(k)] = f
    return out


class CSVLogger(BaseLogger):
    """One row per event: ``event,step,epoch,<metric columns...>``.

    Columns grow as new metrics appear (the file is rewritten with the wider
    header), so the CSV always loads cleanly into pandas/spreadsheets.
    """

    def __init__(self, path: str | Path, rank: int = 0) -> None:
        self.rank = rank
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict[str, Any]] = []
        self._columns: list[str] = ["event", "step", "epoch"]

    def _add(self, event: str, step: int, epoch: int | None, metrics: dict) -> None:
        if self.rank != 0:
            return
        row: dict[str, Any] = {"event": event, "step": int(step), "epoch": epoch}
        row.update(_scalars(metrics))
        new_cols = [k for k in row if k not in self._columns]
        self._rows.append(row)
        if new_cols:
            self._columns += new_cols
            self._rewrite()
        else:
            with self.path.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=self._columns).writerow(row)

    def _rewrite(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self._columns)
            w.writeheader()
            w.writerows(self._rows)

    def log(self, step: int, values: dict[str, Any]) -> None:
        self._add("step", step, None, values)

    def on_step(self, event: Any) -> None:
        self._add("step", getattr(event, "global_step", 0), getattr(event, "epoch", None),
                  getattr(event, "metrics", {}))

    def on_validation(self, event: Any) -> None:
        self._add("validation", -1, getattr(event, "epoch", None), getattr(event, "metrics", {}))

    def on_epoch_end(self, event: Any) -> None:
        metrics = {**dict(getattr(event, "train_metrics", {}) or {}),
                   **dict(getattr(event, "val_metrics", {}) or {})}
        self._add("epoch_end", -1, getattr(event, "epoch", None), metrics)


class TensorBoardLogger(BaseLogger):
    """Scalars to TensorBoard (``pip install 'topologies[tensorboard]'``).

    Step metrics go under ``train/``, validation under ``val/``, epoch means
    under ``epoch/``. View with ``tensorboard --logdir <log_dir>``.
    """

    def __init__(self, log_dir: str | Path, rank: int = 0) -> None:
        self.rank = rank
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as e:
            raise TrainingError(
                "logger='tensorboard' needs the 'tensorboard' package.",
                stage="config",
                hint="pip install 'topologies[tensorboard]' (or use logger='jsonl'/'csv').",
            ) from e
        self._writer = SummaryWriter(log_dir=str(log_dir)) if rank == 0 else None
        self._step = 0

    def _scalars(self, prefix: str, metrics: dict, step: int) -> None:
        if self._writer is None:
            return
        for k, v in _scalars(metrics).items():
            name = k.removeprefix(prefix)
            self._writer.add_scalar(f"{prefix.rstrip('_')}/{name}", v, step)

    def log(self, step: int, values: dict[str, Any]) -> None:
        self._step = int(step)
        self._scalars("train_", values, self._step)

    def on_step(self, event: Any) -> None:
        self.log(int(getattr(event, "global_step", 0)), getattr(event, "metrics", {}) or {})

    def on_validation(self, event: Any) -> None:
        self._scalars("val_", getattr(event, "metrics", {}) or {}, self._step)

    def on_epoch_end(self, event: Any) -> None:
        if self._writer is None:
            return
        epoch = int(getattr(event, "epoch", 0))
        for k, v in _scalars(getattr(event, "train_metrics", {})).items():
            self._writer.add_scalar(f"epoch/{k}", v, epoch)
        for k, v in _scalars(getattr(event, "val_metrics", {})).items():
            self._writer.add_scalar(f"epoch/{k}", v, epoch)
        self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
