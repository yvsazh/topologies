"""JSONL event-stream logger (raw metrics only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from topologies.training.logging.base import BaseLogger


def _safe_float(v: Any) -> Any:
    try:
        import math as _math

        f = float(v)
        if _math.isnan(f) or _math.isinf(f):
            return str(v)
        return f
    except Exception:  # noqa: BLE001
        return v


def _clean_metrics(metrics: dict | None) -> dict:
    out: dict[str, Any] = {}
    for k, v in (metrics or {}).items():
        try:
            if isinstance(v, float):
                import math as _math

                if _math.isnan(v) or _math.isinf(v):
                    out[str(k)] = str(v)
                else:
                    out[str(k)] = v
            elif isinstance(v, (int, str, bool)) or v is None:
                out[str(k)] = v
            else:
                try:
                    out[str(k)] = float(v)  # type: ignore[arg-type]
                    out[str(k)] = _safe_float(out[str(k)])
                except Exception:  # noqa: BLE001
                    out[str(k)] = str(v)
        except Exception:  # noqa: BLE001, S112
            continue
    return out


class JsonlLogger(BaseLogger):
    """Writes one JSON object per line. Dependency-free."""

    def __init__(self, path: str | Path, rank: int = 0) -> None:
        # Duck-type BaseLogger without importing to keep modules trivial.
        self.rank = rank
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    # -- legacy -------------------------------------------------------------
    def log(self, step: int, values: dict[str, Any]) -> None:
        if self.rank != 0:
            return
        rec = {"event": "step", "step": int(step), **_clean_metrics(values)}
        self._write(rec)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001, S110
            pass

    # -- events ---------------------------------------------------------------
    def on_train_begin(self, event: Any) -> None:
        if self.rank != 0:
            return
        info = getattr(event, "info", None)
        rec: dict[str, Any] = {"event": "train_begin"}
        try:
            if info is not None:
                if getattr(info, "model_name", None) is not None:
                    rec["model"] = info.model_name
                if getattr(info, "architecture_id", None) is not None:
                    rec["architecture_id"] = info.architecture_id
                if getattr(info, "param_total", None) is not None:
                    rec["params"] = info.param_total
                if getattr(info, "recipe_name", None) is not None:
                    rec["recipe"] = info.recipe_name
                if getattr(info, "device", None) is not None:
                    rec["device"] = info.device
                if getattr(info, "max_epochs", None) is not None:
                    rec["max_epochs"] = info.max_epochs
                if getattr(info, "max_steps", None) is not None:
                    rec["max_steps"] = info.max_steps
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            total = getattr(event, "total_steps", None)
            if total is not None:
                rec["total_steps"] = int(total)
        except Exception:  # noqa: BLE001, S110
            pass
        self._write(rec)

    def on_epoch_begin(self, event: Any) -> None:
        if self.rank != 0:
            return
        try:
            self._write(
                {
                    "event": "epoch_begin",
                    "epoch": int(getattr(event, "epoch", 0)),
                    "total_epochs": int(getattr(event, "total_epochs", 0)),
                }
            )
        except Exception:  # noqa: BLE001, S110
            pass

    def on_step(self, event: Any) -> None:
        if self.rank != 0:
            return
        try:
            rec = {
                "event": "step",
                "step": int(getattr(event, "global_step", 0)),
                "epoch": int(getattr(event, "epoch", 0)),
                **_clean_metrics(getattr(event, "metrics", {}) or {}),
            }
            self._write(rec)
        except Exception:  # noqa: BLE001, S110
            pass

    def on_validation_begin(self, event: Any) -> None:
        pass

    def on_validation(self, event: Any) -> None:
        if self.rank != 0:
            return
        try:
            rec = {
                "event": "validation",
                "epoch": int(getattr(event, "epoch", 0)),
                **_clean_metrics(getattr(event, "metrics", {}) or {}),
            }
            self._write(rec)
        except Exception:  # noqa: BLE001, S110
            pass

    def on_validation_end(self, event: Any) -> None:
        self.on_validation(event)

    def on_checkpoint(self, event: Any) -> None:
        if self.rank != 0:
            return
        try:
            self._write(
                {
                    "event": "checkpoint",
                    "path": str(getattr(event, "path", "")),
                    "epoch": int(getattr(event, "epoch", 0)),
                    "step": int(getattr(event, "global_step", 0)),
                    "is_best": bool(getattr(event, "is_best", False)),
                }
            )
        except Exception:  # noqa: BLE001, S110
            pass

    def on_epoch_end(self, event: Any) -> None:
        if self.rank != 0:
            return
        try:
            rec: dict[str, Any] = {
                "event": "epoch_end",
                "epoch": int(getattr(event, "epoch", 0)),
            }
            rec.update(_clean_metrics(getattr(event, "train_metrics", {}) or {}))
            # Prefix val metrics with val_ only if not already prefixed? Keep raw
            # keys as stored in history: history uses train_loss/val_loss already.
            # Merge both dicts raw.
            rec.update(_clean_metrics(getattr(event, "val_metrics", {}) or {}))
            if getattr(event, "best_metric", None) is not None:
                rec["best_metric"] = event.best_metric
            self._write(rec)
        except Exception:  # noqa: BLE001, S110
            pass

    def on_train_end(self, event: Any) -> None:
        if self.rank != 0:
            return
        try:
            self._write(
                {
                    "event": "train_end",
                    "epochs": int(getattr(event, "total_epochs", 0)),
                    "steps": int(getattr(event, "global_step", 0)),
                    "duration": float(getattr(event, "duration", 0.0) or 0.0),
                    "best_metric": getattr(event, "best_metric", None),
                }
            )
        except Exception:  # noqa: BLE001, S110
            pass

    # -- io -------------------------------------------------------------------
    def _write(self, rec: dict) -> None:
        try:
            self._fh.write(json.dumps(rec) + "\n")
            self._fh.flush()
        except Exception:  # noqa: BLE001, S110
            pass



