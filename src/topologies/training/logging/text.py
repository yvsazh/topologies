"""Human-readable training.log writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from topologies.training.logging.base import BaseLogger


def _fmt(v: Any) -> str:
    try:
        if isinstance(v, float):
            import math as _math

            if _math.isnan(v) or _math.isinf(v):
                return str(v)
            return f"{v:.4g}"
        return str(v)
    except Exception:  # noqa: BLE001
        return str(v)


def _fmt_metrics(metrics: dict | None) -> str:
    parts = []
    for k, v in (metrics or {}).items():
        try:
            parts.append(f"{k}={_fmt(v)}")
        except Exception:  # noqa: BLE001, S112
            continue
    return " ".join(parts)


class TextLogger(BaseLogger):
    """Human-readable training.log (one line per epoch + begin/end)."""

    def __init__(self, path: str | Path, rank: int = 0) -> None:
        self.rank = rank
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, step: int, values: dict[str, Any]) -> None:
        if self.rank != 0:
            return
        try:
            self._fh.write(f"[step {int(step)}] {_fmt_metrics(values)}\n")
            self._fh.flush()
        except Exception:  # noqa: BLE001, S110
            pass

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001, S110
            pass

    def on_train_begin(self, event: Any) -> None:
        if self.rank != 0:
            return
        try:
            info = getattr(event, "info", None)
            name = getattr(info, "model_name", None) if info else None
            recipe = getattr(info, "recipe_name", None) if info else None
            self._fh.write(f"training begin: model={name} recipe={recipe}\n")
            self._fh.flush()
        except Exception:  # noqa: BLE001, S110
            pass

    def on_epoch_begin(self, event: Any) -> None:
        pass

    def on_step(self, event: Any) -> None:
        # Text log keeps per-epoch granularity; step lines via legacy log().
        # Still record throttled steps as compact lines for debugging.
        try:
            step = int(getattr(event, "global_step", 0))
            metrics = getattr(event, "metrics", {}) or {}
            self.log(step, dict(metrics))
        except Exception:  # noqa: BLE001, S110
            pass

    def on_validation(self, event: Any) -> None:
        if self.rank != 0:
            return
        try:
            ep = int(getattr(event, "epoch", 0))
            m = getattr(event, "metrics", {}) or {}
            self._fh.write(f"[epoch {ep}] validation: {_fmt_metrics(m)}\n")
            self._fh.flush()
        except Exception:  # noqa: BLE001, S110
            pass

    def on_validation_end(self, event: Any) -> None:
        self.on_validation(event)

    def on_checkpoint(self, event: Any) -> None:
        if self.rank != 0:
            return
        try:
            self._fh.write(f"checkpoint: {getattr(event, 'path', '')}\n")
            self._fh.flush()
        except Exception:  # noqa: BLE001, S110
            pass

    def on_epoch_end(self, event: Any) -> None:
        if self.rank != 0:
            return
        try:
            ep = int(getattr(event, "epoch", 0))
            tm = getattr(event, "train_metrics", {}) or {}
            vm = getattr(event, "val_metrics", {}) or {}
            merged = {**dict(tm), **dict(vm)}
            line = f"[epoch {ep}] {_fmt_metrics(merged)}"
            if getattr(event, "checkpoint_path", None):
                line += f" checkpoint={event.checkpoint_path}"
            self._fh.write(line + "\n")
            self._fh.flush()
        except Exception:  # noqa: BLE001, S110
            pass

    def on_train_end(self, event: Any) -> None:
        if self.rank != 0:
            return
        try:
            self._fh.write(
                f"training end: epochs={getattr(event, 'total_epochs', 0)} "
                f"steps={getattr(event, 'global_step', 0)} "
                f"duration={float(getattr(event, 'duration', 0.0) or 0.0):.1f}s\n"
            )
            self._fh.flush()
        except Exception:  # noqa: BLE001, S110
            pass


# Backward-compatible alias: old FileLogger wrote metrics.log.
FileLogger = TextLogger
