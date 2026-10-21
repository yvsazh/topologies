"""Checkpoint policies, save/resume and early stopping for the Trainer."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from topologies.training.checkpoint import load_checkpoint, save_checkpoint
from topologies.training.distributed import (
    barrier,
)
from topologies.training.errors import TrainingError


def _plain(obj: Any) -> Any:
    """Checkpoint-safe (``weights_only``) view of a config dump."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return repr(obj)


class _CheckpointMixin:
    """Trainer mixin (split out of engine.py; state lives on the Trainer)."""

    def _check_checkpoint_policy(self) -> None:
        pol = self.config.checkpoint
        if self.config.checkpoint_dir is None and (
                pol.save_best or pol.every_epochs is not None or pol.every_steps is not None):
            raise TrainingError(
                "Checkpoint policy writes files but checkpoint_dir is not set.",
                stage="config", config_key="checkpoint_dir",
                hint="Pass checkpoint_dir='runs/my-run' (files are never written to an "
                "implicit ./checkpoints directory).")

    def _maybe_checkpoint(self, epoch: int, metrics: dict) -> None:
        if self.dist.rank != 0:
            return
        policy_dir = self.config.checkpoint_dir
        pol = self.config.checkpoint
        if policy_dir is None:
            return  # write policies without a dir are rejected at construction
        self._micro_step_in_epoch = 0  # epoch checkpoints resume at the next epoch's start
        periodic = (pol.every_epochs is not None
                    and (epoch + 1) % max(1, pol.every_epochs) == 0)
        if pol.save_best and pol.monitor in metrics:
            try:
                val = float(metrics[pol.monitor])
            except (TypeError, ValueError):
                val = None
            if val is not None:
                best = self._ckpt_best
                better = best is None or (
                    val < best if pol.monitor_mode == "min" else val > best
                )
                if better:
                    self._ckpt_best = val
                    self._ckpt_best_epoch = int(epoch)
                    _p = self._save(policy_dir, name="best.pt", next_epoch=epoch + 1, is_best=True)
                    if _p:
                        self._last_checkpoint_path = _p
        if periodic or pol.save_last:
            # save_last writes every epoch; only the periodic policy writes epoch files.
            if periodic:
                _p2 = self._save(policy_dir, name=f"epoch-{epoch + 1}.pt", next_epoch=epoch + 1)
                if _p2:
                    self._last_checkpoint_path = _p2
            if pol.save_last:
                _p3 = self._save(policy_dir, name="last.pt", next_epoch=epoch + 1)
                if _p3:
                    self._last_checkpoint_path = _p3

    def _maybe_step_checkpoint(self, epoch: int, micro_in_epoch: int) -> None:
        """``CheckpointPolicy.every_steps``: ``step-{global_step}.pt`` after optimizer updates.

        Resuming from a step checkpoint restarts its epoch from the beginning;
        ``micro_step_in_epoch`` is stored for future mid-epoch resume.
        """
        every = self.config.checkpoint.every_steps
        if every is None or self.config.checkpoint_dir is None:
            return
        if self.global_step % max(1, int(every)) != 0:
            return
        if self.dist.rank == 0:
            self._micro_step_in_epoch = int(micro_in_epoch)
            path = self._save(self.config.checkpoint_dir, name=f"step-{self.global_step}.pt",
                              next_epoch=epoch)
            if path:
                self._last_checkpoint_path = path
        barrier(self.dist)  # other ranks never read a half-written file

    def _save(self, directory: str, *, name: str, next_epoch: int,
              is_best: bool = False) -> str | None:
        """Write a checkpoint; its ``"epoch"`` is the epoch a resumed run starts at."""
        if self.dist.rank != 0:
            return None
        path = str(Path(directory) / name)
        try:
            save_checkpoint(
                path,
                model=self.model,
                optimizers=self.optimizers,
                schedulers=self.schedulers,
                scaler=self.scaler,
                epoch=next_epoch,
                global_step=self.global_step,
                recipe_state={"recipe": self.recipe.name},
                trainer_state={
                    "config": _plain(self.config.model_dump(mode="python")),
                    "history": list(self.history),
                    "samples_seen": int(self.samples_seen),
                    "tokens_seen": int(self.tokens_seen),
                    "micro_step": int(self.micro_step),
                    "micro_step_in_epoch": int(getattr(self, "_micro_step_in_epoch", 0)),
                    # Decision state, so resumed runs continue patience / best tracking.
                    "ckpt_best": self._ckpt_best,
                    "ckpt_best_epoch": self._ckpt_best_epoch,
                    "es_best": self._es_best,
                    "es_best_epoch": self._es_best_epoch,
                    "no_improve": int(self._no_improve),
                    "accum_count": int(self._accum_count),
                    "stop_reason": self.stop_reason,
                },
            )
            self.callbacks.on_checkpoint(self, path)
            # Checkpoint event only after successful save.
            from topologies.training.reporting.events import CheckpointEvent

            self.reporter.on_checkpoint(
                CheckpointEvent(
                    path=str(path),
                    epoch=max(0, int(next_epoch) - 1),
                    global_step=int(self.global_step),
                    is_best=bool(is_best),
                )
            )
            return path
        except Exception as e:
            raise TrainingError(f"Checkpoint save failed: {e}", stage="checkpoint") from e

    def save(self, path: str) -> str:
        """Checkpoint now; resuming from it starts at ``current_epoch``."""
        self._save(str(Path(path).parent), name=Path(path).name, next_epoch=self.current_epoch)
        return path

    def _do_resume(self, resume: Any, *, trust_checkpoint: bool | None = None) -> None:
        if trust_checkpoint is None:
            trust_checkpoint = bool(self.config.trust_checkpoint)
        path: str | None = None
        if resume is True:
            candidates = []
            if self.config.checkpoint_dir:
                for cand in ("last.pt", "best.pt"):
                    p = Path(self.config.checkpoint_dir) / cand
                    if p.exists():
                        candidates.append(p)
            if not candidates:
                raise TrainingError(
                    "resume=True but no checkpoint found.",
                    stage="checkpoint",
                    hint="Set checkpoint_dir=... or resume='/path/to/checkpoint.pt'.",
                )
            path = str(candidates[0])
        elif isinstance(resume, (str, Path)):
            path = str(resume)
        else:
            return
        try:
            info = load_checkpoint(
                path,
                model=self.model,
                optimizers=self.optimizers,
                schedulers=self.schedulers,
                scaler=self.scaler,
                map_location="cpu",
                trust_checkpoint=trust_checkpoint,
            )
            self.model.to(self.device)
        except Exception as e:
            raise TrainingError(f"Resume failed for {path}: {e}", stage="checkpoint") from e
        self.current_epoch = int(info.get("epoch", 0))
        self.global_step = int(info.get("global_step", 0))
        state = info.get("trainer_state") or {}
        if "micro_step" in state:
            self.micro_step = int(state["micro_step"])
        else:
            self.micro_step = self.global_step
            if self.config.accumulation_steps > 1:
                warnings.warn(
                    f"{path} predates optimizer-step counting: its global_step counted "
                    "micro-batches. With accumulation_steps > 1, max_steps now counts "
                    "optimizer updates, so the resumed run may train longer than before.",
                    UserWarning, stacklevel=3)
        if isinstance(state.get("history"), list):
            self.history = [dict(h) for h in state["history"]]
        self.samples_seen = int(state.get("samples_seen", self.samples_seen))
        self.tokens_seen = int(state.get("tokens_seen", self.tokens_seen))
        # Older checkpoints lack decision state; they resume with fresh tracking.
        self._ckpt_best = state.get("ckpt_best", self._ckpt_best)
        self._ckpt_best_epoch = state.get("ckpt_best_epoch", self._ckpt_best_epoch)
        self._es_best = state.get("es_best", self._es_best)
        self._es_best_epoch = state.get("es_best_epoch", self._es_best_epoch)
        self._no_improve = int(state.get("no_improve", self._no_improve))
        self._accum_count = int(state.get("accum_count", 0))
        self.stop_reason = state.get("stop_reason", self.stop_reason)

    def _check_early_stopping(self, metrics: dict) -> bool:
        patience = self.config.early_stopping_patience
        if patience is None:
            return False
        monitor = self.config.early_stopping_monitor
        if monitor not in metrics:
            return False
        val = float(metrics[monitor])
        mode = self.config.early_stopping_mode
        delta = float(self.config.early_stopping_min_delta)
        best = self._es_best
        if best is None:
            better = True
        else:
            better = val < best - delta if mode == "min" else val > best + delta
        if better:
            self._es_best = val
            self._es_best_epoch = int(self.current_epoch)
            self._no_improve = 0
        else:
            self._no_improve += 1
        return self._no_improve >= patience
