"""TTS recipe (mel + duration + optional pitch losses)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from topologies.training.recipe_base import Recipe, RecipeConfig, register_recipe


class TTSConfig(RecipeConfig):
    mel_loss_weight: float = 1.0
    duration_loss_weight: float = 0.1
    pitch_loss_weight: float | None = None


@register_recipe("tts", help="Standard TTS losses (mel/duration/pitch).")
class TTSRecipe(Recipe):
    level = "loss"
    ConfigSchema = TTSConfig
    ComponentsSchema = None

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.01, "scheduler": "cosine"}

    def expected_batch_spec(self) -> dict[str, str]:
        return {
            "tokens": "LongTensor[B, T]",
            "mel": "FloatTensor[B, M, T]",
            "durations": "FloatTensor[B, T]",
        }

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "tokens": torch.randint(0, 64, (2, 16), device=dev),
            "mel": torch.randn(2, 32, 16, device=dev),
            "durations": torch.randint(1, 4, (2, 16), device=dev).float(),
        }

    def batch_to_loss(self, model, batch, cfg: TTSConfig, components, adapter=None):
        self.validate_batch(batch, strict=True)
        if adapter is not None and (
            getattr(adapter, "forward_map", None) is not None
            or getattr(adapter, "output_map", None) is not None
        ):
            out = adapter.call(model, batch)
        else:
            try:
                from topologies.training.adapters import call_with_adapter as _cwa

                out = _cwa(adapter, model, {"tokens": batch["tokens"]}, ["tokens"])
            except Exception:  # noqa: BLE001
                try:
                    out = model(batch["tokens"])
                except Exception:  # noqa: BLE001
                    out = model(batch["tokens"], batch.get("mel"))
        if isinstance(out, (tuple, list)):
            mel_pred = out[0]
            dur_pred = out[1] if len(out) > 1 else None
            pitch_pred = out[2] if len(out) > 2 else None
        elif isinstance(out, dict):
            mel_pred, dur_pred, pitch_pred = (
                out.get("mel"), out.get("durations"), out.get("pitch"))
        else:
            mel_pred, dur_pred, pitch_pred = out, None, None
        mel_loss = F.l1_loss(mel_pred, batch["mel"])
        loss = cfg.mel_loss_weight * mel_loss
        metrics: dict[str, float] = {"loss": 0.0, "mel": float(mel_loss.detach())}
        if dur_pred is not None and "durations" in batch:
            d = F.mse_loss(dur_pred.float().reshape(-1),
                            batch["durations"].float().reshape(-1))
            loss = loss + cfg.duration_loss_weight * d
            metrics["duration"] = float(d.detach())
        if (
            pitch_pred is not None
            and cfg.pitch_loss_weight is not None
            and "pitch" in batch
        ):
            p = F.mse_loss(pitch_pred.float().reshape(-1),
                            batch["pitch"].float().reshape(-1))
            loss = loss + float(cfg.pitch_loss_weight) * p
            metrics["pitch"] = float(p.detach())
        metrics["loss"] = float(loss.detach())
        return loss, metrics
