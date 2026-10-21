"""Vision recipes."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pydantic import Field, model_validator

from topologies.evaluation import EvaluationRecord
from topologies.training.adapters import call_with_adapter
from topologies.training.errors import TrainingError
from topologies.training.losses import LossOverrideConfig, compute_loss
from topologies.training.recipe_base import EvaluationStepResult, Recipe, register_recipe


class VisionClassifyConfig(LossOverrideConfig):
    """``loss`` / ``loss_kwargs``: optional override called as ``fn(logits, labels, **kw)``
    (``label_smoothing`` applies to the built-in loss only)."""

    num_classes: int | None = Field(
        default=None, description="If set, model outputs/labels are checked against it."
    )
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    mixup_alpha: float = Field(default=0.0, ge=0.0, description="Beta(a, a) mixup; 0 = off.")
    cutmix_alpha: float = Field(default=0.0, ge=0.0, description="Beta(a, a) CutMix; 0 = off.")
    mix_prob: float = Field(default=1.0, ge=0.0, le=1.0,
                            description="Probability of mixing a training batch.")

    @model_validator(mode="after")
    def _smoothing_with_override(self) -> VisionClassifyConfig:
        if self.loss is not None and self.label_smoothing:
            raise ValueError("label_smoothing applies to the built-in loss only; with "
                             "loss=... pass it via loss_kwargs (e.g. for 'cross_entropy').")
        return self


def _ce(logits: torch.Tensor, labels: torch.Tensor, cfg: VisionClassifyConfig) -> torch.Tensor:
    if cfg.loss is None:
        return F.cross_entropy(logits, labels, label_smoothing=cfg.label_smoothing)
    return compute_loss(cfg, logits, labels)


def _logits(out: object) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        out = next((o for o in out if isinstance(o, torch.Tensor)), out[0])
    if isinstance(out, dict):
        out = out.get("logits", next(iter(out.values())))
    if not isinstance(out, torch.Tensor):
        raise TrainingError(
            f"Model returned {type(out).__name__}; vision-classify needs logits [B, C].",
            recipe="vision-classify", stage="loss",
            hint="Return logits from forward(), or pass output_map=lambda out: out.logits.",
        )
    return out


def _mix(pixels: torch.Tensor, labels: torch.Tensor, cfg: VisionClassifyConfig):
    """Return (mixed pixels, labels_b, lam) or None when no mixing applies."""
    use_mixup, use_cutmix = cfg.mixup_alpha > 0, cfg.cutmix_alpha > 0
    if not (use_mixup or use_cutmix) or pixels.shape[0] < 2:
        return None
    if float(torch.rand(())) > cfg.mix_prob:
        return None
    perm = torch.randperm(pixels.shape[0], device=pixels.device)
    if use_cutmix and (not use_mixup or float(torch.rand(())) < 0.5):
        lam = float(torch.distributions.Beta(cfg.cutmix_alpha, cfg.cutmix_alpha).sample())
        H, W = pixels.shape[-2:]
        cut_h, cut_w = int(H * (1 - lam) ** 0.5), int(W * (1 - lam) ** 0.5)
        cy, cx = int(torch.randint(H, ())), int(torch.randint(W, ()))
        y0, y1 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
        x0, x1 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
        mixed = pixels.clone()
        mixed[..., y0:y1, x0:x1] = pixels[perm][..., y0:y1, x0:x1]
        lam = 1.0 - (y1 - y0) * (x1 - x0) / float(H * W)
    else:
        lam = float(torch.distributions.Beta(cfg.mixup_alpha, cfg.mixup_alpha).sample())
        mixed = lam * pixels + (1.0 - lam) * pixels[perm]
    return mixed, labels[perm], lam


def _pixel_key(batch: dict, adapter=None) -> str:
    mapped = getattr(adapter, "input_map", {}).get("pixels") if adapter is not None else None
    for k in ("pixels", mapped, "images"):
        if k and k in batch:
            return k
    return "pixels"


@register_recipe("vision-classify", help="Image classification (cross-entropy).")
class VisionClassifyRecipe(Recipe):
    level = "loss"
    ConfigSchema = VisionClassifyConfig
    ComponentsSchema = None

    def default_training_policy(self) -> dict:
        return {"optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.05, "scheduler": "cosine"}

    def expected_batch_spec(self) -> dict[str, str]:
        return {"pixels": "FloatTensor[B, C, H, W]", "labels": "LongTensor[B]"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        images = torch.randn(4, 3, 16, 16, device=dev)
        labels = torch.randint(0, 10, (4,), device=dev)
        return {"pixels": images, "labels": labels}

    def validate_batch(self, batch: dict, *, strict: bool = True) -> None:
        if "images" in batch and "pixels" not in batch:
            batch = {**batch, "pixels": batch["images"]}
        super().validate_batch(batch, strict=strict)

    def forward(self, model, batch: dict, adapter=None, pixels: torch.Tensor | None = None):
        key = _pixel_key(batch, adapter)
        inputs = {key: batch[key] if pixels is None else pixels}
        return _logits(call_with_adapter(adapter, model, {**batch, **inputs}, [key],
                                         primary=key))

    def _check_logits(self, logits: torch.Tensor, cfg: VisionClassifyConfig) -> None:
        if logits.dim() != 2:
            raise TrainingError(
                f"Model output has shape {tuple(logits.shape)}; vision-classify needs [B, C].",
                recipe=self.name, stage="loss",
                hint="Make forward() return class logits (e.g. end with a Linear head).",
            )
        if cfg.num_classes is not None and logits.shape[-1] != cfg.num_classes:
            raise TrainingError(
                f"Model outputs {logits.shape[-1]} classes but recipe_config "
                f"num_classes={cfg.num_classes}.",
                recipe=self.name, stage="config", config_key="num_classes",
                hint="Build the model head with the same number of classes, or fix "
                     "recipe_config={'num_classes': ...}.",
            )

    def preflight(self, model, batch, cfg: VisionClassifyConfig, components, adapter=None):
        labels = batch["labels"]
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                logits = self.forward(model, batch, adapter)
        finally:
            model.train(was_training)
        self._check_logits(logits, cfg)
        lo, hi = int(labels.min()), int(labels.max())
        n = logits.shape[-1]
        if lo < 0 or hi >= n:
            raise TrainingError(
                f"Labels span [{lo}, {hi}] but the model outputs {n} classes.",
                recipe=self.name, stage="batch",
                hint=f"Build the model with num_classes>={hi + 1} (or remap labels to 0..{n - 1}).",
            )

    def batch_to_loss(self, model, batch, cfg: VisionClassifyConfig, components, adapter=None):
        self.validate_batch(batch, strict=True)
        key = _pixel_key(batch, adapter)
        labels = batch["labels"].long()
        mixed = _mix(batch[key], labels, cfg) if model.training else None
        if mixed is None:
            logits = self.forward(model, batch, adapter)
            self._check_logits(logits, cfg)
            loss = _ce(logits, labels, cfg)
        else:
            pixels, labels_b, lam = mixed
            logits = self.forward(model, batch, adapter, pixels=pixels)
            self._check_logits(logits, cfg)
            loss = lam * _ce(logits, labels, cfg) + (1.0 - lam) * _ce(logits, labels_b, cfg)
        acc = (logits.argmax(-1) == labels).float().mean()
        return loss, {"loss": loss.detach(), "acc": acc.detach()}  # tensors: no per-step sync

    def predict_step(self, model, batch, cfg, components, adapter=None):
        return self.forward(model, batch, adapter)

    def evaluation_step(self, model, batch, cfg, components, adapter=None):
        logits = self.forward(model, batch, adapter)
        labels = batch["labels"].long()
        self._check_logits(logits, cfg)
        loss = _ce(logits, labels, cfg)
        metrics = {"loss": float(loss), "acc": float((logits.argmax(-1) == labels).float().mean())}
        return EvaluationStepResult(metrics, EvaluationRecord(logits, labels,
                                    sample_ids=batch.get("sample_ids"), metadata=batch))


@register_recipe("lenet-classify", help="LeNet-5 classification on 32x32 grayscale images.")
class LeNetClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 1, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("alexnet-classify", help="AlexNet classification on 227x227 images.")
class AlexNetClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 227, 227, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("vgg-classify", help="VGG classification on 224x224 images.")
class VGGClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 224, 224, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("inception-classify", help="Inception classification on 32x32 images.")
class InceptionClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("densenet-classify", help="DenseNet classification on 32x32 images.")
class DenseNetClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("efficient-conv-classify", help="Efficient-convolution classification on 32x32 images.")
class EfficientConvClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("modern-conv-classify", help="Modern-convolution classification on 32x32 images.")
class ModernConvClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("deit-classify", help="DeiT classification on 32x32 images.")
class DeiTClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("swin-classify", help="Swin classification on 32x32 images.")
class SwinClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("token-mixer-classify", help="Token-mixer classification on 32x32 images.")
class TokenMixerClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("vision-ssl-classify", help="Vision SSL classification on 32x32 images.")
class VisionSSLClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("masked-image-classify", help="Masked-image classification on 32x32 images.")
class MaskedImageClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("contrastive-vision-classify", help="Contrastive-vision classification on 32x32 images.")
class ContrastiveVisionClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("two-stage-classify", help="Two-stage detection classification on 32x32 images.")
class TwoStageClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("single-stage-classify", help="Single-stage detection classification on 32x32 images.")
class SingleStageClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("set-detect-classify", help="Set-detection classification on 32x32 images.")
class SetDetectClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("detect-ops-classify", help="Detection-component classification on 32x32 images.")
class DetectOpsClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("segment-classify", help="Segmentation classification on 32x32 images.")
class SegmentClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("prompt-seg-classify", help="Promptable-segmentation classification on 32x32 images.")
class PromptSegClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("panoptic-classify", help="Panoptic classification on 32x32 images.")
class PanopticClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("pose-classify", help="Pose classification on 32x32 images.")
class PoseClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("vision-neck-classify", help="Vision-neck classification on 32x32 images.")
class VisionNeckClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("attn2d-classify", help="2D-attention classification on 32x32 images.")
class Attn2DClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("vision-norm-classify", help="Normalization classification on 32x32 images.")
class VisionNormClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("pos2d-classify", help="Positional classification on 32x32 images.")
class Pos2DClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("video-conv-classify", help="Video-convolution classification on 32x32 images.")
class VideoConvClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("video-transformer-classify", help="Video-transformer classification on 32x32 images.")
class VideoTransformerClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("video-ssl-classify", help="Video self-supervision classification on 32x32 images.")
class VideoSSLClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("point-cloud-classify", help="Point-cloud classification on 32x32 images.")
class PointCloudClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("voxel-detect-classify", help="Voxel-detector classification on 32x32 images.")
class VoxelDetectClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("volume-seg-classify", help="Volumetric classification on 32x32 images.")
class VolumeSegClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("neural-field-classify", help="Neural-field classification on 32x32 images.")
class NeuralFieldClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("implicit-surface-classify", help="Implicit-surface classification on 32x32 images.")
class ImplicitSurfaceClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }


@register_recipe("resnet-variant-classify", help="Residual-variant classification on 32x32 images.")
class ResidualVariantClassifyRecipe(VisionClassifyRecipe):
    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {
            "pixels": torch.randn(4, 3, 32, 32, device=dev),
            "labels": torch.randint(0, 10, (4,), device=dev),
        }
