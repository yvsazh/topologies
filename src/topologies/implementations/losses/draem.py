import math

import torch
import torch.nn.functional as F


def draem(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if ignore_index is not None:
        raise ValueError("ignore_index is unsupported for DRAEM binary anomaly masks")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    allowed = {"original", "reconstruction", "ssim_weight", "gamma", "smooth", "ssim_window_size"}
    unexpected = set(params) - allowed
    if unexpected:
        raise ValueError(f"unexpected parameters: {', '.join(sorted(unexpected))}")
    if "original" not in params or "reconstruction" not in params:
        raise ValueError("original and reconstruction tensors are required parameters")

    original = params["original"]
    reconstruction = params["reconstruction"]
    ssim_weight = params.get("ssim_weight", 1.0)
    gamma = params.get("gamma", 2.0)
    smooth = params.get("smooth", 1e-5)
    window_size = params.get("ssim_window_size", 11)

    for name, value in (("ssim_weight", ssim_weight), ("gamma", gamma), ("smooth", smooth)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
    if ssim_weight < 0:
        raise ValueError("ssim_weight must be nonnegative")
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    if smooth <= 0 or smooth > 0.5:
        raise ValueError("smooth must be in the interval (0, 0.5]")
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0 or window_size % 2 == 0:
        raise ValueError("ssim_window_size must be a positive odd integer")

    if (
        not isinstance(logits_or_prediction, torch.Tensor)
        or not isinstance(original, torch.Tensor)
        or not isinstance(reconstruction, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
    ):
        error = ValueError("logits, targets, original, and reconstruction must be tensors")
        raise error
    if logits_or_prediction.ndim != 4 or logits_or_prediction.shape[1] != 2:
        raise ValueError("logits must have shape [B, 2, H, W]")
    if not logits_or_prediction.is_floating_point():
        raise ValueError("logits must be floating-point tensors")
    if original.ndim != 4 or reconstruction.ndim != 4:
        raise ValueError("original and reconstruction must have shape [B, C, H, W]")
    if original.shape != reconstruction.shape:
        raise ValueError("original and reconstruction must have identical shapes")
    batch_size, _, height, width = logits_or_prediction.shape
    if batch_size <= 0 or height <= 0 or width <= 0 or original.shape[1] <= 0:
        raise ValueError("logits, original, and reconstruction dimensions must be nonempty")
    if original.shape[0] != batch_size or original.shape[2:] != (height, width):
        raise ValueError("original and reconstruction batch and spatial shapes must match logits")
    if not original.is_floating_point() or not reconstruction.is_floating_point():
        raise ValueError("original and reconstruction must be floating-point tensors")
    if (
        logits_or_prediction.dtype != original.dtype
        or original.dtype != reconstruction.dtype
        or logits_or_prediction.device != original.device
        or original.device != reconstruction.device
    ):
        raise ValueError("logits, original, and reconstruction must have matching dtype and device")
    if targets.device != logits_or_prediction.device or targets.is_complex():
        raise ValueError("targets must be real-valued and on the logits device")
    if targets.ndim == 4 and targets.shape == (batch_size, 1, height, width):
        mask = targets[:, 0]
    elif targets.ndim == 3 and targets.shape == (batch_size, height, width):
        mask = targets
    else:
        raise ValueError("targets must have shape [B, H, W] or [B, 1, H, W]")
    if not torch.isfinite(logits_or_prediction).all() or not torch.isfinite(original).all() or not torch.isfinite(reconstruction).all():
        raise ValueError("logits, original, and reconstruction must contain only finite values")
    if mask.is_floating_point() and not torch.isfinite(mask).all():
        raise ValueError("targets must contain only finite binary values")
    if not torch.logical_or(mask == 0, mask == 1).all():
        raise ValueError("targets must contain only binary values 0 or 1")

    residual = original - reconstruction
    mse = residual.square().flatten(1).mean(dim=1)

    coordinates = torch.arange(window_size, dtype=original.dtype, device=original.device)
    coordinates = coordinates - (window_size - 1) / 2
    gaussian = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian)
    window = window.expand(original.shape[1], 1, window_size, window_size).contiguous()
    padding = window_size // 2
    original_mean = F.conv2d(original, window, padding=padding, groups=original.shape[1])
    reconstruction_mean = F.conv2d(reconstruction, window, padding=padding, groups=original.shape[1])
    original_variance = F.conv2d(original.square(), window, padding=padding, groups=original.shape[1]) - original_mean.square()
    reconstruction_variance = F.conv2d(reconstruction.square(), window, padding=padding, groups=original.shape[1]) - reconstruction_mean.square()
    covariance = F.conv2d(original * reconstruction, window, padding=padding, groups=original.shape[1]) - original_mean * reconstruction_mean
    if float(original.detach().max()) > 128.0:
        data_range = 255.0
    elif float(original.detach().min()) < -0.5:
        data_range = 2.0
    else:
        data_range = 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_numerator = (2 * original_mean * reconstruction_mean + c1) * (2 * covariance + c2)
    ssim_denominator = (original_mean.square() + reconstruction_mean.square() + c1) * (original_variance + reconstruction_variance + c2)
    ssim_loss = (1 - ssim_numerator / ssim_denominator).flatten(1).mean(dim=1)

    smoothed_target = F.one_hot(mask.to(dtype=torch.long), num_classes=2).permute(0, 3, 1, 2).to(dtype=logits_or_prediction.dtype)
    smoothed_target = smoothed_target.clamp(min=smooth, max=1 - smooth)
    probability = logits_or_prediction.softmax(dim=1)
    probability_true = (smoothed_target * probability).sum(dim=1) + smooth
    focal = -((1 - probability_true) ** gamma) * probability_true.log()
    focal = focal.flatten(1).mean(dim=1)

    per_image = mse + ssim_weight * ssim_loss + focal
    if reduction == "none":
        return per_image
    if reduction == "sum":
        return per_image.sum()
    return per_image.mean()
