import torch


def integral_regression(
    logits_or_prediction,
    targets,
    *,
    ignore_index=None,
    reduction="mean",
    **params,
):
    if params:
        raise ValueError(f"unsupported parameter(s): {', '.join(sorted(params))}")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not isinstance(logits_or_prediction, torch.Tensor) or logits_or_prediction.ndim not in (4, 5):
        raise ValueError("logits_or_prediction must be a 2D or 3D heatmap tensor with rank 4 or 5")
    if not torch.is_tensor(targets):
        raise ValueError("targets must be a tensor")
    if not logits_or_prediction.is_floating_point() or not targets.is_floating_point():
        raise ValueError("logits_or_prediction and targets must be floating-point tensors")
    if logits_or_prediction.dtype != targets.dtype:
        raise ValueError("logits_or_prediction and targets must have the same dtype")
    if logits_or_prediction.device != targets.device:
        raise ValueError("logits_or_prediction and targets must be on the same device")
    if any(size <= 0 for size in logits_or_prediction.shape):
        raise ValueError("logits_or_prediction dimensions must be positive")
    if not torch.isfinite(logits_or_prediction).all() or not torch.isfinite(targets).all():
        raise ValueError("logits_or_prediction and targets must contain only finite values")

    if logits_or_prediction.ndim == 4:
        batch, joints, height, width = logits_or_prediction.shape
        coordinate_count = 2
        y_grid, x_grid = torch.meshgrid(
            torch.arange(1, height + 1, dtype=logits_or_prediction.dtype, device=logits_or_prediction.device),
            torch.arange(1, width + 1, dtype=logits_or_prediction.dtype, device=logits_or_prediction.device),
            indexing="ij",
        )
        coordinates = torch.stack((x_grid, y_grid), dim=-1).reshape(-1, coordinate_count)
    else:
        batch, joints, depth, height, width = logits_or_prediction.shape
        coordinate_count = 3
        z_grid, y_grid, x_grid = torch.meshgrid(
            torch.arange(1, depth + 1, dtype=logits_or_prediction.dtype, device=logits_or_prediction.device),
            torch.arange(1, height + 1, dtype=logits_or_prediction.dtype, device=logits_or_prediction.device),
            torch.arange(1, width + 1, dtype=logits_or_prediction.dtype, device=logits_or_prediction.device),
            indexing="ij",
        )
        coordinates = torch.stack((x_grid, y_grid, z_grid), dim=-1).reshape(-1, coordinate_count)

    if targets.shape != (batch, joints, coordinate_count):
        raise ValueError(
            f"targets must have shape {(batch, joints, coordinate_count)} to match heatmaps"
        )

    probabilities = torch.softmax(logits_or_prediction.reshape(batch, joints, -1), dim=-1)
    expected_coordinates = probabilities @ coordinates
    errors = (expected_coordinates - targets).abs()
    if reduction == "none":
        return errors
    if reduction == "sum":
        return errors.sum()
    return errors.mean()
