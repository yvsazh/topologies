import torch


def chamfer(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if params:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(params))}")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for point clouds")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not isinstance(logits_or_prediction, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("prediction and targets must be torch.Tensor instances")
    if logits_or_prediction.ndim != 3 or targets.ndim != 3:
        raise ValueError("prediction and targets must have rank 3 with shape [B, points, 3]")
    if logits_or_prediction.shape[-1] != 3 or targets.shape[-1] != 3:
        raise ValueError("point clouds must have exactly three coordinates per point")
    if logits_or_prediction.shape[0] == 0 or targets.shape[0] == 0:
        raise ValueError("point clouds must have a nonempty batch")
    if logits_or_prediction.shape[0] != targets.shape[0]:
        raise ValueError("prediction and targets must have matching batch dimensions")
    if logits_or_prediction.shape[1] == 0 or targets.shape[1] == 0:
        raise ValueError("point clouds must have nonempty point axes")
    if not logits_or_prediction.is_floating_point() or not targets.is_floating_point():
        raise ValueError("prediction and targets must use floating-point dtypes")
    if logits_or_prediction.dtype != targets.dtype:
        raise ValueError("prediction and targets must have the same dtype")
    if logits_or_prediction.device != targets.device:
        raise ValueError("prediction and targets must be on the same device")

    pairwise_squared_distances = (
        logits_or_prediction.unsqueeze(2) - targets.unsqueeze(1)
    ).square().sum(dim=-1)
    per_cloud = pairwise_squared_distances.amin(dim=2).sum(dim=1)
    per_cloud = per_cloud + pairwise_squared_distances.amin(dim=1).sum(dim=1)
    if reduction == "none":
        return per_cloud
    if reduction == "sum":
        return per_cloud.sum()
    return per_cloud.mean()
