import torch


def eiou(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    if params:
        raise ValueError(f"unsupported parameters: {', '.join(sorted(params))}")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be none, mean, or sum")
    if not torch.is_tensor(logits_or_prediction) or not torch.is_tensor(targets):
        raise ValueError("prediction and targets must be tensors")
    if logits_or_prediction.shape != targets.shape or logits_or_prediction.ndim < 2 or logits_or_prediction.shape[-1] != 4:
        raise ValueError("prediction and targets must have matching [...,4] xyxy shapes")
    if logits_or_prediction.numel() == 0:
        raise ValueError("prediction and targets must contain at least one box pair")
    if not torch.is_floating_point(logits_or_prediction) or not torch.is_floating_point(targets):
        raise ValueError("prediction and targets must have floating-point dtypes")
    if logits_or_prediction.dtype != targets.dtype or logits_or_prediction.device != targets.device:
        raise ValueError("prediction and targets must have the same dtype and device")
    if not bool(torch.isfinite(logits_or_prediction).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("prediction and targets must contain only finite coordinates")
    if bool((logits_or_prediction[..., 2:] <= logits_or_prediction[..., :2]).any()) or bool(
        (targets[..., 2:] <= targets[..., :2]).any()
    ):
        raise ValueError("xyxy boxes must have x2>x1 and y2>y1")

    prediction_min = logits_or_prediction[..., :2]
    prediction_max = logits_or_prediction[..., 2:]
    targets_min = targets[..., :2]
    targets_max = targets[..., 2:]
    intersection_min = torch.maximum(prediction_min, targets_min)
    intersection_max = torch.minimum(prediction_max, targets_max)
    intersection = (intersection_max - intersection_min).clamp_min(0).prod(dim=-1)
    prediction_size = prediction_max - prediction_min
    targets_size = targets_max - targets_min
    prediction_area = prediction_size.prod(dim=-1)
    targets_area = targets_size.prod(dim=-1)
    union = prediction_area + targets_area - intersection
    iou = intersection / union

    enclosing_min = torch.minimum(prediction_min, targets_min)
    enclosing_max = torch.maximum(prediction_max, targets_max)
    enclosing_size = enclosing_max - enclosing_min
    prediction_center = (prediction_min + prediction_max) / 2
    targets_center = (targets_min + targets_max) / 2
    center_distance = (prediction_center - targets_center).square().sum(dim=-1)
    enclosing_width = enclosing_size[..., 0]
    enclosing_height = enclosing_size[..., 1]
    distance_penalty = center_distance / (enclosing_width.square() + enclosing_height.square())
    width_penalty = (prediction_size[..., 0] - targets_size[..., 0]).square() / enclosing_width.square()
    height_penalty = (prediction_size[..., 1] - targets_size[..., 1]).square() / enclosing_height.square()
    losses = 1 - iou + distance_penalty + width_penalty + height_penalty

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
