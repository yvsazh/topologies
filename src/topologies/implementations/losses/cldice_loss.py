import math
import numbers

import torch
import torch.nn.functional as F


def cldice_loss(logits_or_prediction, targets, *, ignore_index=None, reduction="mean", **params):
    def invalid_value(message):
        raise ValueError(message)

    if set(params) - {"iterations", "epsilon"}:
        raise ValueError("unsupported parameter; only iterations and epsilon are supported")
    if set(params) != {"iterations", "epsilon"}:
        raise ValueError("iterations and epsilon are required")
    iterations = params["iterations"]
    if isinstance(iterations, bool) or not isinstance(iterations, numbers.Integral) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    epsilon = params["epsilon"]
    if isinstance(epsilon, bool) or not isinstance(epsilon, numbers.Real) or not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite real number")
    if ignore_index is not None:
        raise ValueError("ignore_index must be None for binary foreground masks")
    if not isinstance(reduction, str) or reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not isinstance(logits_or_prediction, torch.Tensor) or not logits_or_prediction.is_floating_point():
        raise ValueError("logits_or_prediction must be a floating point probability tensor")
    if not isinstance(targets, torch.Tensor):
        invalid_value("targets must be a floating point binary tensor")
    if logits_or_prediction.device != targets.device:
        raise ValueError("targets and logits_or_prediction must be on the same device")
    if logits_or_prediction.dtype != targets.dtype or not targets.is_floating_point():
        raise ValueError("targets and logits_or_prediction must have the same dtype")
    if logits_or_prediction.ndim not in (4, 5):
        raise ValueError("logits_or_prediction must have 2D or 3D spatial shape")
    if logits_or_prediction.shape[0] < 1 or logits_or_prediction.shape[1] != 1 or any(size < 1 for size in logits_or_prediction.shape[2:]):
        raise ValueError("logits_or_prediction must contain one foreground channel and nonempty dimensions")
    if logits_or_prediction.shape != targets.shape:
        raise ValueError("targets and logits_or_prediction must have matching shapes")
    if not bool(torch.isfinite(logits_or_prediction).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("finite probabilities and targets are required")
    if bool(((logits_or_prediction < 0) | (logits_or_prediction > 1)).any()):
        raise ValueError("logits_or_prediction probabilities must be between 0 and 1")
    if bool(((targets != 0) & (targets != 1)).any()):
        raise ValueError("targets must contain binary values 0 or 1")

    if logits_or_prediction.ndim == 4:
        def soft_erode(image):
            vertical = -F.max_pool2d(-image, (3, 1), (1, 1), (1, 0))
            horizontal = -F.max_pool2d(-image, (1, 3), (1, 1), (0, 1))
            return torch.minimum(vertical, horizontal)

        def soft_dilate(image):
            return F.max_pool2d(image, (3, 3), (1, 1), (1, 1))
    else:
        def soft_erode(image):
            depth = -F.max_pool3d(-image, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            height = -F.max_pool3d(-image, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            width = -F.max_pool3d(-image, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.minimum(torch.minimum(depth, height), width)

        def soft_dilate(image):
            return F.max_pool3d(image, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_skeleton(image):
        opened = soft_dilate(soft_erode(image))
        skeleton = F.relu(image - opened)
        for _ in range(int(iterations)):
            image = soft_erode(image)
            opened = soft_dilate(soft_erode(image))
            delta = F.relu(image - opened)
            skeleton = skeleton + F.relu(delta - skeleton * delta)
        return skeleton

    prediction_skeleton = soft_skeleton(logits_or_prediction)
    target_skeleton = soft_skeleton(targets)
    epsilon = float(epsilon)
    precision = ((prediction_skeleton * targets).flatten(1).sum(dim=1) + epsilon) / (prediction_skeleton.flatten(1).sum(dim=1) + epsilon)
    sensitivity = ((target_skeleton * logits_or_prediction).flatten(1).sum(dim=1) + epsilon) / (target_skeleton.flatten(1).sum(dim=1) + epsilon)
    losses = 1 - (2 * precision * sensitivity) / (precision + sensitivity)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()
