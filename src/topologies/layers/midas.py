import torch
from torch import nn

_VALUE_ERROR = ValueError


class MiDaSScaleShiftLoss(nn.Module):
    component_id = "midas"

    def __init__(self, reduction="mean"):
        super().__init__()
        if reduction not in ("mean", "none"):
            raise ValueError("reduction must be 'mean' or 'none'")
        self.reduction = reduction

    def forward(self, pred, target, valid_mask=None):
        if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise _VALUE_ERROR("pred and target must be tensors")
        if pred.shape != target.shape:
            raise ValueError("pred and target must have the same shape")
        if pred.ndim == 0:
            raise ValueError("pred and target must have at least one dimension")
        if pred.shape[0] == 0 or pred.numel() == 0:
            raise ValueError("each input must contain at least one sample and element")
        if not torch.is_floating_point(pred) or not torch.is_floating_point(target):
            raise ValueError("pred and target must have floating point dtypes")
        if pred.device != target.device:
            raise ValueError("pred and target must be on the same device")
        if valid_mask is not None:
            if not isinstance(valid_mask, torch.Tensor):
                raise ValueError("valid_mask must be a tensor")
            if valid_mask.shape != pred.shape:
                raise ValueError("valid_mask must have the same shape as pred and target")
            if valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must have boolean dtype")
            if valid_mask.device != pred.device:
                raise ValueError("valid_mask must be on the same device as pred and target")

        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
            target = target.reshape(1, -1)
            if valid_mask is not None:
                valid_mask = valid_mask.reshape(1, -1)
        else:
            pred = pred.reshape(pred.shape[0], -1)
            target = target.reshape(target.shape[0], -1)
            if valid_mask is not None:
                valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

        losses = []
        for index in range(pred.shape[0]):
            sample_pred = pred[index]
            sample_target = target[index]
            if valid_mask is not None:
                sample_mask = valid_mask[index]
                if not sample_mask.any():
                    raise ValueError("each sample must have at least one valid element")
                sample_pred = sample_pred[sample_mask]
                sample_target = sample_target[sample_mask]

            centered_pred = sample_pred - sample_pred.mean()
            centered_target = sample_target - sample_target.mean()
            variance = centered_pred.square().sum()
            covariance = (centered_pred * centered_target).sum()
            nonzero_variance = variance > 0
            safe_variance = torch.where(nonzero_variance, variance, torch.ones_like(variance))
            scale = torch.where(nonzero_variance, covariance / safe_variance, torch.zeros_like(variance))
            shift = sample_target.mean() - scale * sample_pred.mean()
            residual = scale * sample_pred + shift - sample_target
            losses.append(residual.square().sum() / (2 * sample_pred.numel()))

        losses = torch.stack(losses)
        return losses.mean() if self.reduction == "mean" else losses
