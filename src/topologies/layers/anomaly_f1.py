import torch
from torch import nn


class AnomalyF1(nn.Module):
    component_id = "anomaly-f1"

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        if not torch.is_tensor(y_true) or not torch.is_tensor(y_pred):
            raise ValueError("y_true and y_pred must be torch tensors")
        if y_true.ndim != y_pred.ndim:
            raise ValueError("y_true and y_pred must have the same rank")
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        if y_true.numel() == 0:
            raise ValueError("y_true and y_pred must contain at least one timestep")
        if y_true.device != y_pred.device:
            raise ValueError("y_true and y_pred must be on the same device")
        if y_true.ndim == 2:
            if y_true.shape[0] == 0 or y_true.shape[1] == 0:
                raise ValueError("batched y_true and y_pred must be non-empty")
            results = [self(y_true[index], y_pred[index]) for index in range(y_true.shape[0])]
            return torch.stack([
                torch.stack((result["raw_f1"], result["point_adjusted_f1"]))
                for result in results
            ])
        if y_true.ndim != 1:
            raise ValueError("y_true and y_pred must be one-dimensional or batched sequences")
        if y_true.dtype.is_complex or y_pred.dtype.is_complex:
            raise ValueError("y_true and y_pred must contain real binary values")
        if not bool(torch.all((y_true == 0) | (y_true == 1))):
            raise ValueError("y_true must contain only binary values 0 and 1")
        if not bool(torch.all((y_pred == 0) | (y_pred == 1))):
            raise ValueError("y_pred must contain only binary values 0 and 1")

        dtype = torch.float64 if y_pred.dtype == torch.float64 else torch.float32
        truth = y_true.to(dtype=dtype)
        prediction = y_pred.to(dtype=dtype)
        adjusted_prediction = prediction.clone()
        labels = y_true.detach().cpu().tolist()
        position = 0
        while position < len(labels):
            if labels[position] == 0:
                position += 1
                continue
            end = position + 1
            while end < len(labels) and labels[end] == 1:
                end += 1
            adjusted_prediction[position:end] = prediction[position:end].max()
            position = end

        def f1(scores):
            true_positives = (truth * scores).sum()
            false_positives = ((1 - truth) * scores).sum()
            false_negatives = (truth * (1 - scores)).sum()
            denominator = 2 * true_positives + false_positives + false_negatives
            value = 2 * true_positives / denominator.clamp_min(1)
            return torch.where(denominator > 0, value, torch.zeros_like(value))

        return {
            "raw_f1": f1(prediction),
            "point_adjusted_f1": f1(adjusted_prediction),
        }
