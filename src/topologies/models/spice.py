import math

import torch
from torch import nn


class SPICERelativePitchLoss(nn.Module):
    architecture_id = "spice"
    PRESETS = {"tiny": {"pitch_scale": 1.0}, "small": {"pitch_scale": 1.0}}

    def __init__(self, pitch_scale, huber_threshold=None):
        try:
            valid_pitch_scale = (
                not isinstance(pitch_scale, bool)
                and isinstance(pitch_scale, (int, float))
                and math.isfinite(pitch_scale)
                and pitch_scale > 0
            )
        except OverflowError:
            valid_pitch_scale = False
        if not valid_pitch_scale:
            raise ValueError("pitch_scale must be a finite positive number")
        if huber_threshold is None:
            huber_threshold = 0.25 * pitch_scale
        try:
            valid_huber_threshold = (
                not isinstance(huber_threshold, bool)
                and isinstance(huber_threshold, (int, float))
                and math.isfinite(huber_threshold)
                and huber_threshold > 0
            )
        except OverflowError:
            valid_huber_threshold = False
        if not valid_huber_threshold:
            raise ValueError("huber_threshold must be a finite positive number")
        super().__init__()
        self.pitch_scale = float(pitch_scale)
        self.huber_threshold = float(huber_threshold)

    def forward(self, first_prediction, second_prediction, first_shift, second_shift):
        values = (first_prediction, second_prediction, first_shift, second_shift)
        if not all(torch.is_tensor(value) for value in values):
            raise ValueError("predictions and shifts must be tensors")
        if any(value.ndim != 1 for value in values):
            raise ValueError("predictions and shifts must have shape [B]")
        if any(value.numel() == 0 for value in values):
            raise ValueError("predictions and shifts must be nonempty")
        if any(value.shape != first_prediction.shape for value in values[1:]):
            raise ValueError("predictions and shifts must have the same shape")
        if first_prediction.device != second_prediction.device:
            raise ValueError("predictions must be on the same device")
        if first_shift.device != second_shift.device:
            raise ValueError("shifts must be on the same device")
        if first_prediction.device != first_shift.device:
            raise ValueError("predictions and shifts must be on the same device")
        if not first_prediction.is_floating_point() or not second_prediction.is_floating_point():
            raise ValueError("predictions must be floating-point tensors")
        if any(
            shift.dtype == torch.bool or shift.is_complex()
            for shift in (first_shift, second_shift)
        ):
            raise ValueError("shifts must have real numeric dtypes")

        error = (
            (first_prediction - second_prediction)
            - self.pitch_scale * (first_shift - second_shift)
        ).abs()
        threshold = self.huber_threshold
        penalties = torch.where(
            error <= threshold,
            0.5 * error.square(),
            threshold * error - 0.5 * threshold**2,
        )
        return penalties.mean()
