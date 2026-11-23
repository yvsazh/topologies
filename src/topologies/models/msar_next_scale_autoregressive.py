import math
from itertools import pairwise

import torch
from torch import nn


class MSARNextScaleAutoregressive(nn.Module):
    architecture_id = "msar-next-scale-autoregressive"
    PRESETS = {
        "tiny": {"scale_factors": (2, 1), "scale_history_lengths": (6, 12), "horizon": 4},
        "small": {
            "scale_factors": (6, 3, 1),
            "scale_history_lengths": (16, 32, 96),
            "horizon": 24,
        },
        "base": {"scale_factors": (6, 1), "scale_history_lengths": (56, 336), "horizon": 96},
    }

    def __init__(self, scale_factors=(6, 1), scale_history_lengths=(56, 336), horizon=96):
        super().__init__()
        try:
            scale_factors = tuple(scale_factors)
        except TypeError as error:
            raise ValueError("scale_factors must be an iterable of integers") from error
        try:
            scale_history_lengths = tuple(scale_history_lengths)
        except TypeError as error:
            raise ValueError("scale_history_lengths must be an iterable of integers") from error
        if len(scale_factors) < 2:
            raise ValueError("scale_factors must contain at least two scales")
        if len(scale_factors) != len(scale_history_lengths):
            raise ValueError("scale_factors and scale_history_lengths must have equal lengths")
        if any(type(factor) is not int or factor <= 0 for factor in scale_factors):
            raise ValueError("scale_factors must contain positive integers")
        if any(type(length) is not int or length <= 0 for length in scale_history_lengths):
            raise ValueError("scale_history_lengths must contain positive integers")
        if any(left <= right for left, right in pairwise(scale_factors)):
            raise ValueError("scale_factors must be strictly descending")
        if scale_factors[-1] != 1:
            raise ValueError("the final scale factor must be 1")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("horizon must be a positive integer")

        self.scale_factors = tuple(scale_factors)
        self.scale_history_lengths = tuple(scale_history_lengths)
        self.horizon = horizon
        self.scale_horizons = tuple(math.ceil(horizon / factor) for factor in self.scale_factors)
        self.forecast_modules = nn.ModuleList(
            [nn.Linear(self.scale_history_lengths[0], self.scale_horizons[0])]
        )
        self.imputation_modules = nn.ModuleList(
            [
                nn.Linear(
                    2 * (self.scale_history_lengths[index] + self.scale_horizons[index]),
                    self.scale_history_lengths[index] + self.scale_horizons[index],
                )
                for index in range(1, len(self.scale_factors))
            ]
        )

    def _sample_histories(self, x):
        return tuple(
            x[:, -(length * factor) :: factor, :]
            for factor, length in zip(self.scale_factors, self.scale_history_lengths)
        )

    def _align_predictions(self, scale_index, predictions, batch_size, channels, device, dtype):
        factor = self.scale_factors[scale_index]
        target_length = self.scale_horizons[scale_index]
        aligned = torch.zeros(batch_size, target_length, channels, device=device, dtype=dtype)
        covered = torch.zeros(batch_size, target_length, 1, device=device, dtype=torch.bool)
        target_positions = torch.arange(target_length, device=device) * factor
        for previous_index in range(scale_index):
            previous_factor = self.scale_factors[previous_index]
            slots = torch.nonzero(target_positions.remainder(previous_factor) == 0).flatten()
            if slots.numel() > 0:
                previous_positions = target_positions.index_select(0, slots) // previous_factor
                aligned[:, slots, :] = predictions[previous_index].index_select(1, previous_positions)
                covered[:, slots, :] = True
        return aligned, covered

    def forward(self, x, return_scales=False, *, target=None, training_stage="inference"):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("input must be a tensor with shape [batch, time, channels]")
        if x.shape[0] <= 0 or x.shape[2] <= 0:
            raise ValueError("input batch and channel dimensions must be positive")
        if training_stage not in ("inference", "teacher_forcing", "joint"):
            raise ValueError("training_stage must be 'inference', 'teacher_forcing', or 'joint'")
        if training_stage == "teacher_forcing" and target is None:
            raise ValueError("target is required for teacher_forcing training")
        if target is not None:
            if not isinstance(target, torch.Tensor) or target.ndim != 3:
                raise ValueError("target must be a tensor with shape [batch, horizon, channels]")
            if target.shape != (x.shape[0], self.horizon, x.shape[2]):
                raise ValueError("target must match the input batch, configured horizon, and channels")
            if not torch.is_floating_point(target):
                raise ValueError("target values must have a floating-point dtype")
        minimum_length = max(
            length * factor for factor, length in zip(self.scale_factors, self.scale_history_lengths)
        )
        if x.shape[1] < minimum_length:
            raise ValueError(f"input sequence must contain at least {minimum_length} time steps")
        if not torch.is_floating_point(x):
            raise ValueError("input values must have a floating-point dtype")

        histories = self._sample_histories(x)
        coarse = self.forecast_modules[0](histories[0].transpose(1, 2)).transpose(1, 2)
        predictions = [coarse]
        aligned_predictions = [torch.zeros_like(coarse)]
        scale_masks = [torch.ones(x.shape[0], coarse.shape[1], 1, device=x.device, dtype=torch.bool)]
        target_scales = (
            tuple(target[:, ::factor, :] for factor in self.scale_factors)
            if target is not None
            else None
        )

        for scale_index in range(1, len(self.scale_factors)):
            conditioning_predictions = (
                target_scales[:scale_index]
                if training_stage == "teacher_forcing"
                else predictions
            )
            aligned, covered = self._align_predictions(
                scale_index, conditioning_predictions, x.shape[0], x.shape[2], x.device, x.dtype
            )
            values = torch.cat((histories[scale_index], aligned), dim=1)
            history_mask = torch.ones(
                x.shape[0], histories[scale_index].shape[1], 1, device=x.device, dtype=torch.bool
            )
            mask = torch.cat((history_mask, covered), dim=1)
            temporal_values = values.transpose(1, 2)
            temporal_mask = mask.expand(-1, -1, x.shape[2]).transpose(1, 2).to(x.dtype)
            imputation_input = torch.cat((temporal_values, temporal_mask), dim=-1)
            refined = self.imputation_modules[scale_index - 1](imputation_input).transpose(1, 2)
            predictions.append(refined[:, -self.scale_horizons[scale_index] :, :])
            aligned_predictions.append(aligned)
            scale_masks.append(mask)

        if return_scales:
            return {
                "forecast": predictions[-1],
                "scale_histories": histories,
                "scale_predictions": tuple(predictions),
                "aligned_predictions": tuple(aligned_predictions),
                "scale_masks": tuple(scale_masks),
            }
        return predictions[-1]

    def training_loss(self, x, target, stage="joint"):
        if stage not in ("teacher_forcing", "joint"):
            raise ValueError("stage must be 'teacher_forcing' or 'joint'")
        result = self.forward(
            x,
            target=target,
            training_stage=stage,
            return_scales=True,
        )
        return self.scale_mae_loss(result["scale_predictions"], target)

    def scale_mae_loss(self, scale_predictions, target):
        if not isinstance(target, torch.Tensor) or target.ndim != 3:
            raise ValueError("target must be a tensor with shape [batch, horizon, channels]")
        if target.shape[1] != self.horizon:
            raise ValueError("target horizon must match the configured horizon")
        if not isinstance(scale_predictions, (tuple, list)) or len(scale_predictions) != len(
            self.scale_factors
        ):
            raise ValueError("scale_predictions must contain one tensor for every configured scale")
        losses = []
        for prediction, factor, expected_length in zip(
            scale_predictions, self.scale_factors, self.scale_horizons
        ):
            expected_target = target[:, ::factor, :]
            if prediction.shape != expected_target.shape or prediction.shape[1] != expected_length:
                raise ValueError("each scale prediction must match its stride-sampled target")
            losses.append((prediction - expected_target).abs().mean())
        return torch.stack(losses).sum()
