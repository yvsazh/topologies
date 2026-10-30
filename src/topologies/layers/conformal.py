import math
from numbers import Real

import torch
from torch import nn


class Conformal(nn.Module):
    component_id = "conformal"
    normalization_tolerance = 1e-6

    def __init__(self, alpha=0.1):
        super().__init__()
        if (not isinstance(alpha, Real) or isinstance(alpha, bool)
                or not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0):
            raise ValueError("alpha must be a finite number in (0, 1)")
        self.alpha = float(alpha)
        self.calibration_rank = None

    def _validate_probabilities(self, probs):
        if (not isinstance(probs, torch.Tensor) or probs.ndim != 2
                or probs.shape[0] == 0 or probs.shape[1] == 0
                or not probs.is_floating_point()):
            raise ValueError("probs must be a nonempty floating tensor shaped [B, C]")
        if not bool(torch.isfinite(probs).all()):
            raise ValueError("probs must contain only finite values")
        if not bool(((probs >= 0.0) & (probs <= 1.0)).all()):
            raise ValueError("probabilities must be between 0 and 1")
        tolerance = max(self.normalization_tolerance, torch.finfo(probs.dtype).eps * 4.0)
        row_sums = probs.sum(dim=-1)
        if not bool(torch.isclose(row_sums, torch.ones_like(row_sums), atol=tolerance, rtol=0.0).all()):
            raise ValueError("probability rows must sum to 1 within the normalization tolerance")

    def _validate_uniforms(self, uniforms, batch_size):
        if not torch.is_tensor(uniforms):
            raise ValueError("uniforms must be a floating tensor shaped [B]")
        if (uniforms.ndim != 1 or uniforms.shape[0] != batch_size
                or not uniforms.is_floating_point()):
            raise ValueError("uniforms must be a floating tensor shaped [B]")
        if (not bool(torch.isfinite(uniforms).all())
                or not bool(((uniforms >= 0.0) & (uniforms <= 1.0)).all())):
            raise ValueError("uniforms must contain finite values between 0 and 1")

    def _prepare_uniforms(self, uniforms, batch_size):
        if (torch.is_tensor(uniforms) and uniforms.ndim == 2
                and uniforms.shape == (batch_size, 1)):
            uniforms = uniforms.squeeze(-1)
        self._validate_uniforms(uniforms, batch_size)
        return uniforms

    def forward(self, probs, uniforms):
        self._validate_probabilities(probs)
        uniforms = self._prepare_uniforms(uniforms, probs.shape[0])
        uniforms = uniforms.to(device=probs.device, dtype=probs.dtype)
        uniforms = uniforms.unsqueeze(-1)
        order = torch.argsort(probs, dim=-1, descending=True, stable=True)
        sorted_probs = probs.gather(-1, order)
        sorted_scores = sorted_probs.cumsum(dim=-1) - uniforms * sorted_probs
        return torch.zeros_like(probs).scatter(-1, order, sorted_scores)

    def calibrate(self, calibration_probs, calibration_labels, calibration_uniforms):
        self._validate_probabilities(calibration_probs)
        count, class_count = calibration_probs.shape
        if (not isinstance(calibration_labels, torch.Tensor) or calibration_labels.ndim != 1
                or calibration_labels.shape[0] != count
                or calibration_labels.dtype not in (torch.uint8, torch.int8, torch.int16,
                                                    torch.int32, torch.int64)):
            raise ValueError("calibration_labels must be an integer tensor shaped [N]")
        if not bool(((calibration_labels >= 0) & (calibration_labels < class_count)).all()):
            raise ValueError("calibration_labels must be in [0, C)")
        self._validate_uniforms(calibration_uniforms, count)
        rank = math.ceil((1.0 - self.alpha) * (count + 1))
        if rank > count:
            raise ValueError("calibration order-statistic rank exceeds the number of examples")
        labels = calibration_labels.to(device=calibration_probs.device, dtype=torch.long)
        scores = self.forward(calibration_probs, calibration_uniforms)
        true_label_scores = scores.gather(-1, labels.unsqueeze(-1)).squeeze(-1).detach()
        threshold = torch.sort(true_label_scores, stable=True).values[rank - 1]
        self.calibration_rank = rank
        return threshold

    def prediction_set(self, probs, uniforms, threshold):
        self._validate_probabilities(probs)
        uniforms = self._prepare_uniforms(uniforms, probs.shape[0])
        if isinstance(threshold, torch.Tensor):
            if threshold.numel() != 1 or not threshold.is_floating_point():
                raise ValueError("threshold must be a finite number in [0, 1]")
            threshold = float(threshold.detach().item())
        if (not isinstance(threshold, Real) or isinstance(threshold, bool)
                or not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0):
            raise ValueError("threshold must be a finite number in [0, 1]")
        threshold = float(threshold)
        order = torch.argsort(probs, dim=-1, descending=True, stable=True)
        sorted_probs = probs.gather(-1, order)
        cumulative = sorted_probs.cumsum(dim=-1)
        lengths = (cumulative < threshold).sum(dim=-1) + 1
        lengths = lengths.clamp(min=1)
        positive_counts = (sorted_probs > 0.0).sum(dim=-1).clamp(min=1)
        lengths = torch.minimum(lengths, positive_counts)
        rows = torch.arange(probs.shape[0], device=probs.device)
        rank_indices = lengths - 1
        cumulative_at_length = cumulative[rows, rank_indices]
        probability_at_length = sorted_probs[rows, rank_indices]
        boundary = ((cumulative_at_length - threshold) / probability_at_length).clamp(0.0, 1.0)
        uniforms = uniforms.to(device=probs.device, dtype=probs.dtype)
        included_counts = torch.where(uniforms <= boundary, lengths - 1, lengths)
        positions = torch.arange(probs.shape[1], device=probs.device).unsqueeze(0)
        sorted_membership = positions < included_counts.unsqueeze(-1)
        return torch.zeros_like(probs, dtype=torch.bool).scatter(-1, order, sorted_membership)
