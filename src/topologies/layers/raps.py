import math

import torch
from torch import nn


class RAPS(nn.Module):
    component_id = "raps"

    def __init__(self, alpha=0.1, k_reg=5, lambda_reg=0.01, randomized=True):
        super().__init__()
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be a finite number strictly between 0 and 1")
        if isinstance(k_reg, bool) or not isinstance(k_reg, int) or k_reg < 0:
            raise ValueError("k_reg must be a nonnegative integer")
        if isinstance(lambda_reg, bool) or not isinstance(lambda_reg, (int, float)) or not math.isfinite(lambda_reg) or lambda_reg < 0:
            raise ValueError("lambda_reg must be a finite nonnegative number")
        if randomized is not True and randomized is not False:
            raise ValueError("randomized must be a boolean")
        self.alpha = float(alpha)
        self.k_reg = k_reg
        self.lambda_reg = float(lambda_reg)
        self.randomized = randomized
        self.register_buffer("tau", torch.tensor(float("inf")))

    def _validate_probabilities(self, probabilities):
        if not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 2:
            raise ValueError("probabilities must be a rank-two tensor with shape (batch, classes)")
        if not torch.is_floating_point(probabilities):
            raise ValueError("probabilities must use a floating-point dtype")
        if probabilities.shape[0] < 1 or probabilities.shape[1] < 1:
            raise ValueError("probabilities must have a nonempty batch and class dimension")
        if not torch.isfinite(probabilities).all().item():
            raise ValueError("probabilities must contain only finite values")
        if (probabilities < 0).any().item():
            raise ValueError("probabilities must be nonnegative")
        totals = probabilities.sum(dim=1)
        if not torch.allclose(totals, torch.ones_like(totals), atol=1e-5, rtol=1e-5):
            raise ValueError("probability rows must sum to 1")

    def _uniforms(self, probabilities, uniforms, default_value):
        batch_size = probabilities.shape[0]
        if uniforms is None:
            if default_value == "random":
                return torch.rand(batch_size, device=probabilities.device, dtype=probabilities.dtype)
            return probabilities.new_full((batch_size,), default_value)
        try:
            uniforms = torch.as_tensor(uniforms, device=probabilities.device, dtype=probabilities.dtype)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("uniforms must be a finite tensor with shape (batch,)") from error
        if uniforms.shape != (batch_size,):
            raise ValueError("uniforms must have shape (batch,)")
        if not torch.isfinite(uniforms).all().item() or (uniforms < 0).any().item() or (uniforms > 1).any().item():
            raise ValueError("uniforms must contain values in [0, 1]")
        return uniforms

    def _ordered(self, probabilities):
        if probabilities.shape[1] > 1:
            ordered = torch.sort(probabilities.detach(), dim=1, descending=True, stable=True).values
            if (ordered[:, 1:] == ordered[:, :-1]).any().item():
                tie_order = torch.argsort(
                    torch.rand_like(probabilities), dim=1, descending=True, stable=True
                )
                shuffled = probabilities.gather(1, tie_order)
                relative_order = torch.argsort(shuffled, dim=1, descending=True, stable=True)
                order = tie_order.gather(1, relative_order)
            else:
                order = torch.argsort(probabilities, dim=1, descending=True, stable=True)
        else:
            order = torch.zeros_like(probabilities, dtype=torch.long)
        return probabilities.gather(1, order), order

    def _scores(self, probabilities):
        sorted_probabilities, order = self._ordered(probabilities)
        rank = torch.arange(
            1, probabilities.shape[1] + 1, device=probabilities.device, dtype=probabilities.dtype
        )
        penalty = (rank - self.k_reg).clamp_min(0) * self.lambda_reg
        cumulative = sorted_probabilities.cumsum(dim=1)
        group_starts = torch.ones_like(order, dtype=torch.bool)
        group_starts[:, 1:] = sorted_probabilities[:, 1:] != sorted_probabilities[:, :-1]
        positions = torch.arange(probabilities.shape[1], device=probabilities.device).expand_as(order)
        group_start_indices = torch.where(group_starts, positions, 0).cummax(dim=1).values
        group_cumulative = cumulative.gather(1, group_start_indices)
        group_probabilities = sorted_probabilities.gather(1, group_start_indices)
        sorted_scores = group_cumulative - group_probabilities + penalty
        return torch.zeros_like(probabilities).scatter(1, order, sorted_scores), sorted_probabilities, order

    def forward(self, probabilities):
        self._validate_probabilities(probabilities)
        scores, _, _ = self._scores(probabilities)
        return scores

    def calibration_scores(self, probabilities, labels, uniforms=None):
        self._validate_probabilities(probabilities)
        if not isinstance(labels, torch.Tensor) or labels.ndim != 1 or labels.shape[0] != probabilities.shape[0]:
            raise ValueError("labels must be a rank-one tensor matching the probability batch")
        if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("labels must contain integer class indices")
        labels = labels.to(device=probabilities.device, dtype=torch.long)
        if (labels < 0).any().item() or (labels >= probabilities.shape[1]).any().item():
            raise ValueError("labels must be valid class indices")
        uniform_default = "random" if self.randomized else 0.0
        uniforms = self._uniforms(probabilities, uniforms, uniform_default)
        sorted_probabilities, order = self._ordered(probabilities)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(
            1, probabilities.shape[1] + 1, device=probabilities.device, dtype=order.dtype
        ).expand_as(order)
        ranks.scatter_(1, order, rank_values)
        label_ranks = ranks.gather(1, labels.unsqueeze(1)).squeeze(1)
        rank_indices = label_ranks - 1
        cumulative = sorted_probabilities.cumsum(dim=1)
        cumulative_at_label = cumulative.gather(1, rank_indices.unsqueeze(1)).squeeze(1)
        label_probabilities = sorted_probabilities.gather(1, rank_indices.unsqueeze(1)).squeeze(1)
        penalty = (label_ranks - self.k_reg).clamp_min(0).to(probabilities.dtype) * self.lambda_reg
        calibration_scores = cumulative_at_label - uniforms * label_probabilities + penalty
        return calibration_scores

    def calibrate(self, probabilities, labels, uniforms=None):
        calibration_scores = self.calibration_scores(probabilities, labels, uniforms)
        quantile_rank = math.ceil((probabilities.shape[0] + 1) * (1 - self.alpha))
        if quantile_rank > probabilities.shape[0]:
            threshold = probabilities.new_tensor(float("inf"))
        else:
            threshold = calibration_scores.detach().kthvalue(quantile_rank).values
        with torch.no_grad():
            self.tau.copy_(threshold.to(device=self.tau.device, dtype=self.tau.dtype))
        return self.tau

    def predict_set(self, probabilities, tau=None, uniforms=None):
        self._validate_probabilities(probabilities)
        uniform_default = "random" if self.randomized else 1.0
        uniforms = self._uniforms(probabilities, uniforms, uniform_default)
        if tau is None:
            tau = self.tau
        try:
            tau = torch.as_tensor(tau, device=probabilities.device, dtype=probabilities.dtype)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("tau must be a scalar finite value or positive infinity") from error
        if tau.ndim != 0 or torch.isnan(tau).item() or (torch.isinf(tau) & (tau < 0)).item():
            raise ValueError("tau must be a scalar finite value or positive infinity")
        scores, _, _ = self._scores(probabilities)
        return scores + probabilities * uniforms.unsqueeze(1) <= tau
