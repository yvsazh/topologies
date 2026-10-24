import math

import torch
from torch import nn


class LotteryTicket(nn.Module):
    component_id = "lottery-ticket"

    def __init__(self, sparsity: float = 0.5) -> None:
        super().__init__()
        value_error = ValueError
        if isinstance(sparsity, bool) or not isinstance(sparsity, (int, float)):
            raise value_error("sparsity must be a finite number in [0, 1]")
        try:
            sparsity_value = float(sparsity)
        except OverflowError:
            raise value_error("sparsity must be a finite number in [0, 1]") from None
        if not math.isfinite(sparsity_value) or not 0.0 <= sparsity_value <= 1.0:
            raise value_error("sparsity must be a finite number in [0, 1]")
        self.sparsity = sparsity_value

    def forward(
        self,
        trained_weights: torch.Tensor,
        initial_weights: torch.Tensor,
    ) -> torch.Tensor:
        value_error = ValueError
        if not isinstance(trained_weights, torch.Tensor):
            raise value_error("trained_weights must be a torch.Tensor")
        if not isinstance(initial_weights, torch.Tensor):
            raise value_error("initial_weights must be a torch.Tensor")
        if trained_weights.shape != initial_weights.shape:
            raise value_error("trained_weights and initial_weights must have the same shape")
        if trained_weights.device != initial_weights.device:
            raise value_error("trained_weights and initial_weights must be on the same device")
        if trained_weights.dtype == torch.bool:
            raise value_error("trained_weights must have a numeric dtype")

        flattened = trained_weights.detach().reshape(-1)
        retained_count = flattened.numel() - math.floor(self.sparsity * flattened.numel())
        indices = torch.argsort(flattened.abs(), descending=True, stable=True)
        mask = torch.zeros_like(initial_weights).reshape(-1)
        mask[indices[:retained_count]] = 1
        return initial_weights * mask.reshape(initial_weights.shape)
