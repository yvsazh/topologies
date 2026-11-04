import math

import torch
from torch import nn


class EarlyExit(nn.Module):
    component_id = "early-exit"

    def __init__(self, threshold):
        super().__init__()
        if type(threshold) not in (int, float):
            raise ValueError("threshold must be a finite nonnegative real number")
        try:
            threshold = float(threshold)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("threshold must be a finite nonnegative real number") from error
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be a finite nonnegative real number")
        self.threshold = threshold

    def forward(self, exit_distributions):
        single_tensor_exit = isinstance(exit_distributions, torch.Tensor)
        if single_tensor_exit:
            if exit_distributions.ndim == 1:
                exit_distributions = exit_distributions.unsqueeze(0)
            iterator = iter((exit_distributions,))
        else:
            try:
                iterator = iter(exit_distributions)
            except TypeError as error:
                raise ValueError("exit_distributions must be a tensor or iterable of tensors") from error

        final_distribution = None
        final_index = -1
        for index, distribution in enumerate(iterator):
            if not isinstance(distribution, torch.Tensor) or distribution.ndim != 2:
                raise ValueError("each exit distribution must be a rank-two torch.Tensor")
            if distribution.shape[0] < 1 or distribution.shape[1] < 2:
                raise ValueError("each exit distribution must have shape [B, C] with B at least 1 and C at least 2")
            if distribution.shape[0] != 1 and not single_tensor_exit:
                raise ValueError("streamed exit distributions must have batch size one")
            if not distribution.is_floating_point():
                raise ValueError("each exit distribution must have a floating-point dtype")
            if not torch.isfinite(distribution).all():
                raise ValueError("each exit distribution must contain only finite probabilities")
            if (distribution < 0).any() or (distribution > 1).any():
                raise ValueError("each exit distribution must contain probabilities in [0, 1]")
            if not torch.allclose(
                distribution.sum(dim=-1),
                torch.ones(1, dtype=distribution.dtype, device=distribution.device),
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("each exit distribution must sum to one")

            final_distribution = distribution
            final_index = index
            if single_tensor_exit:
                return distribution
            positive = distribution > 0
            safe_log = distribution.clamp_min(torch.finfo(distribution.dtype).tiny).log()
            entropy = -(distribution * safe_log * positive).sum()
            if entropy.item() < self.threshold:
                return distribution, index

        if final_distribution is None:
            raise ValueError("exit_distributions must contain at least one probability distribution")
        return final_distribution, final_index
