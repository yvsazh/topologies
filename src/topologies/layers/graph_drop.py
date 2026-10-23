import math
from numbers import Real

import torch
from torch import nn


class DropEdge(nn.Module):
    component_id = "graph-drop"

    def __init__(self, rate: float):
        super().__init__()
        try:
            value = float(rate)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("rate must be a finite number in [0, 1]") from error
        if isinstance(rate, bool) or not isinstance(rate, Real) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("rate must be a finite number in [0, 1]")
        self.rate = value

    def forward(self, adjacency: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        if not isinstance(adjacency, torch.Tensor):
            raise TypeError("adjacency must be a dense floating-point tensor")
        if adjacency.layout != torch.strided or not torch.is_floating_point(adjacency):
            raise ValueError("adjacency must be a dense floating-point tensor")
        if adjacency.ndim not in (2, 3):
            raise ValueError("adjacency must have shape [N, N] or [B, N, N]")
        if adjacency.shape[-2] != adjacency.shape[-1]:
            raise ValueError("adjacency matrices must be square")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise ValueError("generator must be a torch.Generator or None")
        if not self.training or self.rate == 0:
            return adjacency

        matrices = adjacency.unsqueeze(0) if adjacency.ndim == 2 else adjacency
        if matrices.shape[0] == 0:
            return adjacency
        diagonal = torch.eye(matrices.shape[-1], dtype=torch.bool, device=adjacency.device)
        outputs = []
        for matrix in matrices:
            eligible = torch.nonzero((matrix != 0) & ~diagonal, as_tuple=False)
            drop_count = math.floor(self.rate * eligible.shape[0])
            if drop_count == 0:
                outputs.append(matrix)
                continue
            permutation = torch.randperm(eligible.shape[0], generator=generator, device=adjacency.device)
            removed = eligible[permutation[:drop_count]]
            keep = torch.ones_like(matrix, dtype=torch.bool)
            keep[removed[:, 0], removed[:, 1]] = False
            outputs.append(matrix * keep.to(dtype=matrix.dtype))
        result = torch.stack(outputs)
        return result.squeeze(0) if adjacency.ndim == 2 else result
