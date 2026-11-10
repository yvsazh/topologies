import torch
from torch import nn


def _value_error(message: str) -> ValueError:
    return ValueError(message)


def _as_tensor(value, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise _value_error(f"{name} must be a tensor or tensor-like values") from error


class MovementPruning(nn.Module):
    component_id = "movement-pruning"

    def __init__(self, initial_weight: torch.Tensor, initial_scores: torch.Tensor, keep_count: int):
        super().__init__()
        initial_weight = _as_tensor(initial_weight, "initial_weight")
        initial_scores = _as_tensor(initial_scores, "initial_scores")
        if initial_weight.ndim != 2:
            raise _value_error("initial_weight must be a two-dimensional matrix")
        if initial_scores.shape != initial_weight.shape:
            raise _value_error("initial_scores must have the same shape as initial_weight")
        if not initial_weight.is_floating_point() or not initial_scores.is_floating_point():
            raise _value_error("initial_weight and initial_scores must be floating-point tensors")
        if initial_weight.dtype != initial_scores.dtype:
            raise _value_error("initial_weight and initial_scores must have the same dtype")
        if initial_weight.device != initial_scores.device:
            raise _value_error("initial_weight and initial_scores must be on the same device")
        if isinstance(keep_count, bool) or not isinstance(keep_count, int):
            raise _value_error("keep_count must be an integer")
        if not 0 <= keep_count <= initial_weight.numel():
            raise _value_error(
                f"keep_count must be between 0 and {initial_weight.numel()}"
            )
        self.weight = nn.Parameter(initial_weight.detach().clone())
        self.scores = nn.Parameter(initial_scores.detach().clone())
        self._keep_count = keep_count

    @property
    def keep_count(self) -> int:
        return self._keep_count

    @keep_count.setter
    def keep_count(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _value_error("keep_count must be an integer")
        if not 0 <= value <= self.weight.numel():
            raise _value_error(f"keep_count must be between 0 and {self.weight.numel()}")
        self._keep_count = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise _value_error("input must be a torch.Tensor")
        if x.ndim < 1:
            raise _value_error("input must have at least one dimension")
        if x.shape[-1] != self.weight.shape[1]:
            raise _value_error(
                f"input's final dimension must be {self.weight.shape[1]}"
            )
        flat_scores = self.scores.reshape(-1)
        hard_mask = torch.zeros_like(flat_scores)
        if self.keep_count:
            indices = torch.argsort(flat_scores.detach(), descending=True, stable=True)
            hard_mask = hard_mask.scatter(0, indices[:self.keep_count], 1.0)
        hard_mask = hard_mask.reshape_as(self.scores)
        mask = hard_mask + (self.scores - self.scores.detach())
        return torch.matmul(x, (self.weight * mask).transpose(0, 1))
