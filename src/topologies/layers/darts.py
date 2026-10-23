import torch
from torch import nn


class DARTS(nn.Module):
    component_id = "darts"

    def __init__(self, *args, **kwargs):
        super().__init__()
        if args or kwargs:
            raise ValueError("DARTS does not accept constructor arguments")

    def forward(self, candidates, logits):
        if not isinstance(candidates, (list, tuple)) or not candidates:
            raise ValueError("at least one candidate tensor is required")
        if not all(torch.is_tensor(candidate) for candidate in candidates):
            raise ValueError("every candidate must be a tensor")
        if (
            not torch.is_tensor(logits)
            or logits.ndim not in (1, 2)
            or logits.shape[-1] != len(candidates)
        ):
            raise ValueError("expected logits shaped (K,) or (batch, K) for the candidate count")
        if not logits.is_floating_point():
            raise ValueError("logits must have a floating-point dtype")

        reference_shape = candidates[0].shape
        if any(candidate.shape != reference_shape for candidate in candidates[1:]):
            raise ValueError("all candidate tensors must have the same shape")
        if logits.device != candidates[0].device or any(
            candidate.device != candidates[0].device for candidate in candidates[1:]
        ):
            raise ValueError("logits and candidate tensors must be on the same device")

        if logits.ndim == 1:
            stacked_candidates = torch.stack(candidates, dim=0)
            weights = torch.softmax(logits, dim=0)
            weights = weights.reshape((len(candidates),) + (1,) * len(reference_shape))
            return (weights * stacked_candidates).sum(dim=0)

        if not reference_shape or any(candidate.shape[0] != logits.shape[0] for candidate in candidates):
            raise ValueError("batched logits must match the leading candidate batch dimension")
        stacked_candidates = torch.stack(candidates, dim=1)
        weights = torch.softmax(logits, dim=1)
        weights = weights.reshape((logits.shape[0], len(candidates)) + (1,) * (len(reference_shape) - 1))
        return (weights * stacked_candidates).sum(dim=1)
