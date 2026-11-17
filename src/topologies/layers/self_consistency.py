import torch
from torch import nn


class SelfConsistency(nn.Module):
    component_id = "self-consistency"

    def __init__(self):
        super().__init__()

    def forward(self, answer_votes):
        if not isinstance(answer_votes, torch.Tensor):
            raise TypeError("answer_votes must be a torch.Tensor")
        if answer_votes.ndim != 3:
            raise ValueError("answer_votes must have shape [batch, samples, answers]")
        if any(dimension < 1 for dimension in answer_votes.shape):
            raise ValueError("answer_votes must have nonempty batch, sample, and answer dimensions")
        if answer_votes.is_complex() or answer_votes.is_quantized or answer_votes.layout != torch.strided:
            raise ValueError("answer_votes must use a dense real numeric dtype")
        if not torch.isfinite(answer_votes).all().item():
            raise ValueError("answer_votes must contain only finite values")
        binary_entries = (answer_votes == 0) | (answer_votes == 1)
        if not binary_entries.all().item():
            raise ValueError("answer_votes must contain only zero and one values")
        if not (answer_votes.sum(dim=-1) == 1).all().item():
            raise ValueError("each sampled answer must have exactly one active answer")
        return answer_votes.sum(dim=1)
