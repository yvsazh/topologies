import math

import torch
from torch import nn


def _raise_value_error(message: str) -> None:
    raise ValueError(message)


class TASB(nn.Module):
    component_id = "tas-b"

    def __init__(self, pairwise_weight: float = 1.0, in_batch_weight: float = 0.75):
        super().__init__()
        for name, weight in (("pairwise_weight", pairwise_weight), ("in_batch_weight", in_batch_weight)):
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                _raise_value_error(f"{name} must be a finite non-negative number")
            try:
                valid = math.isfinite(weight) and weight >= 0
            except OverflowError:
                valid = False
            if not valid:
                _raise_value_error(f"{name} must be a finite non-negative number")
        self.pairwise_weight = float(pairwise_weight)
        self.in_batch_weight = float(in_batch_weight)

    def forward(
        self,
        student_pair_pos: torch.Tensor,
        student_pair_neg: torch.Tensor,
        teacher_pair_pos: torch.Tensor,
        teacher_pair_neg: torch.Tensor,
        student_inbatch_pos: torch.Tensor,
        student_inbatch_neg: torch.Tensor,
        teacher_inbatch_pos: torch.Tensor,
        teacher_inbatch_neg: torch.Tensor,
    ) -> torch.Tensor:
        values = (
            ("student_pair_pos", student_pair_pos),
            ("student_pair_neg", student_pair_neg),
            ("teacher_pair_pos", teacher_pair_pos),
            ("teacher_pair_neg", teacher_pair_neg),
            ("student_inbatch_pos", student_inbatch_pos),
            ("student_inbatch_neg", student_inbatch_neg),
            ("teacher_inbatch_pos", teacher_inbatch_pos),
            ("teacher_inbatch_neg", teacher_inbatch_neg),
        )
        for name, value in values:
            if not isinstance(value, torch.Tensor):
                _raise_value_error(f"{name} must be a torch.Tensor")

        if student_pair_pos.ndim not in (1, 2):
            raise ValueError("pair score tensors must have shape [B] or [N, B]")
        outer_batch = student_pair_pos.ndim == 2
        pair_dimensions = 2 if outer_batch else 1
        if any(value.ndim != pair_dimensions or value.shape != student_pair_pos.shape
               for _, value in values[:4]):
            raise ValueError("pair score tensors must have matching shape [B] or [N, B]")

        matrix_values = values[4:]
        if not outer_batch and all(value.ndim == 3 and value.shape[0] == 1
                                   for _, value in matrix_values):
            matrices = tuple(value.squeeze(0) for _, value in matrix_values)
            student_inbatch_pos, student_inbatch_neg, teacher_inbatch_pos, teacher_inbatch_neg = matrices
            values = values[:4] + tuple((name, value) for (name, _), value in zip(matrix_values, matrices))

        batch_size = student_pair_pos.shape[-1]
        if batch_size < 1:
            raise ValueError("score inputs must contain at least one batch item")
        matrix_dimensions = 3 if outer_batch else 2
        expected_matrix_shape = ((student_pair_pos.shape[0], batch_size, batch_size)
                                 if outer_batch else (batch_size, batch_size))
        if any(value.ndim != matrix_dimensions or value.shape != expected_matrix_shape
               for _, value in values[4:]):
            shape_text = "[N, B, B]" if outer_batch else "[B, B]"
            raise ValueError(f"in-batch score tensors must have matching shape {shape_text}")

        teacher_pair_margin = teacher_pair_pos.detach() - teacher_pair_neg.detach()
        student_pair_margin = student_pair_pos - student_pair_neg
        pairwise_error = (student_pair_margin - teacher_pair_margin).square()
        pairwise_loss = pairwise_error.mean(dim=-1) if outer_batch else pairwise_error.mean()

        if batch_size == 1:
            in_batch_loss = pairwise_loss.new_zeros(pairwise_loss.shape)
        else:
            off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=student_pair_pos.device)
            teacher_inbatch_pos = teacher_inbatch_pos.detach()
            teacher_inbatch_neg = teacher_inbatch_neg.detach()
            teacher_anchor = teacher_inbatch_pos.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
            positive_student_margin = student_pair_pos.unsqueeze(-1) - student_inbatch_pos
            positive_teacher_margin = teacher_anchor - teacher_inbatch_pos
            negative_student_margin = student_pair_pos.unsqueeze(-1) - student_inbatch_neg
            negative_teacher_margin = teacher_anchor - teacher_inbatch_neg
            positive_error = (positive_student_margin - positive_teacher_margin).square()
            negative_error = (negative_student_margin - negative_teacher_margin).square()
            if outer_batch:
                positive_loss = positive_error[:, off_diagonal].mean(dim=-1)
                negative_loss = negative_error[:, off_diagonal].mean(dim=-1)
            else:
                positive_loss = positive_error[off_diagonal].mean()
                negative_loss = negative_error[off_diagonal].mean()
            in_batch_loss = 0.5 * (positive_loss + negative_loss)

        return self.pairwise_weight * pairwise_loss + self.in_batch_weight * in_batch_loss
