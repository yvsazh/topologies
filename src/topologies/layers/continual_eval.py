import torch
from torch import nn

_MISSING = object()


class ContinualEvalMetrics(nn.Module):
    component_id = "continual-eval"

    def __init__(self):
        super().__init__()

    def forward(self, accuracy_matrix, initial_accuracy=_MISSING):
        if initial_accuracy is _MISSING:
            if not isinstance(accuracy_matrix, torch.Tensor) or accuracy_matrix.ndim != 2:
                raise ValueError("accuracy_matrix must be a non-empty square tensor")
            task_count, column_count = accuracy_matrix.shape
            if task_count == 0 or column_count != task_count:
                raise ValueError("accuracy_matrix must be a non-empty square tensor")

            last_row = accuracy_matrix[-1]
            average_accuracy = last_row.mean()
            if task_count == 1:
                zero = accuracy_matrix.new_zeros(())
                return {"acc": average_accuracy, "bwt": zero, "forgetting": zero}

            previous_count = task_count - 1
            previous_tasks = torch.arange(previous_count, device=accuracy_matrix.device)
            backward_transfer = (
                last_row[:previous_count] - accuracy_matrix[previous_tasks, previous_tasks]
            ).mean()
            prior_history = accuracy_matrix[:previous_count, :previous_count]
            best_prior = prior_history.max(dim=0).values
            forgetting = (best_prior - last_row[:previous_count]).mean()
            return {"acc": average_accuracy, "bwt": backward_transfer, "forgetting": forgetting}

        if not isinstance(accuracy_matrix, torch.Tensor):
            raise TypeError("accuracy_matrix must be a torch.Tensor")
        if not isinstance(initial_accuracy, torch.Tensor):
            raise TypeError("initial_accuracy must be a torch.Tensor")
        if accuracy_matrix.ndim not in (2, 3):
            raise ValueError("accuracy_matrix must have shape (T, T) or (B, T, T)")
        if accuracy_matrix.shape[-2] != accuracy_matrix.shape[-1]:
            raise ValueError("accuracy_matrix must be square over its final two dimensions")
        if accuracy_matrix.ndim == 3 and accuracy_matrix.shape[0] < 1:
            raise ValueError("accuracy_matrix batch dimension must be nonempty")
        task_count = accuracy_matrix.shape[-1]
        if task_count < 2:
            raise ValueError("at least two tasks are required to compute BWT and FWT")
        expected_baseline_shape = (*accuracy_matrix.shape[:-2], task_count)
        if tuple(initial_accuracy.shape) != expected_baseline_shape:
            raise ValueError(
                "initial_accuracy must have one value per task and match the matrix batch shape"
            )
        if not accuracy_matrix.is_floating_point() or not initial_accuracy.is_floating_point():
            raise ValueError("accuracy inputs must have floating-point dtypes")
        if accuracy_matrix.device != initial_accuracy.device:
            raise ValueError("accuracy_matrix and initial_accuracy must be on the same device")
        if not torch.isfinite(accuracy_matrix).all() or not torch.isfinite(initial_accuracy).all():
            raise ValueError("accuracy inputs must contain only finite values")
        if (
            (accuracy_matrix < 0).any()
            or (accuracy_matrix > 1).any()
            or (initial_accuracy < 0).any()
            or (initial_accuracy > 1).any()
        ):
            raise ValueError("accuracies must lie in [0, 1]")

        average_accuracy = accuracy_matrix[..., -1, :].mean(dim=-1)
        diagonal = torch.diagonal(accuracy_matrix, dim1=-2, dim2=-1)
        backward_transfer = (accuracy_matrix[..., -1, :-1] - diagonal[..., :-1]).mean(dim=-1)
        forward_transfer = (
            torch.diagonal(accuracy_matrix, offset=1, dim1=-2, dim2=-1)
            - initial_accuracy[..., 1:]
        ).mean(dim=-1)
        return torch.stack((average_accuracy, backward_transfer, forward_transfer), dim=-1)


ContinualEval = ContinualEvalMetrics
