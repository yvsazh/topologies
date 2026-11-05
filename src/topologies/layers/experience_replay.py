import random

import torch
from torch import nn


class ExperienceReplay(nn.Module):
    component_id = "experience-replay"

    def __init__(self, capacity_per_task, seed=None):
        super().__init__()
        if (
            isinstance(capacity_per_task, bool)
            or not isinstance(capacity_per_task, int)
            or capacity_per_task <= 0
        ):
            raise ValueError("capacity_per_task must be a positive integer")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or None")
        self.capacity_per_task = capacity_per_task
        self.memories = {}
        self._rng = random.Random(seed)

    @property
    def memory_counts(self):
        return {task_id: inputs.shape[0] for task_id, (inputs, _) in self.memories.items()}

    def add(self, task_id, inputs, labels):
        try:
            hash(task_id)
        except TypeError as error:
            raise ValueError("task_id must be hashable") from error
        if not isinstance(inputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("inputs and labels must be tensors")
        if inputs.ndim == 0 or labels.ndim == 0:
            raise ValueError("inputs and labels must have a leading example dimension")
        if inputs.shape[0] != labels.shape[0]:
            raise ValueError("inputs and labels must contain the same number of examples")
        if inputs.shape[0] == 0:
            raise ValueError("inputs and labels must contain at least one example")
        if inputs.device != labels.device:
            raise ValueError("inputs and labels must be on the same device")
        for stored_inputs, stored_labels in self.memories.values():
            if inputs.shape[1:] != stored_inputs.shape[1:]:
                raise ValueError("all task inputs must have matching example shapes")
            if labels.shape[1:] != stored_labels.shape[1:]:
                raise ValueError("all task labels must have matching example shapes")
            if inputs.dtype != stored_inputs.dtype or labels.dtype != stored_labels.dtype:
                raise ValueError("all task tensors must have matching dtypes")
            if inputs.device != stored_inputs.device:
                raise ValueError("all task tensors must be on the same device")
        selected = self._rng.sample(
            range(inputs.shape[0]), min(inputs.shape[0], self.capacity_per_task)
        )
        indices = torch.tensor(selected, dtype=torch.long, device=inputs.device)
        self.memories[task_id] = (
            inputs.detach().index_select(0, indices).clone(),
            labels.detach().index_select(0, indices).clone(),
        )

    def sample(self, batch_size):
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not self.memories:
            raise ValueError("cannot sample from empty episodic memory")
        inputs = torch.cat([task_inputs for task_inputs, _ in self.memories.values()], dim=0)
        labels = torch.cat([task_labels for _, task_labels in self.memories.values()], dim=0)
        if batch_size > inputs.shape[0]:
            raise ValueError("batch_size cannot exceed the number of stored examples")
        selected = self._rng.sample(range(inputs.shape[0]), batch_size)
        indices = torch.tensor(selected, dtype=torch.long, device=inputs.device)
        return inputs.index_select(0, indices), labels.index_select(0, indices)

    def forward(self, gradient, reference_gradient):
        if not isinstance(gradient, torch.Tensor) or not isinstance(reference_gradient, torch.Tensor):
            raise TypeError("gradient and reference_gradient must be tensors")
        if gradient.shape != reference_gradient.shape:
            raise ValueError("gradient and reference_gradient must have the same shape")
        if gradient.ndim == 0:
            raise ValueError("gradient and reference_gradient must contain vectors")
        if gradient.dtype != reference_gradient.dtype:
            raise ValueError("gradient and reference_gradient must have the same dtype")
        if gradient.device != reference_gradient.device:
            raise ValueError("gradient and reference_gradient must be on the same device")
        if not torch.is_floating_point(gradient):
            raise ValueError("gradient and reference_gradient must have a floating-point dtype")
        dot = (gradient * reference_gradient).sum(dim=-1, keepdim=True)
        conflicting = dot < 0
        if not conflicting.any():
            return gradient
        norm = (reference_gradient * reference_gradient).sum(dim=-1, keepdim=True)
        safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))
        coefficient = torch.where(conflicting, dot / safe_norm, torch.zeros_like(dot))
        return gradient - coefficient * reference_gradient
