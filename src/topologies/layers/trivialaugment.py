import torch
from torch import nn


class TrivialAugment(nn.Module):
    component_id = "trivialaugment"

    def __init__(self, operations=None):
        super().__init__()
        if operations is None:
            operations = (self._identity,)
        elif not isinstance(operations, (list, tuple)):
            raise ValueError("operations must be a non-empty list or tuple of callables")
        if not operations:
            raise ValueError("operations must contain at least one callable")
        if any(not callable(operation) for operation in operations):
            raise ValueError("every operation must be callable")
        self.operations = tuple(operations)

    @staticmethod
    def _identity(image, strength):
        return image

    @staticmethod
    def _sample(high, generator):
        return int(torch.randint(high, (), generator=generator, device=generator.device).item())

    def _apply_operation(self, image, generator):
        operation = self.operations[self._sample(len(self.operations), generator)]
        strength = self._sample(31, generator)
        result = operation(image, strength)
        if not isinstance(result, torch.Tensor) or result.shape != image.shape:
            raise ValueError("each operation must return a tensor with the same shape as its image")
        return result

    def forward(self, x, generator=None):
        if not isinstance(x, torch.Tensor) or x.ndim not in (3, 4):
            raise ValueError("input must be a CHW image or an NCHW batch")
        if generator is None:
            generator = torch.default_generator
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator or None")
        if x.ndim == 3:
            return self._apply_operation(x, generator)
        if x.shape[0] == 0:
            return x
        return torch.stack([self._apply_operation(image, generator) for image in x])
