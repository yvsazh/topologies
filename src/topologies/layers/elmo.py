import torch
from torch import nn


class ELMoScalarMix(nn.Module):
    component_id = "elmo"

    def __init__(self, num_representations: int, layer_norm: bool = False) -> None:
        super().__init__()
        if (
            isinstance(num_representations, bool)
            or not isinstance(num_representations, int)
            or num_representations < 1
        ):
            raise ValueError("num_representations must be a positive integer")
        if type(layer_norm) is not bool:
            raise ValueError("layer_norm must be a boolean")
        self.num_representations = num_representations
        self.layer_norm = layer_norm
        self.scalar_logits = nn.Parameter(torch.zeros(num_representations))
        self.gamma = nn.Parameter(torch.ones(()))

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.scalar_logits, dim=0)

    def forward(self, representations: torch.Tensor | tuple | list) -> torch.Tensor:
        if isinstance(representations, torch.Tensor):
            if representations.ndim != 4:
                raise ValueError("stacked representations must have rank 4 (K, B, T, D)")
            if representations.shape[0] != self.num_representations:
                if representations.shape[1] != self.num_representations:
                    raise ValueError(
                        f"expected exactly {self.num_representations} representations"
                    )
                stacked = representations.permute(1, 0, 2, 3)
            else:
                stacked = representations
        elif isinstance(representations, (tuple, list)):
            if len(representations) != self.num_representations:
                raise ValueError(
                    f"expected exactly {self.num_representations} representations"
                )
            if not all(isinstance(value, torch.Tensor) for value in representations):
                raise ValueError("each representation must be a torch.Tensor")
            if any(value.ndim != 3 for value in representations):
                raise ValueError("each representation must have rank 3 (B, T, D)")
            reference_shape = representations[0].shape
            if any(value.shape != reference_shape for value in representations[1:]):
                raise ValueError("all representations must have the same shape")
            reference_device = representations[0].device
            reference_dtype = representations[0].dtype
            if any(
                value.device != reference_device or value.dtype != reference_dtype
                for value in representations[1:]
            ):
                raise ValueError("all representations must have the same device and dtype")
            stacked = torch.stack(representations, dim=0)
        else:
            raise TypeError("representations must be a rank-4 tensor or a sequence of tensors")
        if not stacked.is_floating_point():
            raise ValueError("representations must have a floating-point dtype")
        if any(size < 1 for size in stacked.shape):
            raise ValueError("representation dimensions must be non-empty")
        if self.layer_norm:
            stacked = torch.nn.functional.layer_norm(stacked, (stacked.shape[-1],))
        weighted = self.weights.reshape(self.num_representations, 1, 1, 1) * stacked
        return self.gamma * weighted.sum(dim=0)
