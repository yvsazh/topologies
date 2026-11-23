import torch
from torch import nn


class JointEmbedding(nn.Module):
    component_id = "vilt"
    PRESETS = {
        "tiny": {"dim": 8},
        "small": {"dim": 32},
        "base": {"dim": 768},
    }

    def __init__(self, dim: int) -> None:
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = dim
        self.text_type = nn.Parameter(torch.zeros(dim))
        self.patch_type = nn.Parameter(torch.zeros(dim))

    def forward(self, text: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        if not isinstance(text, torch.Tensor) or not isinstance(patches, torch.Tensor):
            raise TypeError("text and patches must be tensors")
        if text.ndim < 2 or patches.ndim < 2:
            raise ValueError("text and patches must have sequence and feature dimensions")
        if text.shape[-1] != self.dim or patches.shape[-1] != self.dim:
            raise ValueError("text and patches last dimension must match dim")
        if text.shape[:-2] != patches.shape[:-2]:
            raise ValueError("text and patches leading dimensions must match")
        return torch.cat((text + self.text_type, patches + self.patch_type), dim=-2)
