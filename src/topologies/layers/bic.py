import torch
from torch import nn


class BiasCorrection(nn.Module):
    component_id = "bic"
    def __init__(self):
        super().__init__()

    def forward(self, logits, new_class_start, alpha, beta):
        return torch.cat(
            (
                logits[..., :new_class_start],
                alpha * logits[..., new_class_start:] + beta,
            ),
            dim=-1,
        )
