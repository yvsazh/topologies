import torch
from torch import nn


class ConvTasNetMaskApplication(nn.Module):
    component_id = "source-separation"

    def forward(self, encoded, masks):
        encoded = torch.as_tensor(encoded)
        masks = torch.as_tensor(masks, dtype=encoded.dtype, device=encoded.device)
        if encoded.ndim != 3 or masks.ndim != 4:
            raise ValueError("encoded must be (batch, channels, frames) and masks (batch, sources, channels, frames)")
        if masks.shape[0] != encoded.shape[0] or masks.shape[2:] != encoded.shape[1:]:
            raise ValueError("masks must match the encoded batch, channel, and frame dimensions")
        return masks * encoded.unsqueeze(1)
