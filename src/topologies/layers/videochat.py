import torch
from torch import nn


class VideoChat(nn.Module):
    component_id = "videochat"
    PRESETS = {
        "tiny": {"num_segments": 2},
        "small": {"num_segments": 4},
        "base": {"num_segments": 8},
    }

    def __init__(self, num_segments=8):
        super().__init__()
        valid_num_segments = (
            isinstance(num_segments, int)
            and not isinstance(num_segments, bool)
            and num_segments > 0
        )
        if not valid_num_segments:
            raise ValueError("num_segments must be a positive integer")
        self.num_segments = num_segments

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 5:
            raise ValueError("x must be a tensor with shape [B, T, C, H, W]")
        num_frames = x.shape[1]
        if num_frames <= 0:
            raise ValueError("x must contain at least one frame")
        delta = (num_frames - 1) / self.num_segments
        start = int(delta / 2)
        indices = [start + round(delta * index) for index in range(self.num_segments)]
        index_tensor = torch.tensor(indices, dtype=torch.long, device=x.device)
        return x.index_select(1, index_tensor)
