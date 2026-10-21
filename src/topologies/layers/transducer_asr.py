import torch
from torch import nn
import torch.nn.functional as F


def attend(scores, values):
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights.unsqueeze(-2), values).squeeze(-2)


def location_convolution(previous, kernel):
    return F.conv1d(previous.view(1, 1, -1), kernel.view(1, 1, -1), padding=kernel.numel() // 2).view(-1)


def transducer_join(encoder, predictor):
    return torch.tanh(encoder + predictor)


def zipformer_pool(values):
    if values.shape[-2] % 2:
        raise ValueError("time length must be even")
    return values.reshape(*values.shape[:-2], values.shape[-2] // 2, 2, values.shape[-1]).mean(-2)


class TransducerASR(nn.Module):
    component_id = "transducer-asr"
    PRESETS = {
        "tiny": {"width": 4, "mode": "las"},
        "base": {"width": 8, "mode": "rnnt"},
    }

    def __init__(self, width=4, mode="las", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"las", "rnnt", "zipformer"}:
            raise ValueError("unknown transducer mode")
        if width <= 0:
            raise ValueError("invalid transducer width")
        self.width = width
        self.mode = mode

    def forward(self, encoder):
        encoder = torch.as_tensor(encoder, dtype=torch.float32)
        if encoder.ndim != 3 or encoder.shape[-1] != self.width:
            raise ValueError("encoder must have shape (batch, time, width)")
        if self.mode == "zipformer":
            return zipformer_pool(encoder)
        if self.mode == "rnnt":
            predictor = torch.zeros_like(encoder)
            return transducer_join(encoder, predictor)
        scores = encoder.sum(-1)
        return attend(scores, encoder)
