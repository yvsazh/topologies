import torch
from torch import nn
import torch.nn.functional as F


def conv_length(length, kernel, stride, padding):
    return (length + 2 * padding - kernel) // stride + 1


def time_channel_separable(value, depthwise, pointwise):
    mixed = F.conv1d(value, depthwise, padding=depthwise.shape[-1] // 2, groups=value.shape[1])
    return F.conv1d(mixed, pointwise)


class ConvASR(nn.Module):
    architecture_id = "conv-asr"
    PRESETS = {
        "tiny": {"vocab_size": 5, "in_channels": 1, "width": 8, "mode": "quartznet", "blank_id": 4},
        "base": {"vocab_size": 5, "in_channels": 1, "width": 16, "mode": "deepspeech", "blank_id": 4},
    }

    def __init__(self, vocab_size=5, in_channels=1, width=8, mode="quartznet", blank_id=4, dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"quartznet", "deepspeech"}:
            raise ValueError("unknown asr mode")
        if min(vocab_size, in_channels, width) <= 0 or not 0 <= blank_id < vocab_size:
            raise ValueError("invalid asr dimensions")
        self.vocab_size = vocab_size
        self.in_channels = in_channels
        self.width = width
        self.mode = mode
        self.blank_id = blank_id
        self.context_length = 64
        self.stem = nn.Conv1d(in_channels, width, 3, padding=1, bias=False)
        self.depthwise = nn.Conv1d(width, width, 3, padding=1, groups=width, bias=False)
        self.pointwise = nn.Conv1d(width, width, 1, bias=False)
        self.rnn = nn.GRU(width, width, batch_first=True, bidirectional=True)
        out_width = width * 2 if mode == "deepspeech" else width
        self.fc = nn.Linear(out_width, vocab_size)

    def output_lengths(self, lengths):
        return conv_length(lengths, 3, 1, 1)

    def forward(self, features, lengths):
        features = torch.as_tensor(features, dtype=torch.float32)
        lengths = torch.as_tensor(lengths)
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3 or features.shape[1] != self.in_channels:
            raise ValueError("features must have shape (batch, in_channels, time)")
        hidden = F.relu(self.stem(features))
        if self.mode == "quartznet":
            hidden = hidden + time_channel_separable(hidden, self.depthwise.weight, self.pointwise.weight)
            hidden = hidden.transpose(1, 2)
        else:
            hidden, _ = self.rnn(hidden.transpose(1, 2))
        return self.fc(hidden), self.output_lengths(lengths)
