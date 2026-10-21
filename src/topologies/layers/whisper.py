import torch
from torch import nn

SOT = 50258
ENGLISH = 50259
TRANSLATE = 50358
TRANSCRIBE = 50359
NO_TIMESTAMPS = 50363
TIMESTAMP_BEGIN = 50364


def decoder_prompt(language, task, no_timestamps=True):
    tokens = [SOT, language, task]
    if no_timestamps:
        tokens.append(NO_TIMESTAMPS)
    return torch.tensor(tokens)


def timestamp_id(seconds):
    return TIMESTAMP_BEGIN + int(round(seconds / 0.02))


def mel_frames(samples, hop=160):
    return samples // hop


def select_layers(states, indices):
    return states.index_select(0, torch.as_tensor(indices))


class Whisper(nn.Module):
    component_id = "whisper"
    PRESETS = {
        "tiny": {"width": 8, "mode": "transcribe"},
        "base": {"width": 16, "mode": "translate"},
    }

    def __init__(self, width=8, mode="transcribe", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"transcribe", "translate"}:
            raise ValueError("unknown whisper mode")
        if width <= 0:
            raise ValueError("invalid whisper width")
        self.width = width
        self.mode = mode
        self.task_id = TRANSCRIBE if mode == "transcribe" else TRANSLATE

    def forward(self, mel):
        mel = torch.as_tensor(mel, dtype=torch.float32)
        if mel.ndim != 3 or mel.shape[1] != 80:
            raise ValueError("mel must have shape (batch, 80, time)")
        return mel.mean(dim=1)
