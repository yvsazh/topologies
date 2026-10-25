import torch
from torch import nn


class StreamingLLM(nn.Module):
    component_id = "streamingllm"
    PRESETS = {
        "tiny": {"sink_tokens": 4, "window_size": 4},
        "small": {"sink_tokens": 4, "window_size": 32},
        "base": {"sink_tokens": 4, "window_size": 512},
    }

    def __init__(self, sink_tokens: int = 4, window_size: int = 512):
        super().__init__()
        if not isinstance(sink_tokens, int) or isinstance(sink_tokens, bool) or sink_tokens <= 0:
            raise ValueError("sink_tokens must be a positive integer")
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        self.sink_tokens = sink_tokens
        self.window_size = window_size

    def forward(
        self,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
        past_key: torch.Tensor | None = None,
        past_value: torch.Tensor | None = None,
        output_format: str = "cache",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        if output_format not in {"cache", "features"}:
            raise ValueError("output_format must be 'cache' or 'features'")
        if not isinstance(new_key, torch.Tensor) or not isinstance(new_value, torch.Tensor):
            raise TypeError("new_key and new_value must be tensors")
        if new_key.ndim != 4 or new_value.ndim != 4:
            raise ValueError("new_key and new_value must have shape [B, H, T, D]")
        if new_key.shape != new_value.shape:
            raise ValueError("new_key and new_value must have the same shape")
        if new_key.device != new_value.device:
            raise ValueError("new_key and new_value must be on the same device")
        if (past_key is None) != (past_value is None):
            raise ValueError("past_key and past_value must be supplied together")

        if past_key is None:
            key = new_key
            value = new_value
        else:
            if not isinstance(past_key, torch.Tensor) or not isinstance(past_value, torch.Tensor):
                raise ValueError("past_key and past_value must be tensors")
            if past_key.ndim != 4 or past_value.ndim != 4:
                raise ValueError("past_key and past_value must have shape [B, H, L, D]")
            if past_key.shape != past_value.shape:
                raise ValueError("past_key and past_value must have the same shape")
            if past_key.shape[:2] != new_key.shape[:2] or past_key.shape[3] != new_key.shape[3]:
                raise ValueError("past and new keys and values must match in batch, heads, and width")
            if past_key.device != new_key.device or past_value.device != new_value.device:
                raise ValueError("past and new keys and values must be on matching devices")
            key = torch.cat((past_key, new_key), dim=2)
            value = torch.cat((past_value, new_value), dim=2)

        total_length = key.shape[2]
        capacity = self.sink_tokens + self.window_size
        if total_length > capacity:
            recent_start = max(self.sink_tokens, total_length - self.window_size)
            key = torch.cat((key[:, :, : self.sink_tokens, :], key[:, :, recent_start:, :]), dim=2)
            value = torch.cat(
                (value[:, :, : self.sink_tokens, :], value[:, :, recent_start:, :]), dim=2
            )

        if output_format == "features":
            return torch.cat((key, value), dim=-1)

        cache_positions = torch.arange(key.shape[2], device=key.device)
        return key, value, cache_positions
