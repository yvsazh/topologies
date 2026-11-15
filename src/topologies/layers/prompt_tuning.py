import math

import torch
from torch import nn


class PromptTuning(nn.Module):
    component_id = "prompt-tuning"
    PRESETS = {
        "tiny": {"prompt_len": 1, "embed_dim": 8, "init_range": 0.5},
        "small": {"prompt_len": 5, "embed_dim": 32, "init_range": 0.5},
    }

    def __init__(self, prompt_len=100, embed_dim=None, init_range=0.5, initial_prompt=None):
        super().__init__()
        if isinstance(prompt_len, bool) or not isinstance(prompt_len, int) or prompt_len <= 0:
            raise ValueError("prompt_len must be a positive integer")
        if isinstance(embed_dim, bool) or not isinstance(embed_dim, int) or embed_dim is None or embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer")
        if (
            isinstance(init_range, bool)
            or not isinstance(init_range, (int, float))
            or not math.isfinite(init_range)
            or init_range < 0
        ):
            raise ValueError("init_range must be a finite nonnegative number")
        if initial_prompt is not None:
            if not isinstance(initial_prompt, torch.Tensor):
                raise ValueError("initial_prompt must be a tensor with shape [prompt_len, embed_dim]")
            if tuple(initial_prompt.shape) != (prompt_len, embed_dim):
                raise ValueError("initial_prompt must have shape [prompt_len, embed_dim]")
            if not initial_prompt.is_floating_point():
                raise ValueError("initial_prompt must have a floating point dtype")
            prompt = initial_prompt.detach().clone()
        else:
            prompt = torch.empty(prompt_len, embed_dim)
            nn.init.uniform_(prompt, -init_range, init_range)
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        self.init_range = float(init_range)
        self.prompt = nn.Parameter(prompt)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a tensor with shape [batch, sequence, embed_dim]")
        if x.ndim != 3:
            raise ValueError("x must have rank 3 with shape [batch, sequence, embed_dim]")
        if x.shape[-1] != self.embed_dim:
            raise ValueError(f"x last dimension must equal embed_dim ({self.embed_dim})")
        prompt = self.prompt.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat((prompt, x), dim=1)
