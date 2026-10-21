

from __future__ import annotations

import torch
import torch.nn as nn


class BidirectionalRNNWrapper(nn.Module):


    component_id = "bidirectional-rnn-wrapper"

    def __init__(self, dim: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.fwd_in = nn.Linear(dim, hidden)
        self.fwd_rec = nn.Linear(hidden, hidden)
        self.bwd_in = nn.Linear(dim, hidden)
        self.bwd_rec = nn.Linear(hidden, hidden)
        self.proj = nn.Linear(hidden * 2, dim)

    def _run(self, x: torch.Tensor, li: nn.Linear, lr: nn.Linear) -> torch.Tensor:
        B, T, _ = x.shape
        h = torch.zeros(B, lr.out_features, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            h = torch.tanh(li(x[:, t, :]) + lr(h))
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self._run(x, self.fwd_in, self.fwd_rec)
        b = self._run(x.flip(1), self.bwd_in, self.bwd_rec).flip(1)
        return self.proj(torch.cat([f, b], dim=-1))
