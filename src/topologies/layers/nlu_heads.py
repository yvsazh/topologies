import torch
from torch import nn


class NLUHeads(nn.Module):
    component_id = "nlu-heads"

    def __init__(self, d_model=32, num_labels=2, mode="token", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"token", "span", "choice", "pool"}:
            raise ValueError("mode must be token, span, choice, or pool")
        if min(d_model, num_labels) <= 0:
            raise ValueError("invalid head dimensions")
        self.d_model = d_model
        self.num_labels = num_labels
        self.mode = mode
        self.token = nn.Linear(d_model, num_labels, bias=False)
        self.start = nn.Linear(d_model, 1, bias=False)
        self.end = nn.Linear(d_model, 1, bias=False)
        self.choice = nn.Linear(d_model, 1, bias=False)
        self.pool = nn.Linear(d_model, num_labels, bias=False)
        active = {"token": {"token"}, "span": {"start", "end"},
                  "choice": {"choice"}, "pool": {"pool"}}[mode]
        for name in ("token", "start", "end", "choice", "pool"):
            getattr(self, name).requires_grad_(name in active)

    def forward(self, hidden):
        hidden = torch.as_tensor(hidden)
        if self.mode == "token":
            if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
                raise ValueError("token mode expects (batch, length, d_model)")
            return self.token(hidden)
        if self.mode == "span":
            if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
                raise ValueError("span mode expects (batch, length, d_model)")
            return self.start(hidden).squeeze(-1), self.end(hidden).squeeze(-1)
        if self.mode == "choice":
            if hidden.ndim != 4 or hidden.shape[-1] != self.d_model:
                raise ValueError("choice mode expects (batch, choices, length, d_model)")
            pooled = hidden.mean(dim=2)
            return self.choice(pooled).squeeze(-1)
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("pool mode expects (batch, length, d_model)")
        return self.pool(torch.tanh(hidden[:, 0]))
