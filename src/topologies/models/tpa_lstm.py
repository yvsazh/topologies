import torch
from torch import nn


class TPALSTM(nn.Module):
    architecture_id = "tpa-lstm"
    PRESETS = {
        "tiny": {"input_size": 1, "hidden_size": 6, "num_filters": 32, "max_window_size": 30},
        "small": {"input_size": 6, "hidden_size": 25, "num_filters": 32, "max_window_size": 48},
    }

    def __init__(self, input_size, hidden_size, max_window_size, num_filters=32):
        super().__init__()
        dimensions = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "max_window_size": max_window_size,
            "num_filters": num_filters,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_window_size = max_window_size
        self.num_filters = num_filters
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bias=False)
        self.temporal_filters = nn.Parameter(torch.empty(num_filters, max_window_size))
        nn.init.xavier_uniform_(self.temporal_filters)
        self.attention_matrix = nn.Parameter(torch.empty(num_filters, hidden_size))
        nn.init.xavier_uniform_(self.attention_matrix)
        self.hidden_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.context_projection = nn.Linear(num_filters, hidden_size, bias=False)
        self.forecast_projection = nn.Linear(hidden_size, input_size, bias=False)

    def _apply_temporal_filters(self, history):
        window_size = history.shape[-1]
        filters = self.temporal_filters[:, -window_size:]
        return torch.einsum("bmt,kt->bmk", history, filters)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("x must be a tensor with shape [batch, sequence, features]")
        if x.shape[-1] != self.input_size:
            raise ValueError(f"x must have {self.input_size} input features")
        if x.shape[1] < 1:
            raise ValueError("x sequence length must be at least 1")
        if x.shape[1] > self.max_window_size:
            raise ValueError(f"x sequence length must not exceed {self.max_window_size}")
        hidden_sequence, (last_hidden, _) = self.lstm(x)
        history = hidden_sequence.transpose(1, 2)
        temporal_features = self._apply_temporal_filters(history)
        query = last_hidden[-1]
        projected_query = query @ self.attention_matrix.transpose(0, 1)
        attention = torch.sigmoid(torch.einsum("bmk,bk->bm", temporal_features, projected_query))
        context = torch.einsum("bm,bmk->bk", attention, temporal_features)
        fused = self.hidden_projection(query) + self.context_projection(context)
        return self.forecast_projection(fused)
