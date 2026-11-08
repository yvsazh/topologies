import math

import torch
from torch import nn


class MQRNN(nn.Module):
    architecture_id = "mq-rnn"
    PRESETS = {
        "tiny": {
            "input_size": 1,
            "future_size": 1,
            "hidden_size": 8,
            "context_size": 8,
            "decoder_size": 16,
            "horizon": 2,
            "quantiles": (0.1, 0.5, 0.9),
        },
        "small": {
            "input_size": 4,
            "future_size": 4,
            "hidden_size": 64,
            "context_size": 32,
            "decoder_size": 128,
            "horizon": 12,
            "quantiles": (0.1, 0.5, 0.9),
        },
        "base": {
            "input_size": 8,
            "future_size": 8,
            "hidden_size": 128,
            "context_size": 64,
            "decoder_size": 256,
            "horizon": 24,
            "quantiles": (0.1, 0.5, 0.9),
        },
    }

    def __init__(
        self,
        input_size=1,
        future_size=1,
        hidden_size=32,
        context_size=16,
        decoder_size=64,
        horizon=4,
        quantiles=(0.1, 0.5, 0.9),
    ):
        super().__init__()
        dimensions = (input_size, future_size, hidden_size, context_size, decoder_size, horizon)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
               for value in dimensions):
            raise ValueError("input_size, future_size, hidden_size, context_size, decoder_size, and horizon must be positive integers")
        try:
            quantiles = tuple(float(value) for value in quantiles)
        except (TypeError, ValueError):
            raise ValueError("quantiles must be a nonempty increasing sequence in (0, 1)") from None
        if (not quantiles or any(not math.isfinite(value) or value <= 0 or value >= 1
                                 for value in quantiles)
                or any(quantiles[index] >= quantiles[index + 1]
                       for index in range(len(quantiles) - 1))):
            raise ValueError("quantiles must be a nonempty strictly increasing sequence in (0, 1)")
        self.input_size = input_size
        self.future_size = future_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.decoder_size = decoder_size
        self.horizon = horizon
        self.quantiles = quantiles
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_size + horizon * future_size, decoder_size),
            nn.ReLU(),
            nn.Linear(decoder_size, (horizon + 1) * context_size),
        )
        self.local_mlp = nn.Sequential(
            nn.Linear(2 * context_size + future_size, decoder_size),
            nn.ReLU(),
            nn.Linear(decoder_size, len(quantiles)),
        )

    def forward(self, history, future_features):
        if (not isinstance(history, torch.Tensor) or history.ndim != 3
                or not history.is_floating_point() or history.shape[0] < 1
                or history.shape[1] < 1 or history.shape[2] != self.input_size):
            raise ValueError("history must be a nonempty floating tensor shaped [B,T,input_size]")
        if not isinstance(future_features, torch.Tensor) or not future_features.is_floating_point():
            raise ValueError("future_features must be a floating tensor")
        if (history.device != future_features.device or history.dtype != future_features.dtype):
            raise ValueError("history and future_features must have the same device and dtype")
        if future_features.ndim == 3:
            if (future_features.shape[0] != history.shape[0]
                    or future_features.shape[1] != self.horizon
                    or future_features.shape[2] != self.future_size):
                raise ValueError("rank-3 future_features must have shape [B,horizon,future_size]")
            hidden = self.encoder(history)[0][:, -1]
            contexts = self.global_mlp(
                torch.cat((hidden, future_features.reshape(history.shape[0], -1)), dim=-1)
            ).reshape(history.shape[0], self.horizon + 1, self.context_size)
            horizon_context = contexts[:, :self.horizon]
            shared_context = contexts[:, self.horizon:self.horizon + 1].expand(
                -1, self.horizon, -1
            )
        elif future_features.ndim == 4:
            if (future_features.shape[0] != history.shape[0]
                    or future_features.shape[1] != history.shape[1]
                    or future_features.shape[2] != self.horizon
                    or future_features.shape[3] != self.future_size):
                raise ValueError("rank-4 future_features must have shape [B,T,horizon,future_size]")
            batch, time = history.shape[:2]
            hidden = self.encoder(history)[0]
            contexts = self.global_mlp(
                torch.cat((hidden, future_features.reshape(batch, time, -1)), dim=-1)
            ).reshape(batch, time, self.horizon + 1, self.context_size)
            horizon_context = contexts[:, :, :self.horizon]
            shared_context = contexts[:, :, self.horizon:self.horizon + 1].expand(
                -1, -1, self.horizon, -1
            )
        else:
            raise ValueError("future_features must have rank 3 or 4")
        local_input = torch.cat((horizon_context, shared_context, future_features), dim=-1)
        return self.local_mlp(local_input)

    def pinball_loss(self, target, forecast, mask=None, reduction="sum"):
        if not isinstance(forecast, torch.Tensor) or not forecast.is_floating_point():
            raise ValueError("forecast must be a floating tensor")
        if (not isinstance(target, torch.Tensor) or not target.is_floating_point()
                or target.shape != forecast.shape[:-1]
                or forecast.shape[-1] != len(self.quantiles)):
            raise ValueError("target must be floating with shape forecast.shape[:-1]")
        if reduction not in ("none", "sum", "mean"):
            raise ValueError("reduction must be 'none', 'sum', or 'mean'")
        valid = None
        if mask is not None:
            if (not isinstance(mask, torch.Tensor) or mask.shape != target.shape
                    or (mask.dtype != torch.bool and not mask.is_floating_point())):
                raise ValueError("mask must be boolean or floating with the target shape")
            if mask.is_floating_point() and (not torch.isfinite(mask).all()
                                               or ((mask != 0) & (mask != 1)).any()):
                raise ValueError("floating mask values must be finite zeros or ones")
            valid = mask.to(device=forecast.device, dtype=torch.bool)
            target = target.to(device=forecast.device, dtype=forecast.dtype)
            target = torch.where(valid, target, torch.zeros_like(target))
        quantiles = forecast.new_tensor(self.quantiles)
        error = target.to(device=forecast.device, dtype=forecast.dtype).unsqueeze(-1) - forecast
        terms = torch.maximum(quantiles * error, (quantiles - 1) * error)
        if valid is not None:
            terms = torch.where(valid.unsqueeze(-1), terms, torch.zeros_like(terms))
        if reduction == "none":
            return terms
        if reduction == "sum":
            return terms.sum()
        denominator = terms.numel() if valid is None else valid.sum() * len(self.quantiles)
        if isinstance(denominator, torch.Tensor):
            return terms.sum() / denominator.clamp_min(1)
        return terms.mean()
