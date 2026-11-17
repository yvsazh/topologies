import math

import torch
from torch import nn
from torch.nn import functional as F


class TinyHAR(nn.Module):
    architecture_id = "tinyhar"
    PRESETS = {
        "tiny": {"filter_num": 4},
        "small": {"filter_num": 10},
        "base": {"filter_num": 20},
    }

    def __init__(
        self,
        channels: int,
        num_classes: int,
        filter_num: int = 20,
        num_heads: int = 4,
        head_dim: int = 16,
        feedforward_dim: int = 16,
        transformer_dropout: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        dimensions = {
            "channels": channels,
            "num_classes": num_classes,
            "filter_num": filter_num,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "feedforward_dim": feedforward_dim,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("transformer_dropout", transformer_dropout),
            ("dropout", dropout),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0.0 <= value < 1.0
            ):
                raise ValueError(f"{name} must be in the range [0, 1)")
        self.channels = channels
        self.num_classes = num_classes
        self.filter_num = filter_num
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.transformer_dropout_probability = float(transformer_dropout)
        self.temporal_convs = nn.ModuleList(
            nn.Conv2d(
                1 if index == 0 else filter_num,
                filter_num,
                kernel_size=(5, 1),
                stride=(2, 1),
            )
            for index in range(4)
        )
        self.temporal_norms = nn.ModuleList(nn.BatchNorm2d(filter_num) for _ in range(4))
        attention_width = num_heads * head_dim
        self.transformer_norm1 = nn.LayerNorm(filter_num)
        self.transformer_qkv = nn.Linear(filter_num, 3 * attention_width)
        self.transformer_output = nn.Linear(attention_width, filter_num)
        self.transformer_norm2 = nn.LayerNorm(filter_num)
        self.transformer_feedforward1 = nn.Linear(filter_num, feedforward_dim)
        self.transformer_feedforward2 = nn.Linear(feedforward_dim, filter_num)
        self.transformer_dropout = nn.Dropout(float(transformer_dropout))
        self.interaction_dropout = nn.Dropout(float(dropout))
        self.channel_fusion = nn.Linear(channels * filter_num, 2 * filter_num)
        self.lstm = nn.LSTM(2 * filter_num, 2 * filter_num, num_layers=1, batch_first=True)
        self.temporal_attention_projection = nn.Linear(2 * filter_num, 2 * filter_num)
        self.temporal_attention_score = nn.Linear(2 * filter_num, 1)
        self.attention_gamma = nn.Parameter(torch.zeros(()))
        self.classifier = nn.Linear(2 * filter_num, num_classes)

    def _transform_channels(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, timesteps, channels, features = tokens.shape
        sequence = tokens.reshape(batch * timesteps, channels, features)
        normalized = self.transformer_norm1(sequence)
        qkv = self.transformer_qkv(normalized).reshape(
            batch * timesteps, channels, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        weights = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1
        )
        weights = F.dropout(weights, p=self.transformer_dropout_probability, training=self.training)
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(
            batch * timesteps, channels, self.num_heads * self.head_dim
        )
        sequence = sequence + self.transformer_dropout(self.transformer_output(attended))
        feedforward = self.transformer_feedforward1(self.transformer_norm2(sequence))
        feedforward = F.gelu(feedforward)
        feedforward = self.transformer_feedforward2(feedforward)
        sequence = sequence + self.transformer_dropout(feedforward)
        return sequence.reshape(batch, timesteps, channels, features)

    def _temporal_attention(self, hidden: torch.Tensor) -> torch.Tensor:
        scores = self.temporal_attention_score(
            torch.tanh(self.temporal_attention_projection(hidden))
        ).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        return hidden[:, -1, :] + self.attention_gamma * context

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have rank 3 with shape (batch, channels, timesteps)")
        if inputs.shape[0] < 1:
            raise ValueError("batch size must be positive")
        if inputs.shape[1] != self.channels:
            raise ValueError("input channel count must match configured channels")
        if inputs.shape[2] < 61:
            raise ValueError("input must contain at least 61 timesteps")
        features = inputs.unsqueeze(1).permute(0, 1, 3, 2)
        for convolution, normalization in zip(self.temporal_convs, self.temporal_norms):
            features = convolution(features)
            if self.training and features.numel() // features.shape[1] == 1:
                features = F.batch_norm(
                    features,
                    normalization.running_mean,
                    normalization.running_var,
                    normalization.weight,
                    normalization.bias,
                    False,
                    normalization.momentum,
                    normalization.eps,
                )
            else:
                features = normalization(features)
            features = F.relu(features)
        tokens = features.permute(0, 2, 3, 1)
        tokens = self._transform_channels(tokens)
        tokens = self.interaction_dropout(tokens)
        batch, timesteps, channels, features = tokens.shape
        fused = F.relu(self.channel_fusion(tokens.reshape(batch, timesteps, channels * features)))
        hidden, _ = self.lstm(fused)
        representation = self._temporal_attention(hidden)
        return self.classifier(representation)
