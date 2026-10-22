import math

import torch
from torch import nn
from torch.nn import functional as F


class Chronos(nn.Module):
    architecture_id = "chronos"
    pad_id = 0
    eos_id = 1
    value_token_offset = 2
    PRESETS = {
        "tiny": {
            "vocab_size": 4096,
            "context_length": 512,
            "forecast_length": 64,
            "d_model": 128,
            "num_layers": 2,
            "num_heads": 4,
            "d_ff": 512,
            "dropout": 0.0,
        },
        "small": {
            "vocab_size": 4096,
            "context_length": 512,
            "forecast_length": 64,
            "d_model": 512,
            "num_layers": 6,
            "num_heads": 8,
            "d_ff": 2048,
            "dropout": 0.0,
        },
    }

    def __init__(
        self,
        vocab_size=4096,
        bin_min=-15.0,
        bin_max=15.0,
        context_length=512,
        forecast_length=64,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        dropout=0.0,
        scale_eps=1e-8,
    ):
        super().__init__()
        dimensions = (vocab_size, context_length, forecast_length, d_model, num_layers,
                      num_heads, d_ff)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in dimensions):
            raise ValueError("vocabulary, sequence lengths, and model dimensions must be positive integers")
        if vocab_size < 4:
            raise ValueError("vocab_size must be at least four for two special tokens and two value bins")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if (isinstance(bin_min, bool) or isinstance(bin_max, bool)
                or not isinstance(bin_min, (int, float)) or not isinstance(bin_max, (int, float))
                or not math.isfinite(bin_min) or not math.isfinite(bin_max) or bin_min >= bin_max):
            raise ValueError("bin_min and bin_max must be finite numbers with bin_min < bin_max")
        if (isinstance(dropout, bool) or not isinstance(dropout, (int, float))
                or not math.isfinite(dropout) or not 0.0 <= dropout < 1.0):
            raise ValueError("dropout must be finite and in [0, 1)")
        if (isinstance(scale_eps, bool) or not isinstance(scale_eps, (int, float))
                or not math.isfinite(scale_eps) or scale_eps <= 0.0):
            raise ValueError("scale_eps must be a finite positive number")
        self.vocab_size = vocab_size
        self.bin_min = float(bin_min)
        self.bin_max = float(bin_max)
        self.history_length = context_length
        self.forecast_length = forecast_length
        self.max_sequence_length = context_length + forecast_length + 1
        self.context_length = self.max_sequence_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.scale_eps = float(scale_eps)
        centers = torch.linspace(self.bin_min, self.bin_max, vocab_size - self.value_token_offset)
        self.register_buffer("bin_centers", centers)
        self.register_buffer("bin_edges", (centers[:-1] + centers[1:]) / 2)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.max_sequence_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, tokens):
        if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 2
                or tokens.dtype != torch.long or tokens.shape[0] < 1
                or not 0 < tokens.shape[1] <= self.max_sequence_length):
            raise ValueError("tokens must be a non-empty LongTensor [batch, time] within max_sequence_length")
        if bool((tokens < 0).any()) or bool((tokens >= self.vocab_size).any()):
            raise ValueError("tokens must contain IDs within the configured vocabulary")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None]
        causal_mask = torch.triu(
            torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask)
        return self.lm_head(self.final_norm(hidden))

    def quantize(self, values):
        if not isinstance(values, torch.Tensor) or not values.is_floating_point():
            raise ValueError("values must be a floating point tensor")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("values must be finite")
        edges = self.bin_edges.to(device=values.device, dtype=values.dtype)
        bins = torch.bucketize(values.contiguous(), edges, right=True)
        return bins + self.value_token_offset

    def dequantize(self, tokens, scale=None):
        if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long:
            raise ValueError("tokens must be a LongTensor")
        if bool((tokens < self.value_token_offset).any()) or bool((tokens >= self.vocab_size).any()):
            raise ValueError("dequantize accepts only value-bin token IDs")
        centers = self.bin_centers.to(device=tokens.device)
        values = centers[tokens - self.value_token_offset]
        if scale is not None:
            scale = torch.as_tensor(scale, device=tokens.device, dtype=values.dtype)
            if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
                raise ValueError("scale must contain finite positive values")
            try:
                values = values * scale
            except RuntimeError as exc:
                raise ValueError("scale must broadcast to the token shape") from exc
        return values

    def tokenize(self, context, future=None):
        if (not isinstance(context, torch.Tensor) or context.ndim != 2
                or not context.is_floating_point() or context.shape[0] < 1
                or not 0 < context.shape[1] <= self.history_length):
            raise ValueError("context must be a floating point [batch, time] tensor within context_length")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("context must contain finite values")
        scale = context.abs().mean(dim=1, keepdim=True).clamp_min(self.scale_eps)
        values = context / scale
        append_eos = future is not None
        if future is not None:
            if (not isinstance(future, torch.Tensor) or future.ndim != 2
                    or not future.is_floating_point() or future.shape[0] != context.shape[0]
                    or not 0 < future.shape[1] <= self.forecast_length
                    or future.device != context.device):
                raise ValueError("future must be a floating point [batch, horizon] tensor matching context")
            if not bool(torch.isfinite(future).all()):
                raise ValueError("future must contain finite values")
            values = torch.cat((values, future.to(dtype=context.dtype) / scale), dim=1)
        tokens = self.quantize(values)
        if append_eos:
            eos = torch.full((context.shape[0], 1), self.eos_id, dtype=torch.long, device=context.device)
            tokens = torch.cat((tokens, eos), dim=1)
        return tokens, scale

    def compute_loss(self, logits, targets):
        if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point()
                or logits.ndim < 2 or logits.shape[0] < 1 or logits.shape[-1] != self.vocab_size):
            raise ValueError("logits must be a floating point tensor ending in vocab_size")
        if (not isinstance(targets, torch.Tensor) or targets.dtype != torch.long
                or targets.shape != logits.shape[:-1] or targets.device != logits.device):
            raise ValueError("targets must be a LongTensor matching all non-vocabulary logit dimensions")
        active = targets.ne(self.pad_id) & targets.ne(-100)
        if bool(((targets < 0) & targets.ne(-100)).any()) or bool((targets >= self.vocab_size).any()):
            raise ValueError("targets must be valid vocabulary IDs, pad_id, or -100")
        safe_targets = targets.masked_fill(~active, -100)
        losses = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), safe_targets.reshape(-1),
            ignore_index=-100, reduction="none",
        ).reshape(targets.shape)
        per_series = (losses * active.to(losses.dtype)).reshape(targets.shape[0], -1).sum(dim=1)
        return per_series.mean()

    def forecast(self, context, horizon=None, num_samples=1, temperature=1.0):
        if horizon is None:
            horizon = self.forecast_length
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 0 < horizon <= self.forecast_length:
            raise ValueError("horizon must be a positive integer no greater than forecast_length")
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("temperature must be a finite positive number")
        context_tokens, scale = self.tokenize(context)
        batch = context.shape[0]
        tokens = context_tokens.unsqueeze(0).expand(num_samples, -1, -1).reshape(
            num_samples * batch, context.shape[1]
        )
        sample_scale = scale.unsqueeze(0).expand(num_samples, -1, -1).reshape(num_samples * batch, 1)
        generated = []
        with torch.no_grad():
            for _ in range(horizon):
                next_logits = self(tokens)[:, -1, self.value_token_offset:] / float(temperature)
                next_token = torch.multinomial(F.softmax(next_logits, dim=-1), 1) + self.value_token_offset
                generated.append(next_token[:, 0])
                tokens = torch.cat((tokens, next_token), dim=1)
            forecast_tokens = torch.stack(generated, dim=1)
            values = self.dequantize(forecast_tokens, scale=sample_scale)
        return values.reshape(num_samples, batch, horizon)
