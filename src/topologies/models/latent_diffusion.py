import math

import torch
from torch import nn
from torch.nn import functional as F


class LatentDiffusion(nn.Module):
    architecture_id = "latent-diffusion"
    PRESETS = {
        "tiny": {
            "latent_channels": 3,
            "model_channels": 32,
            "channel_mults": (1, 2),
            "num_res_blocks": 1,
            "attention_downsample_factors": (2,),
            "timesteps": 1000,
            "beta_start": 0.0015,
            "beta_end": 0.0195,
            "norm_groups": 32,
            "dropout": 0.0,
            "context_dim": None,
        },
        "small": {
            "latent_channels": 3,
            "model_channels": 64,
            "channel_mults": (1, 2, 3),
            "num_res_blocks": 2,
            "attention_downsample_factors": (2, 4),
            "timesteps": 1000,
            "beta_start": 0.0015,
            "beta_end": 0.0195,
            "norm_groups": 32,
            "dropout": 0.0,
            "context_dim": None,
        },
    }

    def __init__(
        self,
        latent_channels=3,
        model_channels=32,
        channel_mults=(1, 2),
        num_res_blocks=1,
        attention_downsample_factors=(2, 4, 8),
        timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0195,
        norm_groups=32,
        dropout=0.0,
        context_dim=None,
    ):
        super().__init__()
        for name, value in (
            ("latent_channels", latent_channels),
            ("model_channels", model_channels),
            ("num_res_blocks", num_res_blocks),
            ("timesteps", timesteps),
            ("norm_groups", norm_groups),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(channel_mults, (tuple, list)) or not channel_mults:
            raise ValueError("channel_mults must be a non-empty sequence of positive integers")
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in channel_mults):
            raise ValueError("channel_mults must contain only positive integers")
        if type(attention_downsample_factors) not in (tuple, list):
            raise ValueError("attention_downsample_factors must be a sequence of positive integers")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in attention_downsample_factors
        ):
            raise ValueError("attention_downsample_factors must contain only positive integers")
        if type(beta_start) not in (int, float):
            raise ValueError("beta_start must be a finite number in (0, 1)")
        if type(beta_end) not in (int, float):
            raise ValueError("beta_end must be a finite number in (0, 1)")
        if not math.isfinite(beta_start) or not math.isfinite(beta_end):
            raise ValueError("beta schedule endpoints must be finite")
        if not 0 < beta_start <= beta_end < 1:
            raise ValueError("beta schedule must satisfy 0 < beta_start <= beta_end < 1")
        if type(dropout) not in (int, float):
            raise ValueError("dropout must be a finite number in [0, 1)")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a finite number in [0, 1)")
        if context_dim is not None and (
            not isinstance(context_dim, int) or isinstance(context_dim, bool) or context_dim <= 0
        ):
            raise ValueError("context_dim must be None or a positive integer")

        self.latent_channels = latent_channels
        self.model_channels = model_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = num_res_blocks
        self.attention_downsample_factors = tuple(attention_downsample_factors)
        self.timesteps = timesteps
        self.norm_groups = norm_groups
        self.dropout = float(dropout)
        self.context_dim = context_dim
        self.time_dim = model_channels * 4
        channels = tuple(model_channels * value for value in self.channel_mults)
        if any(channel % norm_groups for channel in channels):
            raise ValueError("norm_groups must divide every UNet feature width")

        betas = torch.linspace(float(beta_start), float(beta_end), timesteps)
        self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, dim=0))
        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.input_conv = nn.Conv2d(latent_channels, channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        for index, channel in enumerate(channels):
            stage_blocks = nn.ModuleList()
            stage_attentions = nn.ModuleList()
            factor = 2**index
            for _ in range(num_res_blocks):
                stage_blocks.append(self._make_residual(channel, channel))
                stage_attentions.append(self._make_attention(channel) if factor in self.attention_downsample_factors else nn.Identity())
            self.down_blocks.append(stage_blocks)
            self.down_attentions.append(stage_attentions)

        self.downsamples = nn.ModuleList(
            nn.Conv2d(channels[index], channels[index + 1], kernel_size=3, stride=2, padding=1)
            for index in range(len(channels) - 1)
        )
        self.middle = self._make_residual(channels[-1], channels[-1])
        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        for index, channel in enumerate(channels):
            stage_blocks = nn.ModuleList()
            stage_attentions = nn.ModuleList()
            factor = 2**index
            for _ in range(num_res_blocks):
                stage_blocks.append(self._make_residual(channel * 2, channel))
                stage_attentions.append(self._make_attention(channel) if factor in self.attention_downsample_factors else nn.Identity())
            self.up_blocks.append(stage_blocks)
            self.up_attentions.append(stage_attentions)
        self.upsamples = nn.ModuleList(
            nn.Conv2d(channels[index + 1], channels[index], kernel_size=3, padding=1)
            for index in range(len(channels) - 1)
        )
        self.output_norm = nn.GroupNorm(norm_groups, channels[0])
        self.output_conv = nn.Conv2d(channels[0], latent_channels, kernel_size=3, padding=1)

    def _make_residual(self, in_channels, out_channels):
        return nn.ModuleDict(
            {
                "norm1": nn.GroupNorm(self.norm_groups, in_channels),
                "conv1": nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                "time": nn.Linear(self.time_dim, out_channels),
                "norm2": nn.GroupNorm(self.norm_groups, out_channels),
                "dropout": nn.Dropout(self.dropout),
                "conv2": nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                "skip": nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1),
            }
        )

    def _make_attention(self, channels):
        if self.context_dim is None:
            return nn.Identity()
        return nn.ModuleDict(
            {
                "norm": nn.LayerNorm(channels),
                "query": nn.Linear(channels, channels, bias=False),
                "key": nn.Linear(self.context_dim, channels, bias=False),
                "value": nn.Linear(self.context_dim, channels, bias=False),
                "output": nn.Linear(channels, channels),
            }
        )

    def _run_residual(self, x, time_embedding, block):
        residual = block["skip"](x)
        hidden = block["conv1"](F.silu(block["norm1"](x)))
        hidden = hidden + block["time"](F.silu(time_embedding))[:, :, None, None]
        hidden = block["conv2"](block["dropout"](F.silu(block["norm2"](hidden))))
        return hidden + residual

    @staticmethod
    def scaled_dot_product_attention(query, key, value):
        if not all(isinstance(tensor, torch.Tensor) for tensor in (query, key, value)):
            raise ValueError("query, key, and value must be tensors")
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("query, key, and value must have shape [batch, tokens, features]")
        if query.shape[0] != key.shape[0] or key.shape[:2] != value.shape[:2]:
            raise ValueError("query, key, and value batch and token dimensions must match")
        if query.shape[-1] != key.shape[-1] or key.shape[-1] == 0:
            raise ValueError("query and key feature dimensions must match and be positive")
        weights = torch.softmax(torch.bmm(query, key.transpose(1, 2)) / math.sqrt(key.shape[-1]), dim=-1)
        return torch.bmm(weights, value)

    def _run_attention(self, x, context, attention):
        if isinstance(attention, nn.Identity):
            return x
        batch, channels, height, width = x.shape
        features = x.flatten(2).transpose(1, 2)
        normalized = attention["norm"](features)
        query = attention["query"](normalized)
        key = attention["key"](context)
        value = attention["value"](context)
        attended = self.scaled_dot_product_attention(query, key, value)
        attended = attention["output"](attended).transpose(1, 2).reshape(batch, channels, height, width)
        return x + attended

    def _time_embedding(self, timesteps):
        half = self.model_channels // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half, 1)
        )
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        if embedding.shape[-1] < self.model_channels:
            embedding = F.pad(embedding, (0, self.model_channels - embedding.shape[-1]))
        return self.time_mlp(embedding)

    def _validate_timesteps(self, timesteps, batch_size):
        if not isinstance(timesteps, torch.Tensor) or timesteps.ndim != 1:
            raise ValueError("timesteps must be a one-dimensional integer tensor")
        if timesteps.shape[0] != batch_size:
            raise ValueError("timesteps batch size must match the latent batch size")
        if timesteps.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("timesteps must use an integer dtype")
        if torch.any(timesteps < 0) or torch.any(timesteps >= self.timesteps):
            raise ValueError(f"timesteps must be in [0, {self.timesteps - 1}]")
        return timesteps.long()

    def q_sample(self, z0, timesteps, noise):
        if not isinstance(z0, torch.Tensor) or z0.ndim != 4:
            raise ValueError("z0 must have shape [batch, channels, height, width]")
        if z0.shape[1] != self.latent_channels:
            raise ValueError(f"z0 must have {self.latent_channels} latent channels")
        if not z0.is_floating_point():
            raise ValueError("z0 must be a floating point tensor")
        indices = self._validate_timesteps(timesteps, z0.shape[0]).to(z0.device)
        if not isinstance(noise, torch.Tensor) or noise.shape != z0.shape:
            raise ValueError("noise must be a tensor with exactly the same shape as z0")
        alpha_bar = self.alpha_bars.to(device=z0.device, dtype=z0.dtype)[indices].view(-1, 1, 1, 1)
        noise = noise.to(device=z0.device, dtype=z0.dtype)
        return alpha_bar.sqrt() * z0 + (1.0 - alpha_bar).sqrt() * noise

    @staticmethod
    def epsilon_loss(predicted_noise, noise):
        if not torch.is_tensor(predicted_noise) or not torch.is_tensor(noise):
            raise ValueError("predicted_noise and noise must be tensors")
        if predicted_noise.shape != noise.shape:
            raise ValueError("predicted_noise and noise must have the same shape")
        return F.mse_loss(predicted_noise, noise)

    def training_loss(self, z0, timesteps, noise, context=None):
        noisy_latent = self.q_sample(z0, timesteps, noise)
        return self.epsilon_loss(self(noisy_latent, timesteps, context), noise)

    def forward(self, z_t, t, context=None):
        if not isinstance(z_t, torch.Tensor) or z_t.ndim != 4:
            raise ValueError("z_t must have shape [batch, channels, height, width]")
        if z_t.shape[1] != self.latent_channels:
            raise ValueError(f"z_t must have {self.latent_channels} latent channels")
        if z_t.shape[0] == 0 or z_t.shape[2] == 0 or z_t.shape[3] == 0:
            raise ValueError("z_t batch, height, and width dimensions must be positive")
        if not z_t.is_floating_point():
            raise ValueError("z_t must be a floating point tensor")
        timesteps = self._validate_timesteps(t, z_t.shape[0]).to(z_t.device)
        if self.context_dim is None:
            if context is not None:
                raise ValueError("context must be omitted when context_dim is None")
        else:
            if not isinstance(context, torch.Tensor) or context.ndim != 3:
                raise ValueError("context must have shape [batch, tokens, context_dim]")
            if context.shape[0] != z_t.shape[0]:
                raise ValueError("context batch size must match the latent batch size")
            if context.shape[1] == 0 or context.shape[2] != self.context_dim:
                raise ValueError(f"context must contain tokens with final dimension {self.context_dim}")
            context = context.to(device=z_t.device, dtype=z_t.dtype)

        time_embedding = self._time_embedding(timesteps)
        hidden = self.input_conv(z_t)
        skips = []
        for index, blocks in enumerate(self.down_blocks):
            stage_skips = []
            for block, attention in zip(blocks, self.down_attentions[index]):
                hidden = self._run_residual(hidden, time_embedding, block)
                hidden = self._run_attention(hidden, context, attention)
                stage_skips.append(hidden)
            skips.append(stage_skips)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)

        hidden = self._run_residual(hidden, time_embedding, self.middle)
        for index in range(len(self.up_blocks) - 1, -1, -1):
            if index < len(self.upsamples):
                size = skips[index][-1].shape[-2:]
                hidden = F.interpolate(hidden, size=size, mode="nearest")
                hidden = self.upsamples[index](hidden)
            for block, attention in zip(self.up_blocks[index], self.up_attentions[index]):
                skip = skips[index].pop()
                hidden = self._run_residual(torch.cat((hidden, skip), dim=1), time_embedding, block)
                hidden = self._run_attention(hidden, context, attention)

        return self.output_conv(F.silu(self.output_norm(hidden)))
