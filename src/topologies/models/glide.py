import math

import torch
from torch import nn
from torch.nn import functional as F


class GLIDE(nn.Module):
    architecture_id = "glide"
    PRESETS = {
        "tiny": {
            "vocab_size": 512,
            "image_size": 16,
            "model_channels": 32,
            "num_res_blocks": 1,
            "channel_mult": (1, 2),
            "attention_resolutions": (1, 2),
            "num_heads": 1,
            "num_head_channels": 32,
            "dropout": 0.0,
            "text_ctx": 1,
            "xf_width": 32,
            "xf_layers": 1,
            "xf_heads": 2,
            "xf_final_ln": True,
            "xf_padding": True,
            "use_scale_shift_norm": True,
            "resblock_updown": True,
        },
        "small": {
            "vocab_size": 4096,
            "image_size": 32,
            "model_channels": 64,
            "num_res_blocks": 1,
            "channel_mult": (1, 2, 3),
            "attention_resolutions": (1, 2, 4),
            "num_heads": 1,
            "num_head_channels": 32,
            "dropout": 0.1,
            "text_ctx": 32,
            "xf_width": 128,
            "xf_layers": 4,
            "xf_heads": 4,
            "xf_final_ln": True,
            "xf_padding": True,
            "use_scale_shift_norm": True,
            "resblock_updown": True,
        },
    }

    class _ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, time_dim, dropout, scale_shift, up=False,
                     down=False):
            super().__init__()
            if up and down:
                raise ValueError("a residual block cannot upsample and downsample together")
            self.up = up
            self.down = down
            self.norm1 = nn.GroupNorm(GLIDE._groups(in_channels), in_channels)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.norm2 = nn.GroupNorm(GLIDE._groups(out_channels), out_channels)
            self.time_projection = nn.Linear(time_dim, out_channels * (2 if scale_shift else 1))
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.dropout = dropout
            self.scale_shift = scale_shift
            self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(
                in_channels, out_channels, 1
            )

        def forward(self, x, time):
            residual = x
            hidden = F.silu(self.norm1(x))
            if self.up:
                hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
                residual = F.interpolate(residual, scale_factor=2, mode="nearest")
            if self.down:
                hidden = F.avg_pool2d(hidden, 2)
                residual = F.avg_pool2d(residual, 2)
            hidden = self.conv1(hidden)
            time = self.time_projection(F.silu(time))[:, :, None, None]
            hidden = self.norm2(hidden)
            if self.scale_shift:
                scale, shift = time.chunk(2, dim=1)
                hidden = hidden * (1 + scale) + shift
            else:
                hidden = hidden + time
            hidden = F.silu(hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            return self.skip(residual) + self.conv2(hidden)

    class _AttentionBlock(nn.Module):
        def __init__(self, channels, text_width, heads):
            super().__init__()
            if channels % heads:
                raise ValueError("attention channels must be divisible by the number of heads")
            self.channels = channels
            self.heads = heads
            self.head_dim = channels // heads
            self.norm = nn.GroupNorm(GLIDE._groups(channels), channels)
            self.image_qkv = nn.Conv1d(channels, channels * 3, 1)
            self.text_kv = nn.Linear(text_width, channels * 2)
            self.output_projection = nn.Conv1d(channels, channels, 1)

        def forward(self, x, text, text_mask):
            batch, channels, height, width = x.shape
            spatial = height * width
            image = self.norm(x).reshape(batch, channels, spatial)
            query, key, value = self.image_qkv(image).chunk(3, dim=1)
            text_key, text_value = self.text_kv(text).chunk(2, dim=-1)
            query = query.reshape(batch, self.heads, self.head_dim, spatial).transpose(-2, -1)
            key = key.reshape(batch, self.heads, self.head_dim, spatial).transpose(-2, -1)
            value = value.reshape(batch, self.heads, self.head_dim, spatial).transpose(-2, -1)
            text_key = text_key.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
            text_value = text_value.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
            key = torch.cat((key, text_key), dim=-2)
            value = torch.cat((value, text_value), dim=-2)
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if text_mask is not None:
                invalid_text = ~text_mask[:, None, None, :]
                scores = torch.cat((
                    scores[..., :spatial],
                    scores[..., spatial:].masked_fill(invalid_text, -torch.inf),
                ), dim=-1)
            weights = scores.softmax(dim=-1)
            hidden = torch.matmul(weights, value).transpose(-2, -1).reshape(batch, channels, spatial)
            hidden = self.output_projection(hidden).reshape(batch, channels, height, width)
            return x + hidden

    class _TransformerBlock(nn.Module):
        def __init__(self, width, heads, dropout):
            super().__init__()
            self.norm1 = nn.LayerNorm(width)
            self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(width)
            self.feedforward = nn.Sequential(
                nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 4, width)
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, padding_mask):
            normalized = self.norm1(x)
            attended = self.attention(
                normalized,
                normalized,
                normalized,
                key_padding_mask=padding_mask,
                need_weights=False,
            )[0]
            x = x + self.dropout(attended)
            return x + self.dropout(self.feedforward(self.norm2(x)))

    class _TextEncoder(nn.Module):
        def __init__(self, vocab_size, text_ctx, width, layers, heads, final_ln, dropout):
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, width)
            self.position_embedding = nn.Parameter(torch.empty(1, text_ctx, width))
            nn.init.normal_(self.position_embedding, std=0.01)
            self.blocks = nn.ModuleList(
                GLIDE._TransformerBlock(width, heads, dropout) for _ in range(layers)
            )
            self.final_norm = nn.LayerNorm(width) if final_ln else nn.Identity()

        def forward(self, tokens, padding_mask):
            positions = self.position_embedding[:, : tokens.shape[1]]
            hidden = self.token_embedding(tokens) + positions
            key_padding_mask = None if padding_mask is None else ~padding_mask
            for block in self.blocks:
                hidden = block(hidden, key_padding_mask)
            return self.final_norm(hidden)

    class _UNet(nn.Module):
        def __init__(self, image_size, model_channels, num_res_blocks, channel_mult,
                     attention_resolutions, num_head_channels, num_heads, text_width, dropout,
                     scale_shift, resblock_updown, out_channels):
            super().__init__()
            time_dim = model_channels * 4
            self.time_dim = time_dim
            self.time_mlp = nn.Sequential(
                nn.Linear(model_channels, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
            )
            self.input_projection = nn.Conv2d(3, model_channels, 3, padding=1)
            self.down_blocks = nn.ModuleList()
            self.down_attentions = nn.ModuleList()
            self.downsamples = nn.ModuleList()
            channels = model_channels
            skip_channels = [channels]
            downsample_factor = 1
            for level, multiplier in enumerate(channel_mult):
                level_channels = model_channels * multiplier
                blocks = nn.ModuleList()
                attentions = nn.ModuleList()
                heads = level_channels // num_head_channels if num_head_channels > 0 else num_heads
                for _ in range(num_res_blocks):
                    blocks.append(GLIDE._ResidualBlock(
                        channels, level_channels, time_dim, dropout, scale_shift
                    ))
                    channels = level_channels
                    attentions.append(
                        GLIDE._AttentionBlock(channels, text_width, heads)
                        if downsample_factor in attention_resolutions else nn.Identity()
                    )
                    skip_channels.append(channels)
                self.down_blocks.append(blocks)
                self.down_attentions.append(attentions)
                if level < len(channel_mult) - 1:
                    next_channels = model_channels * channel_mult[level + 1]
                    if resblock_updown:
                        downsample = GLIDE._ResidualBlock(
                            channels, next_channels, time_dim, dropout, scale_shift, down=True
                        )
                    else:
                        downsample = nn.Sequential(
                            nn.Conv2d(channels, next_channels, 3, stride=2, padding=1)
                        )
                    self.downsamples.append(downsample)
                    channels = next_channels
                    skip_channels.append(channels)
                    downsample_factor *= 2
            self.middle1 = GLIDE._ResidualBlock(
                channels, channels, time_dim, dropout, scale_shift
            )
            middle_heads = channels // num_head_channels if num_head_channels > 0 else num_heads
            self.middle_attention = GLIDE._AttentionBlock(channels, text_width, middle_heads)
            self.middle2 = GLIDE._ResidualBlock(
                channels, channels, time_dim, dropout, scale_shift
            )
            self.up_blocks = nn.ModuleList()
            self.up_attentions = nn.ModuleList()
            self.upsamples = nn.ModuleList()
            downsample_factor = 2 ** (len(channel_mult) - 1)
            for level in reversed(range(len(channel_mult))):
                level_channels = model_channels * channel_mult[level]
                blocks = nn.ModuleList()
                attentions = nn.ModuleList()
                heads = level_channels // num_head_channels if num_head_channels > 0 else num_heads
                for _ in range(num_res_blocks + 1):
                    skip_width = skip_channels.pop()
                    blocks.append(GLIDE._ResidualBlock(
                        channels + skip_width, level_channels, time_dim, dropout, scale_shift
                    ))
                    channels = level_channels
                    attentions.append(
                        GLIDE._AttentionBlock(channels, text_width, heads)
                        if downsample_factor in attention_resolutions else nn.Identity()
                    )
                self.up_blocks.append(blocks)
                self.up_attentions.append(attentions)
                if level > 0:
                    if resblock_updown:
                        upsample = GLIDE._ResidualBlock(
                            channels, model_channels * channel_mult[level - 1], time_dim,
                            dropout, scale_shift, up=True
                        )
                    else:
                        upsample = nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(channels, model_channels * channel_mult[level - 1], 3, padding=1),
                        )
                    self.upsamples.append(upsample)
                    channels = model_channels * channel_mult[level - 1]
                    downsample_factor //= 2
            if skip_channels:
                raise ValueError("channel_mult does not produce a balanced U-Net")
            self.output_norm = nn.GroupNorm(GLIDE._groups(model_channels), model_channels)
            self.output_projection = nn.Conv2d(model_channels, out_channels, 3, padding=1)
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)
            self.image_size = image_size

        def _time_embedding(self, timesteps):
            half = self.time_dim // 8
            frequencies = torch.exp(
                torch.arange(half, device=timesteps.device, dtype=torch.float32)
                * (-math.log(10000.0) / max(half - 1, 1))
            )
            phase = timesteps.float().reshape(-1, 1) * frequencies.reshape(1, -1)
            embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
            embedding = F.pad(embedding, (0, self.time_dim // 4 - embedding.shape[-1]))
            return self.time_mlp(embedding)

        def forward(self, image, timesteps, text, text_mask, global_text):
            time = self._time_embedding(timesteps) + global_text
            hidden = self.input_projection(image)
            skips = [hidden]
            for level, blocks in enumerate(self.down_blocks):
                for block, attention in zip(blocks, self.down_attentions[level]):
                    hidden = attention(block(hidden, time), text, text_mask) if isinstance(
                        attention, GLIDE._AttentionBlock
                    ) else block(hidden, time)
                    skips.append(hidden)
                if level < len(self.downsamples):
                    hidden = self.downsamples[level](hidden, time) if isinstance(
                        self.downsamples[level], GLIDE._ResidualBlock
                    ) else self.downsamples[level](hidden)
                    skips.append(hidden)
            hidden = self.middle1(hidden, time)
            hidden = self.middle_attention(hidden, text, text_mask)
            hidden = self.middle2(hidden, time)
            for level, blocks in enumerate(self.up_blocks):
                for block, attention in zip(blocks, self.up_attentions[level]):
                    hidden = block(torch.cat((hidden, skips.pop()), dim=1), time)
                    if isinstance(attention, GLIDE._AttentionBlock):
                        hidden = attention(hidden, text, text_mask)
                if level < len(self.upsamples):
                    upsample = self.upsamples[level]
                    hidden = upsample(hidden, time) if isinstance(
                        upsample, GLIDE._ResidualBlock
                    ) else upsample(hidden)
            return self.output_projection(F.silu(self.output_norm(hidden)))

    @staticmethod
    def _groups(channels):
        return next(group for group in range(min(32, channels), 0, -1) if channels % group == 0)

    @staticmethod
    def classifier_free_guidance(unconditional, conditional, scale=3.0):
        if unconditional.shape != conditional.shape:
            raise ValueError("conditional and unconditional predictions must have matching shapes")
        if not isinstance(scale, (int, float)) or scale < 1:
            raise ValueError("guidance scale must be a number greater than or equal to 1")
        return unconditional + scale * (conditional - unconditional)

    @staticmethod
    def noise_mse(prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("noise prediction and target must have matching shapes")
        return F.mse_loss(prediction, target)

    @staticmethod
    def hybrid_loss(simple_loss, vlb_loss):
        if simple_loss.shape != vlb_loss.shape:
            raise ValueError("simple and variational losses must have matching shapes")
        return simple_loss + 0.001 * vlb_loss

    @staticmethod
    def learned_variance(values, beta, posterior_variance):
        beta = torch.as_tensor(beta, device=values.device, dtype=values.dtype)
        posterior_variance = torch.as_tensor(
            posterior_variance, device=values.device, dtype=values.dtype
        )
        if torch.any(beta <= 0) or torch.any(posterior_variance <= 0):
            raise ValueError("beta and posterior_variance must be positive")
        fraction = (values + 1) / 2
        return torch.exp(
            fraction * torch.log(beta) + (1 - fraction) * torch.log(posterior_variance)
        )

    def __init__(self, vocab_size, image_size=64, in_channels=3, out_channels=6,
                 model_channels=192, num_res_blocks=3, channel_mult=(1, 2, 3, 4),
                 attention_resolutions=(2, 4, 8), num_heads=1, num_head_channels=64,
                 dropout=0.1, text_ctx=128, xf_width=512, xf_layers=16, xf_heads=8,
                 xf_final_ln=True, xf_padding=True, use_scale_shift_norm=True,
                 resblock_updown=True):
        super().__init__()
        dimensions = {
            "vocab_size": vocab_size,
            "image_size": image_size,
            "model_channels": model_channels,
            "num_res_blocks": num_res_blocks,
            "num_heads": num_heads,
            "text_ctx": text_ctx,
            "xf_width": xf_width,
            "xf_layers": xf_layers,
            "xf_heads": xf_heads,
        }
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
               for value in dimensions.values()):
            raise ValueError("vocabulary, image, width, depth, and head dimensions must be positive integers")
        if in_channels != 3 or out_channels != 6:
            raise ValueError("GLIDE expects three RGB input channels and six output channels")
        if not isinstance(channel_mult, (tuple, list)) or not channel_mult or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in channel_mult
        ):
            raise ValueError("channel_mult must be a nonempty sequence of positive integers")
        if not isinstance(attention_resolutions, (tuple, list)) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in attention_resolutions
        ):
            raise ValueError("attention_resolutions must contain positive downsampling factors")
        max_factor = 2 ** (len(channel_mult) - 1)
        if any(value > max_factor for value in attention_resolutions):
            raise ValueError("attention_resolutions cannot exceed the U-Net downsampling factor")
        if num_head_channels != -1 and (
            not isinstance(num_head_channels, int) or isinstance(num_head_channels, bool)
            or num_head_channels < 1
        ):
            raise ValueError("num_head_channels must be positive or -1")
        if xf_width % xf_heads:
            raise ValueError("xf_width must be divisible by xf_heads")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a number in the range [0, 1)")
        if not all(isinstance(value, bool) for value in (
            xf_final_ln, xf_padding, use_scale_shift_norm, resblock_updown
        )):
            raise ValueError("normalization, padding, and updown options must be booleans")
        if image_size < max_factor * 2 or image_size % max_factor:
            raise ValueError("image_size must be at least twice and divisible by the U-Net downsampling factor")
        if num_head_channels > 0 and any(
            model_channels * multiplier % num_head_channels for multiplier in channel_mult
        ):
            raise ValueError("each U-Net width must be divisible by num_head_channels")
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.text_ctx = text_ctx
        self.xf_padding = xf_padding
        self.text_encoder = self._TextEncoder(
            vocab_size, text_ctx, xf_width, xf_layers, xf_heads, xf_final_ln, dropout
        )
        self.text_projection = nn.Linear(xf_width, model_channels * 4)
        self.unet = self._UNet(
            image_size, model_channels, num_res_blocks, tuple(channel_mult),
            tuple(attention_resolutions), num_head_channels, num_heads, xf_width, dropout,
            use_scale_shift_norm, resblock_updown, out_channels
        )

    @staticmethod
    def split_output(output):
        if output.ndim != 4 or output.shape[1] != 6:
            raise ValueError("GLIDE output must have shape [B, 6, H, W]")
        return output.chunk(2, dim=1)

    @property
    def output_projection(self):
        return self.unet.output_projection

    def forward(self, noisy_image, timestep, tokens, padding_mask=None):
        if not isinstance(noisy_image, torch.Tensor) or noisy_image.ndim != 4 or noisy_image.shape[1] != 3:
            raise ValueError("noisy_image must have shape [B, 3, H, W]")
        if not noisy_image.is_floating_point():
            raise ValueError("noisy_image must use a floating-point dtype")
        if noisy_image.shape[2:] != (self.image_size, self.image_size):
            raise ValueError(f"noisy_image spatial dimensions must be {self.image_size}x{self.image_size}")
        batch = noisy_image.shape[0]
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.shape[0] != batch or not 1 <= tokens.shape[1] <= self.text_ctx:
            raise ValueError("tokens must have shape [B, K] with 1 <= K <= text_ctx")
        if tokens.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("tokens must contain integer vocabulary IDs")
        if torch.any(tokens < 0) or torch.any(tokens >= self.vocab_size):
            raise ValueError("tokens contain an ID outside the configured vocabulary")
        if self.xf_padding and padding_mask is None:
            raise ValueError("padding_mask is required when xf_padding is enabled")
        if padding_mask is not None:
            if not isinstance(padding_mask, torch.Tensor) or padding_mask.shape != tokens.shape or padding_mask.dtype != torch.bool:
                raise ValueError("padding_mask must be boolean with shape [B, K]")
            if torch.any(~padding_mask.any(dim=1)):
                raise ValueError("padding_mask must mark at least one token as valid per example")
            padding_mask = padding_mask.to(device=noisy_image.device)
        timesteps = torch.as_tensor(timestep, device=noisy_image.device)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(batch)
        if timesteps.shape != (batch,) or timesteps.dtype == torch.bool:
            raise ValueError("timestep must be a scalar or have shape [B]")
        tokens = tokens.to(device=noisy_image.device, dtype=torch.long)
        text = self.text_encoder(tokens, padding_mask)
        global_text = self.text_projection(text[:, -1])
        return self.unet(noisy_image, timesteps, text, padding_mask, global_text)
