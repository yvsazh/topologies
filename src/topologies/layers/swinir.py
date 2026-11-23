import math
import numbers

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class SwinIR(nn.Module):
    component_id = "swinir"
    PRESETS = {
        "tiny": {
            "embed_dim": 12,
            "depths": (2,),
            "num_heads": (3,),
            "window_size": 2,
            "mlp_ratio": 2.0,
            "upscale": 2,
            "upsampler": "pixelshuffle",
        },
        "small": {
            "embed_dim": 24,
            "depths": (2, 2),
            "num_heads": (4, 4),
            "window_size": 4,
            "mlp_ratio": 2.0,
            "upscale": 2,
            "upsampler": "pixelshuffle",
        },
    }

    def __init__(
        self,
        upscale=2,
        in_chans=3,
        img_size=64,
        img_range=1.0,
        depths=(6, 6, 6, 6),
        embed_dim=96,
        num_heads=(6, 6, 6, 6),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        patch_size=1,
        use_checkpoint=False,
        upsampler="",
        resi_connection="1conv",
        num_feat=64,
    ):
        super().__init__()
        integers = {
            "upscale": upscale,
            "in_chans": in_chans,
            "embed_dim": embed_dim,
            "window_size": window_size,
            "patch_size": patch_size,
            "num_feat": num_feat,
        }
        for name, value in integers.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(img_size, int) and not isinstance(img_size, bool):
            image_size = (img_size, img_size)
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            image_size = tuple(img_size)
        else:
            raise ValueError("img_size must be a positive integer or a pair of positive integers")
        if any(not isinstance(side, int) or isinstance(side, bool) or side <= 0 for side in image_size):
            raise ValueError("img_size must be a positive integer or a pair of positive integers")
        if patch_size != 1:
            raise ValueError("patch_size must be 1 for image restoration")
        if not isinstance(depths, (tuple, list)) or not depths:
            raise ValueError("depths must be a nonempty sequence of positive integers")
        if any(not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0 for depth in depths):
            raise ValueError("depths must contain only positive integers")
        if not isinstance(num_heads, (tuple, list)) or len(num_heads) != len(depths):
            raise ValueError("num_heads must contain one value for each depth")
        if any(not isinstance(head, int) or isinstance(head, bool) or head <= 0 for head in num_heads):
            raise ValueError("num_heads must contain only positive integers")
        if any(embed_dim % head for head in num_heads):
            raise ValueError("embed_dim must be divisible by every num_heads value")
        if not isinstance(mlp_ratio, numbers.Real) or not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be a positive finite number")
        if int(embed_dim * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce a positive hidden dimension")
        if qk_scale is not None and (
            not isinstance(qk_scale, numbers.Real) or not math.isfinite(qk_scale) or qk_scale <= 0
        ):
            raise ValueError("qk_scale must be a positive finite number or None")
        for name, value in (
            ("img_range", img_range),
            ("drop_rate", drop_rate),
            ("attn_drop_rate", attn_drop_rate),
            ("drop_path_rate", drop_path_rate),
        ):
            if not isinstance(value, numbers.Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if img_range <= 0:
            raise ValueError("img_range must be positive")
        if not 0 <= drop_rate < 1 or not 0 <= attn_drop_rate < 1:
            raise ValueError("drop_rate and attn_drop_rate must be in [0, 1)")
        if not 0 <= drop_path_rate < 1:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if type(qkv_bias) is not bool or type(ape) is not bool or type(patch_norm) is not bool:
            raise ValueError("qkv_bias, ape, and patch_norm must be booleans")
        if type(use_checkpoint) is not bool:
            raise ValueError("use_checkpoint must be a boolean")
        if upsampler not in ("", "pixelshuffle", "pixelshuffledirect", "nearest+conv"):
            raise ValueError("upsampler must be '', 'pixelshuffle', 'pixelshuffledirect', or 'nearest+conv'")
        if upsampler == "pixelshuffle" and upscale != 3 and upscale & (upscale - 1):
            raise ValueError("pixelshuffle supports powers of two or scale 3")
        if upsampler == "nearest+conv" and upscale & (upscale - 1):
            raise ValueError("nearest+conv supports powers-of-two scales")
        if resi_connection not in ("1conv", "3conv"):
            raise ValueError("resi_connection must be '1conv' or '3conv'")

        self.upscale = upscale
        self.in_chans = in_chans
        self.img_range = float(img_range)
        self.depths = tuple(depths)
        self.num_heads = tuple(num_heads)
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.upsampler = upsampler
        self.shift_size = window_size // 2
        self.qk_scale = qk_scale
        self.drop_rate = float(drop_rate)
        self.attn_drop_rate = float(attn_drop_rate)
        self.use_checkpoint = use_checkpoint
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, padding=1)
        self.patch_norm = nn.LayerNorm(embed_dim) if patch_norm else nn.Identity()
        if ape:
            self.absolute_position = nn.Parameter(torch.zeros(1, image_size[0], image_size[1], embed_dim))
            nn.init.trunc_normal_(self.absolute_position, std=0.02)
        else:
            self.register_parameter("absolute_position", None)

        total_blocks = sum(depths)
        self.layers = nn.ModuleList()
        self.relative_position_bias_tables = nn.ParameterList()
        self.block_groups = []
        self.block_indices = []
        for group_index, (depth, heads) in enumerate(zip(depths, num_heads)):
            for block_index in range(depth):
                hidden_dim = int(embed_dim * mlp_ratio)
                self.layers.append(
                    nn.ModuleDict(
                        {
                            "norm1": nn.LayerNorm(embed_dim),
                            "qkv": nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias),
                            "proj": nn.Linear(embed_dim, embed_dim),
                            "norm2": nn.LayerNorm(embed_dim),
                            "mlp1": nn.Linear(embed_dim, hidden_dim),
                            "mlp2": nn.Linear(hidden_dim, embed_dim),
                        }
                    )
                )
                table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, heads))
                nn.init.trunc_normal_(table, std=0.02)
                self.relative_position_bias_tables.append(table)
                self.block_groups.append(group_index)
                self.block_indices.append(block_index)
        rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        self.drop_path_rates = tuple(rates)

        self.rstb_convs = nn.ModuleList()
        for _ in depths:
            if resi_connection == "1conv":
                self.rstb_convs.append(nn.Conv2d(embed_dim, embed_dim, 3, padding=1))
            else:
                reduced = max(1, embed_dim // 4)
                self.rstb_convs.append(
                    nn.Sequential(
                        nn.Conv2d(embed_dim, reduced, 3, padding=1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(reduced, reduced, 1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(reduced, embed_dim, 3, padding=1),
                    )
                )
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

        if upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, padding=1), nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
            stages = []
            remaining = upscale
            while remaining > 1:
                factor = 3 if remaining == 3 else 2
                stages.append(
                    nn.Sequential(
                        nn.Conv2d(num_feat, num_feat * factor * factor, 3, padding=1),
                        nn.PixelShuffle(factor),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
                remaining //= factor
            self.upsample_stages = nn.ModuleList(stages)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, padding=1)
            self.conv_last = nn.Conv2d(num_feat, in_chans, 3, padding=1)
        elif upsampler == "pixelshuffledirect":
            self.conv_last = nn.Conv2d(embed_dim, in_chans * upscale * upscale, 3, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(upscale)
        elif upsampler == "nearest+conv":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, padding=1), nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
            stages = []
            for _ in range(int(math.log2(upscale))):
                stages.append(nn.Conv2d(num_feat, num_feat, 3, padding=1))
            self.nearest_convs = nn.ModuleList(stages)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, padding=1)
            self.conv_last = nn.Conv2d(num_feat, in_chans, 3, padding=1)
        else:
            self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, padding=1)

        mean = (0.4488, 0.4371, 0.4040) if in_chans == 3 else (0.0,) * in_chans
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, in_chans, 1, 1))

    @staticmethod
    def _window_partition(x, window_size):
        if x.ndim != 4:
            raise ValueError("window partition input must have shape [batch, height, width, channels]")
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        batch, height, width, channels = x.shape
        if height % window_size or width % window_size:
            raise ValueError("height and width must be divisible by window_size")
        x = x.contiguous().view(
            batch,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            channels,
        )
        return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, channels)

    @staticmethod
    def _window_reverse(windows, window_size, height, width, batch):
        if height % window_size or width % window_size:
            raise ValueError("height and width must be divisible by window_size")
        channels = windows.shape[-1]
        expected = batch * height * width // (window_size * window_size)
        if windows.ndim != 3 or windows.shape[0] != expected:
            raise ValueError("window count does not match the requested output dimensions")
        x = windows.reshape(
            batch,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            channels,
        )
        return x.permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, channels)

    @staticmethod
    def _window_attention(query, key, value, relative_bias, mask=None, scale=None):
        factor = scale if scale is not None else query.shape[-1] ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * factor
        scores = scores + relative_bias.unsqueeze(0)
        if mask is not None:
            scores = scores + (mask.unsqueeze(1) if mask.ndim == 3 else mask)
        return torch.matmul(torch.softmax(scores, dim=-1), value)

    @staticmethod
    def _relative_position_index(window_size, device):
        coordinates = torch.stack(
            torch.meshgrid(torch.arange(window_size, device=device), torch.arange(window_size, device=device), indexing="ij")
        ).flatten(1)
        relative = coordinates[:, :, None] - coordinates[:, None, :]
        relative[0] += window_size - 1
        relative[1] += window_size - 1
        return relative[0] * (2 * window_size - 1) + relative[1]

    @staticmethod
    def _padding(x, window_size):
        _, height, width, _ = x.shape
        pad_height = (window_size - height % window_size) % window_size
        pad_width = (window_size - width % window_size) % window_size
        if pad_height or pad_width:
            mode = "reflect" if pad_height < height and pad_width < width else "replicate"
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_width, 0, pad_height), mode=mode)
            x = x.permute(0, 2, 3, 1)
        return x, height + pad_height, width + pad_width

    def _shift_mask(self, height, width, shift_size, device):
        if shift_size == 0:
            return None
        mask = torch.zeros((1, height, width, 1), device=device)
        height_slices = (slice(0, -self.window_size), slice(-self.window_size, -shift_size), slice(-shift_size, None))
        width_slices = (slice(0, -self.window_size), slice(-self.window_size, -shift_size), slice(-shift_size, None))
        region = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                mask[:, height_slice, width_slice, :] = region
                region += 1
        windows = self._window_partition(mask, self.window_size).squeeze(-1)
        differences = windows.unsqueeze(1) - windows.unsqueeze(2)
        return torch.where(differences == 0, 0.0, -100.0)

    def _drop_path(self, x, rate):
        if not self.training or rate == 0:
            return x
        keep = 1 - rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (keep + torch.rand(shape, dtype=x.dtype, device=x.device)).floor_()
        return x * mask / keep

    def _run_layer(self, x, layer_index):
        block = self.layers[layer_index]
        batch, height, width, channels = x.shape
        shift = self.shift_size if self.block_indices[layer_index] % 2 else 0
        residual = x
        normalized = block["norm1"](x)
        padded, padded_height, padded_width = self._padding(normalized, self.window_size)
        if shift:
            padded = torch.roll(padded, shifts=(-shift, -shift), dims=(1, 2))
        windows = self._window_partition(padded, self.window_size)
        token_count = self.window_size * self.window_size
        heads = self.num_heads[self.block_groups[layer_index]]
        head_dim = channels // heads
        qkv = block["qkv"](windows).reshape(-1, token_count, 3, heads, head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        position_index = self._relative_position_index(self.window_size, x.device)
        bias_table = self.relative_position_bias_tables[layer_index]
        relative_bias = bias_table[position_index.reshape(-1)].reshape(token_count, token_count, heads)
        relative_bias = relative_bias.permute(2, 0, 1)
        mask = self._shift_mask(padded_height, padded_width, shift, x.device)
        if mask is not None:
            mask = mask.repeat(batch, 1, 1)
        attention = self._window_attention(query, key, value, relative_bias, mask, self.qk_scale)
        attention = F.dropout(attention, p=self.attn_drop_rate, training=self.training)
        attention = attention.transpose(1, 2).reshape(-1, token_count, channels)
        attention = F.dropout(block["proj"](attention), p=self.drop_rate, training=self.training)
        attention = self._window_reverse(
            attention, self.window_size, padded_height, padded_width, batch
        )
        if shift:
            attention = torch.roll(attention, shifts=(shift, shift), dims=(1, 2))
        attention = attention[:, :height, :width, :]
        rate = self.drop_path_rates[layer_index]
        x = residual + self._drop_path(attention, rate)
        mlp = block["mlp1"](block["norm2"](x))
        mlp = F.gelu(mlp)
        mlp = F.dropout(mlp, p=self.drop_rate, training=self.training)
        mlp = block["mlp2"](mlp)
        mlp = F.dropout(mlp, p=self.drop_rate, training=self.training)
        return x + self._drop_path(mlp, rate)

    def _features(self, x):
        batch, _, height, width = x.shape
        shallow = self.conv_first(x)
        tokens = shallow.permute(0, 2, 3, 1)
        tokens = self.patch_norm(tokens)
        if self.absolute_position is not None:
            position = self.absolute_position.permute(0, 3, 1, 2)
            position = F.interpolate(position, size=(height, width), mode="bicubic", align_corners=False)
            tokens = tokens + position.permute(0, 2, 3, 1)
        layer_index = 0
        for group_index, depth in enumerate(self.depths):
            residual = tokens
            for _ in range(depth):
                if self.use_checkpoint and self.training and torch.is_grad_enabled():
                    tokens = checkpoint(self._run_layer, tokens, layer_index, use_reentrant=False)
                else:
                    tokens = self._run_layer(tokens, layer_index)
                layer_index += 1
            group = tokens.permute(0, 3, 1, 2)
            tokens = (self.rstb_convs[group_index](group) + residual.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        tokens = self.norm(tokens).view(batch, height, width, self.embed_dim)
        deep = self.conv_after_body(tokens.permute(0, 3, 1, 2).contiguous()) + shallow
        return shallow, deep

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("input must be a tensor with shape [batch, channels, height, width]")
        if x.shape[1] != self.in_chans:
            raise ValueError(f"input must have {self.in_chans} channels")
        if x.shape[0] <= 0 or x.shape[2] <= 0 or x.shape[3] <= 0:
            raise ValueError("input batch, height, and width must be positive")
        mean = self.mean.to(dtype=x.dtype, device=x.device)
        normalized = (x - mean) * self.img_range
        _, deep = self._features(normalized)
        if self.upsampler == "pixelshuffle":
            output = self.conv_before_upsample(deep)
            for stage in self.upsample_stages:
                output = stage(output)
            output = self.conv_last(F.leaky_relu(self.conv_hr(output), negative_slope=0.2))
        elif self.upsampler == "pixelshuffledirect":
            output = self.pixel_shuffle(self.conv_last(deep))
        elif self.upsampler == "nearest+conv":
            output = self.conv_before_upsample(deep)
            for convolution in self.nearest_convs:
                output = convolution(F.interpolate(output, scale_factor=2, mode="nearest"))
                output = F.leaky_relu(output, negative_slope=0.2)
            output = self.conv_last(F.leaky_relu(self.conv_hr(output), negative_slope=0.2))
        else:
            output = self.conv_last(deep) + normalized
        output = output[..., : x.shape[2] * self.upscale, : x.shape[3] * self.upscale]
        return output / self.img_range + mean
