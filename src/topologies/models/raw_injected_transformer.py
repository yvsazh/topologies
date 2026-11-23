import math

import torch
from torch import nn
from torch.nn import functional as F


class RawInjectedTransformer(nn.Module):
    architecture_id = "raw-injected-transformer"
    PRESETS = {
        "tiny": {
            "data_channels": 3,
            "data_size": 8,
            "patch_size": 4,
            "hidden_size": 32,
            "depth": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "num_classes": 10,
            "bottleneck_dim": 8,
            "in_context_len": 0,
            "in_context_start": 0,
            "patch_size_final": 2,
            "hidden_size_final": 32,
            "depth_final": 1,
            "num_heads_final": 4,
            "decoder_input": True,
            "shortcut_type": "fraction_anchored",
        },
        "small": {
            "data_channels": 3,
            "data_size": 32,
            "patch_size": 8,
            "hidden_size": 128,
            "depth": 4,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "num_classes": 10,
            "bottleneck_dim": 32,
            "in_context_len": 4,
            "in_context_start": 2,
            "patch_size_final": 4,
            "hidden_size_final": 128,
            "depth_final": 2,
            "num_heads_final": 4,
            "decoder_input": True,
            "shortcut_type": "fraction_anchored",
        },
    }

    class _RMSNorm(nn.Module):
        def __init__(self, width, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(width))
            self.eps = eps

        def forward(self, x):
            dtype = x.dtype
            x = x.float()
            x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
            return (x * self.weight).to(dtype)

    class _Attention(nn.Module):
        def __init__(self, width, heads, attn_drop=0.0, proj_drop=0.0, use_sdpa=True):
            super().__init__()
            self.heads = heads
            self.head_dim = width // heads
            self.qkv = nn.Linear(width, width * 3)
            self.q_norm = RawInjectedTransformer._RMSNorm(self.head_dim)
            self.k_norm = RawInjectedTransformer._RMSNorm(self.head_dim)
            self.attn_drop = attn_drop
            self.use_sdpa = use_sdpa
            self.proj = nn.Linear(width, width)
            self.proj_drop = nn.Dropout(proj_drop)

        @staticmethod
        def _rotate(x):
            pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
            return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)

        def forward(self, x, rope_cos, rope_sin):
            batch, length, width = x.shape
            qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            q = self.q_norm(q)
            k = self.k_norm(k)
            cos = rope_cos.to(device=x.device, dtype=q.dtype)[None, None]
            sin = rope_sin.to(device=x.device, dtype=q.dtype)[None, None]
            q = q * cos + self._rotate(q) * sin
            k = k * cos + self._rotate(k) * sin
            if self.use_sdpa:
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.attn_drop if self.training else 0.0
                )
            else:
                weights = torch.matmul(q.float(), k.float().transpose(-2, -1))
                weights = torch.softmax(weights / math.sqrt(self.head_dim), dim=-1)
                weights = F.dropout(weights, p=self.attn_drop, training=self.training)
                x = torch.matmul(weights.to(dtype=v.dtype), v)
            x = x.transpose(1, 2).reshape(batch, length, width)
            return self.proj_drop(self.proj(x))

    class _FeedForward(nn.Module):
        def __init__(self, width, hidden_width, drop=0.0):
            super().__init__()
            hidden_width = max(1, int(hidden_width * 2 / 3))
            self.w12 = nn.Linear(width, hidden_width * 2)
            self.w3 = nn.Linear(hidden_width, width)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            first, second = self.w12(x).chunk(2, dim=-1)
            return self.w3(self.drop(F.silu(first) * second))

    class _Block(nn.Module):
        def __init__(self, width, heads, mlp_ratio, attn_drop=0.0, proj_drop=0.0, use_sdpa=True):
            super().__init__()
            self.norm1 = RawInjectedTransformer._RMSNorm(width)
            self.attention = RawInjectedTransformer._Attention(
                width, heads, attn_drop, proj_drop, use_sdpa
            )
            self.norm2 = RawInjectedTransformer._RMSNorm(width)
            self.mlp = RawInjectedTransformer._FeedForward(width, int(width * mlp_ratio), proj_drop)
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6))

        def forward(self, x, condition, rope_cos, rope_sin):
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(condition).chunk(6, dim=-1)
            )
            attention_input = self.norm1(x) * (1.0 + scale_attn[:, None]) + shift_attn[:, None]
            x = x + gate_attn[:, None] * self.attention(attention_input, rope_cos, rope_sin)
            mlp_input = self.norm2(x) * (1.0 + scale_mlp[:, None]) + shift_mlp[:, None]
            return x + gate_mlp[:, None] * self.mlp(mlp_input)

    class _TimestepEmbedder(nn.Module):
        def __init__(self, width, frequency_width=256):
            super().__init__()
            self.frequency_width = frequency_width
            self.mlp = nn.Sequential(
                nn.Linear(frequency_width, width),
                nn.SiLU(),
                nn.Linear(width, width),
            )

        def forward(self, t):
            half = self.frequency_width // 2
            frequencies = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=t.device, dtype=torch.float32)
                / half
            )
            phase = t.float()[:, None] * frequencies[None]
            embedding = torch.cat((phase.cos(), phase.sin()), dim=-1)
            if self.frequency_width % 2:
                embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
            return self.mlp(embedding.to(dtype=self.mlp[0].weight.dtype))

    @staticmethod
    def _rotary_tables(grid_size, head_dim, prefix_tokens=0):
        axis_width = head_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, axis_width, 2, dtype=torch.float32)
            / axis_width
        )
        positions = torch.arange(grid_size, dtype=torch.float32)
        angles = positions[:, None] * frequencies[None]
        angles = angles.repeat_interleave(2, dim=-1)
        row = angles[:, None, :].expand(grid_size, grid_size, axis_width)
        column = angles[None, :, :].expand(grid_size, grid_size, axis_width)
        angles = torch.cat((row, column), dim=-1).reshape(grid_size * grid_size, head_dim)
        cos = angles.cos()
        sin = angles.sin()
        if prefix_tokens:
            cos = torch.cat((torch.ones(prefix_tokens, head_dim), cos), dim=0)
            sin = torch.cat((torch.zeros(prefix_tokens, head_dim), sin), dim=0)
        return cos, sin

    @staticmethod
    def _position_embedding(grid_size, width):
        axis_width = width // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(axis_width // 2, dtype=torch.float32)
            / (axis_width // 2)
        )
        positions = torch.arange(grid_size, dtype=torch.float32)
        angles = positions[:, None] * frequencies[None]
        axis = torch.cat((angles.sin(), angles.cos()), dim=-1)
        row = axis[:, None, :].expand(grid_size, grid_size, axis_width)
        column = axis[None, :, :].expand(grid_size, grid_size, axis_width)
        return torch.cat((column, row), dim=-1).reshape(1, grid_size * grid_size, width)

    @staticmethod
    def _patchify(x, patch_size):
        batch, channels, height, width = x.shape
        grid_h = height // patch_size
        grid_w = width // patch_size
        return (
            x.reshape(batch, channels, grid_h, patch_size, grid_w, patch_size)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, grid_h * grid_w, channels * patch_size * patch_size)
        )

    @staticmethod
    def _unpatchify(patches, patch_size, channels, image_size):
        batch, count, _ = patches.shape
        grid = image_size // patch_size
        image = (
            patches.reshape(batch, grid, grid, channels, patch_size, patch_size)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, image_size, image_size)
        )
        if count != grid * grid:
            raise ValueError("decoder patch count does not match data_size")
        return image

    class _Decoder(nn.Module):
        def __init__(
            self,
            hidden_size,
            patch_size,
            hidden_size_final,
            patch_size_final,
            num_heads_final,
            depth_final,
            mlp_ratio,
            data_size,
            data_channels,
            decoder_input,
            use_sdpa,
        ):
            super().__init__()
            self.patch_size = patch_size
            self.patch_size_final = patch_size_final
            self.image_size = data_size
            self.channels = data_channels
            self.hidden_size_final = hidden_size_final
            self.ratio = patch_size // patch_size_final
            self.patch_dim_final = data_channels * patch_size_final * patch_size_final
            self.norm = RawInjectedTransformer._RMSNorm(hidden_size_final)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size_final, hidden_size_final * 5)
            )
            if decoder_input:
                self.x_proj = nn.Linear(self.patch_dim_final, hidden_size_final)
            if hidden_size_final != hidden_size or self.ratio != 1:
                self.h_in = nn.Linear(hidden_size, hidden_size_final * self.ratio**2)
            if hidden_size_final != hidden_size:
                self.c_proj = nn.Linear(hidden_size, hidden_size_final)
            grid = data_size // patch_size_final
            rope_cos, rope_sin = RawInjectedTransformer._rotary_tables(
                grid, hidden_size_final // num_heads_final
            )
            self.register_buffer("rope_cos", rope_cos, persistent=False)
            self.register_buffer("rope_sin", rope_sin, persistent=False)
            self.blocks_final = nn.ModuleList(
                RawInjectedTransformer._Block(
                    hidden_size_final, num_heads_final, mlp_ratio, use_sdpa=use_sdpa
                )
                for _ in range(depth_final)
            )
            self.x_in = nn.Linear(self.patch_dim_final, hidden_size_final)
            self.x_out = nn.Linear(hidden_size_final, self.patch_dim_final)
            self.h_out = nn.Linear(hidden_size_final, hidden_size_final)

        def forward(self, h, condition, raw_x_t, t):
            batch, backbone_tokens, _ = h.shape
            backbone_grid = self.image_size // self.patch_size
            if backbone_tokens != backbone_grid * backbone_grid:
                raise ValueError("backbone token count does not match configured patch grid")
            if hasattr(self, "h_in"):
                h = self.h_in(h)
            h = (
                h.reshape(
                    batch,
                    backbone_grid,
                    backbone_grid,
                    self.hidden_size_final,
                    self.ratio,
                    self.ratio,
                )
                .permute(0, 1, 4, 2, 5, 3)
                .reshape(batch, -1, self.hidden_size_final)
            )
            raw_patches = RawInjectedTransformer._patchify(raw_x_t, self.patch_size_final)
            if hasattr(self, "x_proj"):
                h = h + self.x_proj(raw_patches)
            if hasattr(self, "c_proj"):
                condition = self.c_proj(condition)
            for block in self.blocks_final:
                h = block(h, condition, self.rope_cos, self.rope_sin)
            c1, c2, c3, c4, c5 = self.adaLN_modulation(condition).chunk(5, dim=-1)
            t = t.reshape(batch, 1, 1)
            denominator = (t + (1.0 - t) * c2[:, None]).clamp(min=0.05)
            raw_gain = (1.0 + (1.0 - t) * c1[:, None]) / denominator
            hidden_gain = (1.0 + (1.0 - t) * c3[:, None]) / denominator
            raw_features = self.x_in(raw_patches)
            hidden_features = self.h_out(
                self.norm(h) * (1.0 + c4[:, None]) + c5[:, None]
            )
            output_patches = self.x_out(raw_gain * raw_features + hidden_gain * hidden_features)
            return RawInjectedTransformer._unpatchify(
                output_patches, self.patch_size_final, self.channels, self.image_size
            )

    def __init__(
        self,
        data_channels=3,
        data_size=256,
        patch_size=16,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=4,
        patch_size_final=8,
        hidden_size_final=384,
        depth_final=2,
        num_heads_final=6,
        decoder_input=True,
        shortcut_type="fraction_anchored",
        use_F_sdpa=True,
    ):
        super().__init__()
        integer_values = {
            "data_channels": data_channels,
            "data_size": data_size,
            "patch_size": patch_size,
            "hidden_size": hidden_size,
            "depth": depth,
            "num_heads": num_heads,
            "num_classes": num_classes,
            "bottleneck_dim": bottleneck_dim,
            "in_context_len": in_context_len,
            "in_context_start": in_context_start,
            "patch_size_final": patch_size_final,
            "hidden_size_final": hidden_size_final,
            "depth_final": depth_final,
            "num_heads_final": num_heads_final,
        }
        if any(type(value) is not int for value in integer_values.values()):
            raise ValueError("integer model dimensions and context settings must be integers")
        if data_channels < 1 or num_classes < 1:
            raise ValueError("data_channels and num_classes must be positive")
        if data_size < 1 or patch_size < 1 or patch_size > data_size:
            raise ValueError("data_size and patch_size must be positive and patch_size must fit data_size")
        if data_size % patch_size:
            raise ValueError("data_size must be divisible by patch_size")
        if patch_size_final < 1 or patch_size_final > patch_size or patch_size % patch_size_final:
            raise ValueError("patch_size_final must be a positive divisor of patch_size")
        if data_size % patch_size_final:
            raise ValueError("data_size must be divisible by patch_size_final")
        if hidden_size < 4 or hidden_size % 4 or hidden_size_final < 4 or hidden_size_final % 4:
            raise ValueError("hidden sizes must be positive multiples of four")
        if num_heads < 1 or hidden_size % num_heads or (hidden_size // num_heads) % 4:
            raise ValueError("hidden_size must divide evenly into heads with head dimensions divisible by four")
        if num_heads_final < 1 or hidden_size_final % num_heads_final or (hidden_size_final // num_heads_final) % 4:
            raise ValueError("hidden_size_final must divide evenly into heads with head dimensions divisible by four")
        if depth < 1 or depth_final < 0 or bottleneck_dim < 1:
            raise ValueError("depth and bottleneck_dim must be positive and depth_final cannot be negative")
        if type(mlp_ratio) not in (int, float) or not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if (
            type(attn_drop) not in (int, float)
            or type(proj_drop) not in (int, float)
            or not math.isfinite(attn_drop)
            or not math.isfinite(proj_drop)
            or not 0.0 <= attn_drop < 1.0
            or not 0.0 <= proj_drop < 1.0
        ):
            raise ValueError("dropout probabilities must be in [0, 1)")
        if in_context_len < 0 or not 0 <= in_context_start < depth:
            raise ValueError("in_context_start must identify a backbone block")
        if in_context_len > 0 and in_context_start >= depth:
            raise ValueError("in_context_start must identify a block when context tokens are enabled")
        if type(decoder_input) is not bool or type(use_F_sdpa) is not bool:
            raise ValueError("decoder_input and use_F_sdpa must be boolean")
        if shortcut_type != "fraction_anchored":
            raise ValueError("shortcut_type must be 'fraction_anchored'")

        self.data_channels = data_channels
        self.data_size = data_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.t_embedder = self._TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)
        patch_dim = data_channels * patch_size * patch_size
        self.x_embedder = nn.Sequential(
            nn.Linear(patch_dim, bottleneck_dim, bias=False),
            nn.Linear(bottleneck_dim, hidden_size),
        )
        grid = data_size // patch_size
        self.register_buffer("pos_embed", self._position_embedding(grid, hidden_size), persistent=True)
        if in_context_len:
            self.in_context_posemb = nn.Parameter(torch.empty(1, in_context_len, hidden_size))
        backbone_cos, backbone_sin = self._rotary_tables(grid, hidden_size // num_heads)
        context_cos, context_sin = self._rotary_tables(
            grid, hidden_size // num_heads, in_context_len
        )
        self.register_buffer("feat_rope_cos", backbone_cos, persistent=False)
        self.register_buffer("feat_rope_sin", backbone_sin, persistent=False)
        self.register_buffer("feat_rope_incontext_cos", context_cos, persistent=False)
        self.register_buffer("feat_rope_incontext_sin", context_sin, persistent=False)
        self.blocks = nn.ModuleList()
        for index in range(depth):
            use_dropout = depth // 4 * 3 > index >= depth // 4
            self.blocks.append(
                self._Block(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    attn_drop if use_dropout else 0.0,
                    proj_drop if use_dropout else 0.0,
                    use_F_sdpa,
                )
            )
        self.decoder = self._Decoder(
            hidden_size,
            patch_size,
            hidden_size_final,
            patch_size_final,
            num_heads_final,
            depth_final,
            mlp_ratio,
            data_size,
            data_channels,
            decoder_input,
            use_F_sdpa,
        )
        self.apply(self._initialize_linear)
        if in_context_len:
            nn.init.normal_(self.in_context_posemb, std=0.02)
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        with torch.no_grad():
            for block in self.blocks:
                block.adaLN_modulation[-1].weight.zero_()
                block.adaLN_modulation[-1].bias.zero_()
            for block in self.decoder.blocks_final:
                block.adaLN_modulation[-1].weight.zero_()
                block.adaLN_modulation[-1].bias.zero_()
            decoder = self.decoder
            nn.init.orthogonal_(decoder.x_in.weight)
            decoder.x_in.bias.zero_()
            decoder.x_out.weight.copy_(decoder.x_in.weight.T)
            decoder.x_out.bias.zero_()
            decoder.h_out.weight.zero_()
            decoder.h_out.bias.zero_()
            if hasattr(decoder, "x_proj"):
                nn.init.orthogonal_(decoder.x_proj.weight)
                decoder.x_proj.bias.zero_()
            decoder.adaLN_modulation[-1].weight.zero_()
            decoder.adaLN_modulation[-1].bias.zero_()

    @staticmethod
    def _initialize_linear(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, t, y):
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("x must be a rank-four image tensor [batch, channels, height, width]")
        if x.shape[0] < 1 or x.shape[1] != self.data_channels:
            raise ValueError("x batch size must be positive and channels must match data_channels")
        if x.shape[-2] != x.shape[-1]:
            raise ValueError("x images must be square")
        if x.shape[-2:] != (self.data_size, self.data_size):
            raise ValueError("x spatial dimensions must match configured data_size")
        if not x.is_floating_point():
            raise ValueError("x must have a floating point dtype")
        if not isinstance(t, torch.Tensor) or t.shape != (x.shape[0],):
            raise ValueError("t must have shape [batch] matching x batch size")
        if not t.is_floating_point():
            raise ValueError("t must have a floating point dtype")
        if not bool(torch.isfinite(t).all()) or bool((t < 0).any()) or bool((t > 1).any()):
            raise ValueError("t values must be finite and in [0, 1]")
        if not isinstance(y, torch.Tensor) or y.shape != (x.shape[0],):
            raise ValueError("labels must have shape [batch] matching x batch size")
        if y.dtype == torch.bool or y.is_floating_point() or y.is_complex():
            raise ValueError("labels must use an integer dtype")
        if bool((y < 0).any()) or bool((y > self.num_classes).any()):
            raise ValueError("labels must be between 0 and num_classes, inclusive")
        time_condition = self.t_embedder(t)
        label_condition = self.y_embedder(y.long())
        condition = time_condition + label_condition
        tokens = self.x_embedder(self._patchify(x, self.patch_size)) + self.pos_embed
        for index, block in enumerate(self.blocks):
            rope_cos = self.feat_rope_cos
            rope_sin = self.feat_rope_sin
            if self.in_context_len and index == self.in_context_start:
                context = label_condition[:, None] + self.in_context_posemb
                tokens = torch.cat((context, tokens), dim=1)
            if self.in_context_len and index >= self.in_context_start:
                rope_cos = self.feat_rope_incontext_cos
                rope_sin = self.feat_rope_incontext_sin
            tokens = block(tokens, condition, rope_cos, rope_sin)
        if self.in_context_len:
            tokens = tokens[:, self.in_context_len:]
        return self.decoder(tokens, condition, x, t)
