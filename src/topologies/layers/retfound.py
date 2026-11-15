import math

import torch
from torch import nn


def _sincos_1d(positions, dim):
    frequencies = torch.arange(dim // 2, dtype=torch.float32) / (dim / 2)
    frequencies = 1.0 / (10000.0 ** frequencies)
    angles = positions.reshape(-1, 1).to(torch.float32) * frequencies.reshape(1, -1)
    return torch.cat((angles.sin(), angles.cos()), dim=1)


def _sincos_2d(size, dim):
    axis = torch.arange(size, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(axis, axis, indexing="ij")
    embedding = torch.cat(
        (_sincos_1d(grid_h.reshape(-1), dim // 2),
         _sincos_1d(grid_w.reshape(-1), dim // 2)),
        dim=1,
    )
    return torch.cat((torch.zeros(1, dim), embedding), dim=0).unsqueeze(0)


class RETFound(nn.Module):
    component_id = "retfound"
    PRESETS = {
        "tiny": {
            "img_size": 32,
            "patch_size": 16,
            "in_chans": 3,
            "embed_dim": 32,
            "depth": 1,
            "num_heads": 4,
            "decoder_embed_dim": 16,
            "decoder_depth": 1,
            "decoder_num_heads": 4,
            "mlp_ratio": 4.0,
            "mask_ratio": 0.75,
            "norm_pix_loss": False,
        },
        "small": {
            "img_size": 64,
            "patch_size": 16,
            "in_chans": 3,
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 4,
            "decoder_embed_dim": 32,
            "decoder_depth": 2,
            "decoder_num_heads": 4,
            "mlp_ratio": 4.0,
            "mask_ratio": 0.75,
            "norm_pix_loss": False,
        },
    }

    def __init__(
        self,
        img_size=32,
        patch_size=16,
        in_chans=3,
        embed_dim=32,
        depth=1,
        num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
        mlp_ratio=4.0,
        mask_ratio=0.75,
        norm_pix_loss=False,
    ):
        super().__init__()
        integer_values = {
            "img_size": img_size,
            "patch_size": patch_size,
            "in_chans": in_chans,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "decoder_embed_dim": decoder_embed_dim,
            "decoder_depth": decoder_depth,
            "decoder_num_heads": decoder_num_heads,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if img_size % patch_size:
            raise ValueError("img_size must be divisible by patch_size")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if decoder_embed_dim % decoder_num_heads:
            raise ValueError("decoder_embed_dim must be divisible by decoder_num_heads")
        if embed_dim % 4 or decoder_embed_dim % 4:
            raise ValueError("embed_dim and decoder_embed_dim must be divisible by 4")
        if isinstance(mlp_ratio, bool) or not isinstance(mlp_ratio, (int, float)):
            error_type = ValueError
            raise error_type("mlp_ratio must be a positive finite number")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be a positive finite number")
        if int(embed_dim * mlp_ratio) <= 0 or int(decoder_embed_dim * mlp_ratio) <= 0:
            raise ValueError("mlp_ratio must produce positive hidden dimensions")
        if not isinstance(norm_pix_loss, bool):
            error_type = ValueError
            raise error_type("norm_pix_loss must be a bool")
        self._validate_mask_ratio(mask_ratio)

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.mask_ratio = float(mask_ratio)
        self.norm_pix_loss = norm_pix_loss
        self.num_patches_per_side = img_size // patch_size
        self.num_patches = self.num_patches_per_side**2
        self.patch_dim = in_chans * patch_size**2

        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(_sincos_2d(self.num_patches_per_side, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList(
            [self._make_block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            _sincos_2d(self.num_patches_per_side, decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList(
            [self._make_block(decoder_embed_dim, decoder_num_heads, mlp_ratio)
             for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim)
        self.initialize_weights()

    @staticmethod
    def _make_block(dim, heads, mlp_ratio):
        return nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            layer_norm_eps=1e-6,
            batch_first=True,
            norm_first=True,
        )

    def initialize_weights(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.patch_embed.weight.flatten(1))
            nn.init.zeros_(self.patch_embed.bias)
            self.cls_token.normal_(std=0.02)
            self.mask_token.normal_(std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _validate_mask_ratio(mask_ratio):
        if isinstance(mask_ratio, bool) or not isinstance(mask_ratio, (int, float)):
            error_type = ValueError
            raise error_type("mask_ratio must be a number in (0, 1]")
        if not math.isfinite(mask_ratio) or not 0 < mask_ratio <= 1:
            raise ValueError("mask_ratio must be a number in (0, 1]")

    def patchify(self, images):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")
        if images.shape[1] != self.in_chans:
            raise ValueError("image channel count must match in_chans")
        height, width = images.shape[-2:]
        if height != width:
            raise ValueError("images must be square")
        if height % self.patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        if height != self.img_size:
            raise ValueError("image dimensions must match img_size")
        batch, channels = images.shape[:2]
        grid = height // self.patch_size
        patches = images.reshape(
            batch, channels, grid, self.patch_size, grid, self.patch_size
        )
        return patches.permute(0, 2, 4, 1, 3, 5).reshape(
            batch, grid * grid, channels * self.patch_size**2
        )

    def unpatchify(self, patches):
        if not isinstance(patches, torch.Tensor) or patches.ndim != 3:
            raise ValueError("patches must have shape (batch, number_of_patches, patch_dim)")
        if patches.shape[1:] != (self.num_patches, self.patch_dim):
            raise ValueError("patches must match the configured patch count and patch dimension")
        batch = patches.shape[0]
        grid = self.num_patches_per_side
        images = patches.reshape(
            batch, grid, grid, self.in_chans, self.patch_size, self.patch_size
        )
        return images.permute(0, 3, 1, 4, 2, 5).reshape(
            batch, self.in_chans, self.img_size, self.img_size
        )

    def random_masking(self, tokens, mask_ratio=None):
        ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        self._validate_mask_ratio(ratio)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError("tokens must have shape (batch, sequence, embedding)")
        batch, length, width = tokens.shape
        if length <= 0 or width <= 0:
            raise ValueError("tokens must have positive sequence and embedding dimensions")
        keep_count = int(length * (1 - ratio))
        noise = torch.rand(batch, length, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep_count]
        visible = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, width))
        shuffled_mask = torch.ones(batch, length, device=tokens.device)
        shuffled_mask[:, :keep_count] = 0
        mask = torch.gather(shuffled_mask, 1, ids_restore)
        return visible, mask, ids_restore

    def forward_encoder(self, images, mask_ratio=None):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")
        self.patchify(images)
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, 1:, :]
        tokens, mask, ids_restore = self.random_masking(tokens, mask_ratio)
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(tokens.shape[0], -1, -1)
        latents = torch.cat((cls, tokens), dim=1)
        for block in self.blocks:
            latents = block(latents)
        return self.norm(latents), mask, ids_restore

    def forward_decoder(self, latents, ids_restore):
        latents = self.decoder_embed(latents)
        visible = latents[:, 1:, :]
        missing_count = ids_restore.shape[1] - visible.shape[1]
        mask_tokens = self.mask_token.expand(latents.shape[0], missing_count, -1)
        shuffled = torch.cat((visible, mask_tokens), dim=1)
        restored = torch.gather(
            shuffled,
            1,
            ids_restore.unsqueeze(-1).expand(-1, -1, shuffled.shape[-1]),
        )
        latents = torch.cat((latents[:, :1, :], restored), dim=1)
        latents = latents + self.decoder_pos_embed
        for block in self.decoder_blocks:
            latents = block(latents)
        latents = self.decoder_norm(latents)
        return self.decoder_pred(latents[:, 1:, :])

    def forward_loss(self, images, predictions, mask):
        targets = self.patchify(images)
        if predictions.shape != targets.shape:
            raise ValueError("predictions must match the target patch shape")
        if mask.shape != targets.shape[:2]:
            raise ValueError("mask must match the batch and patch dimensions")
        if self.norm_pix_loss:
            mean = targets.mean(dim=-1, keepdim=True)
            variance = targets.var(dim=-1, keepdim=True)
            targets = (targets - mean) / torch.sqrt(variance + 1e-6)
        per_patch = (predictions - targets).pow(2).mean(dim=-1)
        mask = mask.to(device=per_patch.device, dtype=per_patch.dtype)
        return (per_patch * mask).sum() / mask.sum()

    def forward(self, images, mask_ratio=None, *, return_details=True):
        if not isinstance(return_details, bool):
            error_type = ValueError
            raise error_type("return_details must be a bool")
        ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        self._validate_mask_ratio(ratio)
        latents, mask, ids_restore = self.forward_encoder(images, ratio)
        predictions = self.forward_decoder(latents, ids_restore)
        if not return_details:
            return predictions
        loss = self.forward_loss(images, predictions, mask)
        return loss, predictions, mask
