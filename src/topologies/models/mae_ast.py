import math

import torch
from torch import nn
from torch.nn import functional as F


class MAEAST(nn.Module):
    architecture_id = "mae-ast"
    PRESETS = {
        "tiny": {
            "num_mels": 128,
            "patch_freq": 16,
            "patch_time": 16,
            "embed_dim": 64,
            "num_heads": 4,
            "encoder_layers": 2,
            "decoder_layers": 1,
            "mlp_ratio": 4.0,
            "mask_ratio": 0.75,
            "mask_strategy": "random",
            "chunk_size": 3,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.0,
            "classification_weight": 1.0,
            "reconstruction_weight": 10.0,
            "layer_norm_first": False,
            "activation": "gelu",
        },
        "small": {
            "num_mels": 128,
            "patch_freq": 16,
            "patch_time": 16,
            "embed_dim": 512,
            "num_heads": 8,
            "encoder_layers": 6,
            "decoder_layers": 2,
            "mlp_ratio": 4.0,
            "mask_ratio": 0.75,
            "mask_strategy": "random",
            "chunk_size": 3,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.0,
            "classification_weight": 1.0,
            "reconstruction_weight": 10.0,
            "layer_norm_first": False,
            "activation": "gelu",
        },
        "base": {
            "num_mels": 128,
            "patch_freq": 16,
            "patch_time": 16,
            "embed_dim": 768,
            "num_heads": 12,
            "encoder_layers": 6,
            "decoder_layers": 2,
            "mlp_ratio": 4.0,
            "mask_ratio": 0.75,
            "mask_strategy": "random",
            "chunk_size": 3,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "activation_dropout": 0.0,
            "classification_weight": 1.0,
            "reconstruction_weight": 10.0,
            "layer_norm_first": False,
            "activation": "gelu",
        },
    }

    def __init__(
        self,
        num_mels=128,
        patch_freq=16,
        patch_time=16,
        embed_dim=768,
        num_heads=12,
        encoder_layers=6,
        decoder_layers=2,
        mlp_ratio=4.0,
        mask_ratio=0.75,
        mask_strategy="random",
        chunk_size=3,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        classification_weight=1.0,
        reconstruction_weight=10.0,
        layer_norm_first=False,
        activation="gelu",
    ):
        super().__init__()
        integer_values = {
            "num_mels": num_mels,
            "patch_freq": patch_freq,
            "patch_time": patch_time,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "encoder_layers": encoder_layers,
            "decoder_layers": decoder_layers,
            "chunk_size": chunk_size,
        }
        for name, value in integer_values.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if patch_freq > num_mels:
            raise ValueError("patch_freq cannot exceed num_mels")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if chunk_size not in (3, 4, 5):
            raise ValueError("chunk_size must be one of 3, 4, or 5")
        if mask_strategy not in ("random", "chunked"):
            raise ValueError("mask_strategy must be 'random' or 'chunked'")
        if type(mask_ratio) not in (float, int) or not math.isfinite(mask_ratio):
            raise ValueError("mask_ratio must be a finite number in [0, 1)")
        if not 0 <= mask_ratio < 1:
            raise ValueError("mask_ratio must be in [0, 1)")
        for name, value in (
            ("dropout", dropout),
            ("attention_dropout", attention_dropout),
            ("activation_dropout", activation_dropout),
        ):
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number in [0, 1)")
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if type(mlp_ratio) not in (float, int) or not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be a positive finite number")
        if int(embed_dim * mlp_ratio) <= 0:
            raise ValueError("mlp_ratio produces an empty feed-forward dimension")
        for name, value in (
            ("classification_weight", classification_weight),
            ("reconstruction_weight", reconstruction_weight),
        ):
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite number")
        if type(layer_norm_first) is not bool:
            raise ValueError("layer_norm_first must be a boolean")
        if activation != "gelu":
            raise ValueError("activation must be 'gelu'")

        self.num_mels = num_mels
        self.patch_freq = patch_freq
        self.patch_time = patch_time
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_ratio = float(mask_ratio)
        self.mask_strategy = mask_strategy
        self.chunk_size = chunk_size
        self.classification_weight = float(classification_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.layer_norm_first = layer_norm_first
        self.patch_dim = patch_freq * patch_time
        self.frequency_patches = num_mels // patch_freq
        self.input_norm = nn.BatchNorm2d(1, affine=False)
        self.patch_projection = nn.Linear(self.patch_dim, embed_dim)
        self.encoder_input_norm = nn.Identity() if layer_norm_first else nn.LayerNorm(embed_dim)
        self.decoder_input_norm = nn.Identity() if layer_norm_first else nn.LayerNorm(embed_dim)

        def make_layers(count):
            layers = []
            for _ in range(count):
                layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout=activation_dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=layer_norm_first,
                )
                layer.self_attn.dropout = float(attention_dropout)
                layer.dropout1 = nn.Dropout(float(dropout))
                layer.dropout2 = nn.Dropout(float(dropout))
                layers.append(layer)
            return nn.ModuleList(layers)

        self.encoder_layers = make_layers(encoder_layers)
        self.decoder_layers = make_layers(decoder_layers)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.reconstruction_head = nn.Linear(embed_dim, self.patch_dim)
        self.classification_head = nn.Linear(embed_dim, self.patch_dim)

    def _normalize(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("input must be a tensor shaped [batch, time, num_mels]")
        if x.shape[0] <= 0 or x.shape[1] < self.patch_time or x.shape[2] != self.num_mels:
            raise ValueError("input must have a non-empty batch, enough frames, and num_mels features")
        return 0.5 * self.input_norm(x.unsqueeze(1)).squeeze(1)

    def _patchify(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("patch input must be a tensor shaped [batch, time, num_mels]")
        if x.shape[-1] != self.num_mels or x.shape[1] < self.patch_time:
            raise ValueError("patch input has incompatible time or feature dimensions")
        unfolded = F.unfold(
            x.unsqueeze(1),
            kernel_size=(self.patch_time, self.patch_freq),
            stride=(self.patch_time, self.patch_freq),
        )
        time_patches = x.shape[1] // self.patch_time
        grid = unfolded.transpose(1, 2).reshape(
            x.shape[0], time_patches, self.frequency_patches, self.patch_dim
        )
        return grid.permute(0, 2, 1, 3).reshape(x.shape[0], -1, self.patch_dim)

    def _position_embeddings(self, length, device, dtype):
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.embed_dim)
        )
        angles = positions * frequencies.unsqueeze(0)
        embeddings = torch.zeros(length, self.embed_dim, device=device, dtype=torch.float32)
        embeddings[:, 0::2] = torch.sin(angles)
        embeddings[:, 1::2] = torch.cos(angles[:, : self.embed_dim // 2])
        return embeddings.to(dtype=dtype).unsqueeze(0)

    def _run_layers(self, x, layers):
        for layer in layers:
            x = layer(x)
        return x

    def _encode_all(self, x):
        patches = self._patchify(self._normalize(x))
        positions = self._position_embeddings(patches.shape[1], patches.device, patches.dtype)
        encoded = self.encoder_input_norm(self.patch_projection(patches) + positions)
        return self._run_layers(encoded, self.encoder_layers)

    def _mask_tokens(self, tokens, mask_ratio=None, mask_strategy=None):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError("tokens must be a tensor shaped [batch, sequence, features]")
        batch_size, token_count, _ = tokens.shape
        if batch_size <= 0 or token_count <= 0:
            raise ValueError("tokens must have a non-empty batch and sequence")
        ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        strategy = self.mask_strategy if mask_strategy is None else mask_strategy
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 <= ratio < 1:
            raise ValueError("mask_ratio must be a finite number in [0, 1)")
        if strategy not in ("random", "chunked"):
            raise ValueError("mask_strategy must be 'random' or 'chunked'")
        visible_count = max(1, int((1.0 - ratio) * token_count))
        masked_count = token_count - visible_count
        if strategy == "random":
            shuffled = torch.rand(batch_size, token_count, device=tokens.device).argsort(dim=1)
            return shuffled[:, :visible_count], shuffled[:, visible_count:]

        if token_count % self.frequency_patches != 0:
            raise ValueError("chunked masking requires a complete frequency-by-time token grid")
        time_patches = token_count // self.frequency_patches
        mask_grid = torch.zeros(
            self.frequency_patches, time_patches, device=tokens.device, dtype=torch.bool
        )
        attempts = max(token_count * 8, 8)
        for _ in range(attempts):
            if int(mask_grid.sum().item()) >= masked_count:
                break
            frequency_start = int(torch.randint(self.frequency_patches, (), device=tokens.device).item())
            time_start = int(torch.randint(time_patches, (), device=tokens.device).item())
            frequency_end = min(frequency_start + self.chunk_size, self.frequency_patches)
            time_end = min(time_start + self.chunk_size, time_patches)
            mask_grid[frequency_start:frequency_end, time_start:time_end] = True
        mask = mask_grid.reshape(-1)
        current_masked = torch.nonzero(mask, as_tuple=False).flatten()
        if current_masked.numel() < masked_count:
            candidates = torch.nonzero(~mask, as_tuple=False).flatten()
            order = torch.randperm(candidates.numel(), device=tokens.device)
            mask[candidates[order[: masked_count - current_masked.numel()]]] = True
            current_masked = torch.nonzero(mask, as_tuple=False).flatten()
        if current_masked.numel() > masked_count:
            order = torch.randperm(current_masked.numel(), device=tokens.device)
            current_masked = current_masked[order[:masked_count]]
            mask.zero_()
            mask[current_masked] = True
        current_visible = torch.nonzero(~mask, as_tuple=False).flatten()
        return (
            current_visible.unsqueeze(0).expand(batch_size, -1),
            current_masked.unsqueeze(0).expand(batch_size, -1),
        )

    def _pretraining_losses(self, reconstruction, classification, targets):
        if reconstruction.shape != classification.shape or reconstruction.shape != targets.shape:
            raise ValueError("reconstruction, classification, and targets must have the same shape")
        if reconstruction.ndim != 3:
            raise ValueError("pretraining predictions and targets must have three dimensions")
        if reconstruction.shape[1] == 0:
            zero = reconstruction.sum() + classification.sum()
            loss_reconstruction = zero
            loss_classification = zero
        else:
            loss_reconstruction = F.mse_loss(reconstruction, targets)
            logits = torch.matmul(classification, targets.transpose(1, 2))
            labels = torch.arange(logits.shape[1], device=logits.device).expand(logits.shape[0], -1)
            loss_classification = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss = (
            self.classification_weight * loss_classification
            + self.reconstruction_weight * loss_reconstruction
        )
        return {
            "loss": loss,
            "loss_classification": loss_classification,
            "loss_reconstruction": loss_reconstruction,
        }

    def _pretrain(self, x, mask_ratio=None, mask_strategy=None):
        patches = self._patchify(self._normalize(x))
        batch_size, token_count, _ = patches.shape
        visible_indices, masked_indices = self._mask_tokens(
            patches, mask_ratio=mask_ratio, mask_strategy=mask_strategy
        )
        positions = self._position_embeddings(token_count, patches.device, patches.dtype)
        visible_patches = torch.gather(
            patches, 1, visible_indices.unsqueeze(-1).expand(-1, -1, self.patch_dim)
        )
        visible_positions = torch.gather(
            positions.expand(batch_size, -1, -1),
            1,
            visible_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )
        encoded = self.encoder_input_norm(self.patch_projection(visible_patches) + visible_positions)
        encoded = self._run_layers(encoded, self.encoder_layers)
        decoder_tokens = self.mask_token.expand(batch_size, token_count, -1).clone()
        decoder_tokens = decoder_tokens.scatter(
            1,
            visible_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            encoded,
        )
        decoder_tokens = self.decoder_input_norm(decoder_tokens + positions)
        decoder_tokens = self._run_layers(decoder_tokens, self.decoder_layers)
        masked_hidden = torch.gather(
            decoder_tokens,
            1,
            masked_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )
        targets = torch.gather(
            patches, 1, masked_indices.unsqueeze(-1).expand(-1, -1, self.patch_dim)
        )
        reconstruction = self.reconstruction_head(masked_hidden)
        classification = self.classification_head(masked_hidden)
        losses = self._pretraining_losses(reconstruction, classification, targets)
        return {
            "reconstruction": reconstruction,
            "classification": classification,
            "targets": targets,
            "visible_indices": visible_indices,
            "masked_indices": masked_indices,
            **losses,
        }

    def forward(self, x, pretrain=False, mask_ratio=None, mask_strategy=None):
        if type(pretrain) is not bool:
            raise ValueError("pretrain must be a boolean")
        if pretrain:
            return self._pretrain(x, mask_ratio=mask_ratio, mask_strategy=mask_strategy)
        if mask_ratio is not None or mask_strategy is not None:
            raise ValueError("mask_ratio and mask_strategy are only valid when pretrain=True")
        return self._encode_all(x).mean(dim=1)
