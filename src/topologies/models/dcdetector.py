import math

import torch
from torch import nn


class DCdetector(nn.Module):
    architecture_id = "dcdetector"
    PRESETS = {
        "tiny": {
            "window_size": 8,
            "patch_sizes": [2, 4],
            "d_model": 16,
            "n_heads": 1,
            "e_layers": 1,
            "dropout": 0.0,
        },
        "small": {
            "window_size": 32,
            "patch_sizes": [2, 4, 8],
            "d_model": 64,
            "n_heads": 2,
            "e_layers": 2,
            "dropout": 0.0,
        },
        "base": {
            "window_size": 100,
            "patch_sizes": [5],
            "d_model": 256,
            "n_heads": 1,
            "e_layers": 3,
            "dropout": 0.0,
        },
    }

    def __init__(
        self,
        window_size=100,
        patch_sizes=(5,),
        d_model=256,
        n_heads=1,
        e_layers=3,
        dropout=0.0,
        revin_eps=1e-5,
        kl_eps=1e-4,
    ):
        super().__init__()
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if not isinstance(patch_sizes, (list, tuple)) or not patch_sizes:
            raise ValueError("patch_sizes must be a nonempty list or tuple of positive integers")
        if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in patch_sizes):
            raise ValueError("patch_sizes must contain only positive integers")
        if any(window_size % size for size in patch_sizes):
            raise ValueError("window_size must be divisible by every patch size")
        if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if isinstance(n_heads, bool) or not isinstance(n_heads, int) or n_heads <= 0:
            raise ValueError("n_heads must be a positive integer")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if isinstance(e_layers, bool) or not isinstance(e_layers, int) or e_layers <= 0:
            raise ValueError("e_layers must be a positive integer")
        try:
            valid_dropout = (
                not isinstance(dropout, bool)
                and isinstance(dropout, (int, float))
                and math.isfinite(dropout)
                and 0.0 <= dropout < 1.0
            )
        except TypeError:
            valid_dropout = False
        if not valid_dropout:
            raise ValueError("dropout must be a finite number in [0, 1)")
        try:
            valid_revin_eps = (
                not isinstance(revin_eps, bool)
                and isinstance(revin_eps, (int, float))
                and math.isfinite(revin_eps)
                and revin_eps > 0
            )
        except TypeError:
            valid_revin_eps = False
        if not valid_revin_eps:
            raise ValueError("revin_eps must be a positive finite number")
        try:
            valid_kl_eps = (
                not isinstance(kl_eps, bool)
                and isinstance(kl_eps, (int, float))
                and math.isfinite(kl_eps)
                and kl_eps > 0
            )
        except TypeError:
            valid_kl_eps = False
        if not valid_kl_eps:
            raise ValueError("kl_eps must be a positive finite number")

        self.window_size = window_size
        self.patch_sizes = tuple(patch_sizes)
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.revin_eps = float(revin_eps)
        self.kl_eps = float(kl_eps)
        self.patch_size_embeddings = nn.ModuleList(
            nn.Linear(patch_size, d_model) for patch_size in self.patch_sizes
        )
        self.patch_num_embeddings = nn.ModuleList(
            nn.Linear(window_size // patch_size, d_model)
            for patch_size in self.patch_sizes
        )
        head_dim = d_model // n_heads
        self.query_weights = nn.Parameter(
            torch.empty(e_layers, n_heads, head_dim, head_dim)
        )
        self.key_weights = nn.Parameter(
            torch.empty(e_layers, n_heads, head_dim, head_dim)
        )
        self.embedding_dropout = nn.Dropout(float(dropout))
        self.attention_dropout = nn.Dropout(float(dropout))
        for layer in range(e_layers):
            for head in range(n_heads):
                nn.init.xavier_uniform_(self.query_weights[layer, head])
                nn.init.xavier_uniform_(self.key_weights[layer, head])

    def _attention(self, tokens, layer_index):
        batch, length, _ = tokens.shape
        head_dim = self.d_model // self.n_heads
        tokens = tokens.reshape(batch, length, self.n_heads, head_dim)
        queries = torch.einsum(
            "blhd,hde->blhe", tokens, self.query_weights[layer_index]
        )
        keys = torch.einsum("blhd,hde->blhe", tokens, self.key_weights[layer_index])
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) / math.sqrt(head_dim)
        return self.attention_dropout(torch.softmax(scores, dim=-1))

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("x must have rank 3 with shape [batch, window_size, channels]")
        if x.shape[1] != self.window_size:
            raise ValueError("x sequence length must equal window_size")
        if x.shape[0] <= 0 or x.shape[2] <= 0:
            raise ValueError("x batch size and channel count must be positive")
        if not torch.is_floating_point(x):
            raise ValueError("x must have a floating-point dtype")
        if x.dtype != self.query_weights.dtype:
            raise ValueError("x dtype must match the model parameter dtype")

        batch, _, channels = x.shape
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, unbiased=False, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.revin_eps)
        flattened = normalized.permute(0, 2, 1).reshape(batch * channels, self.window_size)
        patchwise_maps = []
        inpatch_maps = []

        for scale_index, patch_size in enumerate(self.patch_sizes):
            patch_count = self.window_size // patch_size
            patches = flattened.reshape(batch * channels, patch_count, patch_size)
            patchwise_tokens = self.embedding_dropout(
                self.patch_size_embeddings[scale_index](patches)
            )
            inpatch_input = patches.transpose(1, 2)
            inpatch_tokens = self.embedding_dropout(
                self.patch_num_embeddings[scale_index](inpatch_input)
            )

            for layer_index in range(self.e_layers):
                patchwise_attention = self._attention(patchwise_tokens, layer_index)
                inpatch_attention = self._attention(inpatch_tokens, layer_index)
                patchwise_attention = patchwise_attention.repeat_interleave(
                    patch_size, dim=-2
                ).repeat_interleave(patch_size, dim=-1)
                inpatch_attention = inpatch_attention.repeat(
                    1, 1, patch_count, patch_count
                )
                patchwise_maps.append(
                    patchwise_attention.reshape(
                        batch, channels, self.n_heads, self.window_size, self.window_size
                    ).mean(dim=1)
                )
                inpatch_maps.append(
                    inpatch_attention.reshape(
                        batch, channels, self.n_heads, self.window_size, self.window_size
                    ).mean(dim=1)
                )

        return patchwise_maps, inpatch_maps

    def contrastive_loss(self, patchwise_maps, inpatch_maps):
        if not isinstance(patchwise_maps, (list, tuple)) or not patchwise_maps:
            raise ValueError("patchwise_maps must be a nonempty list or tuple")
        if not isinstance(inpatch_maps, (list, tuple)) or len(inpatch_maps) != len(patchwise_maps):
            raise ValueError("patchwise_maps and inpatch_maps must have the same nonzero length")
        loss_patchwise = patchwise_maps[0].new_zeros(())
        loss_inpatch = inpatch_maps[0].new_zeros(())
        for patchwise, inpatch in zip(patchwise_maps, inpatch_maps):
            if not isinstance(patchwise, torch.Tensor) or not isinstance(inpatch, torch.Tensor):
                raise TypeError("every attention map must be a tensor")
            if patchwise.shape != inpatch.shape or patchwise.ndim != 4:
                raise ValueError("paired attention maps must have the same rank-4 shape")
            stop_patchwise = patchwise.detach()
            stop_inpatch = inpatch.detach()
            loss_patchwise = loss_patchwise + self._kl(inpatch, stop_patchwise)
            loss_patchwise = loss_patchwise + self._kl(stop_patchwise, inpatch)
            loss_inpatch = loss_inpatch + self._kl(patchwise, stop_inpatch)
            loss_inpatch = loss_inpatch + self._kl(stop_inpatch, patchwise)
        return (loss_inpatch - loss_patchwise) / len(patchwise_maps)

    def anomaly_score(self, patchwise_maps, inpatch_maps):
        if not isinstance(patchwise_maps, (list, tuple)) or not patchwise_maps:
            raise ValueError("patchwise_maps must be a nonempty list or tuple")
        if not isinstance(inpatch_maps, (list, tuple)) or len(inpatch_maps) != len(patchwise_maps):
            raise ValueError("patchwise_maps and inpatch_maps must have the same nonzero length")
        scores = None
        for patchwise, inpatch in zip(patchwise_maps, inpatch_maps):
            if not isinstance(patchwise, torch.Tensor) or not isinstance(inpatch, torch.Tensor):
                raise TypeError("every attention map must be a tensor")
            if patchwise.shape != inpatch.shape or patchwise.ndim != 4:
                raise ValueError("paired attention maps must have the same rank-4 shape")
            epsilon = self.kl_eps
            point_scores = (
                inpatch * ((inpatch + epsilon) / (patchwise + epsilon)).log()
                + patchwise * ((patchwise + epsilon) / (inpatch + epsilon)).log()
            ).sum(dim=1).sum(dim=-1)
            scores = point_scores if scores is None else scores + point_scores
        return scores

    def _kl(self, left, right):
        epsilon = self.kl_eps
        return (left * ((left + epsilon).log() - (right + epsilon).log())).sum()
