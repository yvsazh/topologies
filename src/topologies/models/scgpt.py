import math

import torch
from torch import nn


def masked_mse_loss(predictions, targets, masked_positions=None):
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if predictions.numel() == 0:
        raise ValueError("predictions and targets must not be empty")
    squared_errors = (predictions - targets).square()
    if masked_positions is None:
        return squared_errors.mean()
    if masked_positions.shape != predictions.shape or masked_positions.dtype != torch.bool:
        raise ValueError("masked_positions must be boolean and match predictions")
    count = masked_positions.sum()
    if count == 0:
        raise ValueError("masked_positions must select at least one value")
    return squared_errors.masked_select(masked_positions).sum() / count


class MaskedGeneValuePredictor(nn.Module):
    architecture_id = "scgpt"
    PRESETS = {
        "tiny": {
            "n_genes": 16,
            "d_model": 16,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "mask_value": -1.0,
            "max_value": 512.0,
        },
        "small": {
            "n_genes": 256,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 128,
            "dropout": 0.1,
            "mask_value": -1.0,
            "max_value": 512.0,
        },
    }

    def __init__(
        self,
        n_genes,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        mask_value=-1.0,
        max_value=512.0,
    ):
        super().__init__()
        integer_args = (n_genes, d_model, nhead, num_layers, dim_feedforward)
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in integer_args):
            raise ValueError("n_genes, d_model, nhead, num_layers, and dim_feedforward must be positive integers")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0.0 <= dropout < 1.0
        ):
            raise ValueError("dropout must be finite and in [0, 1)")
        if (
            isinstance(mask_value, bool)
            or not isinstance(mask_value, (int, float))
            or not math.isfinite(mask_value)
        ):
            raise ValueError("mask_value must be finite")
        if (
            isinstance(max_value, bool)
            or not isinstance(max_value, (int, float))
            or not math.isfinite(max_value)
            or max_value <= 0
        ):
            raise ValueError("max_value must be finite and positive")
        self.n_genes = n_genes
        self.d_model = d_model
        self.mask_value = float(mask_value)
        self.max_value = float(max_value)
        self.gene_embedding = nn.Embedding(n_genes, d_model)
        self.gene_norm = nn.LayerNorm(d_model)
        self.value_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.expression_head = nn.Linear(d_model, 1)

    def forward(self, gene_ids, values, masked_positions):
        try:
            gene_ids = torch.as_tensor(gene_ids, device=self.gene_embedding.weight.device)
            values = torch.as_tensor(
                values,
                dtype=self.value_encoder[0].weight.dtype,
                device=self.value_encoder[0].weight.device,
            )
            masked_positions = torch.as_tensor(masked_positions, device=gene_ids.device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("gene_ids, values, and masked_positions must be regular tensors") from error
        if gene_ids.ndim != 2 or values.ndim != 2 or masked_positions.ndim != 2:
            raise ValueError("gene_ids, values, and masked_positions must have shape (batch, sequence)")
        if gene_ids.shape != values.shape or gene_ids.shape != masked_positions.shape:
            raise ValueError("gene_ids, values, and masked_positions must have the same shape")
        if gene_ids.shape[0] < 1 or gene_ids.shape[1] < 1:
            raise ValueError("batch and sequence dimensions must be positive")
        if gene_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("gene IDs must have an integer dtype")
        if masked_positions.dtype != torch.bool:
            raise ValueError("masked_positions must have a boolean dtype")
        if gene_ids.numel() and ((gene_ids < 0).any() or (gene_ids >= self.n_genes).any()):
            raise ValueError("gene IDs must be within the configured vocabulary")
        counts = masked_positions.sum(dim=1)
        if (counts == 0).any():
            raise ValueError("masked_positions must select at least one value per sample")
        if not (counts == counts[0]).all():
            raise ValueError("masked_positions must select the same number of values per sample")
        input_values = values.masked_fill(masked_positions, self.mask_value)
        gene_tokens = self.gene_norm(self.gene_embedding(gene_ids.long()))
        value_tokens = self.value_encoder(input_values.clamp(max=self.max_value).unsqueeze(-1))
        hidden = self.transformer(gene_tokens + value_tokens)
        predictions = self.expression_head(hidden).squeeze(-1)
        return predictions.masked_select(masked_positions).reshape(gene_ids.shape[0], -1)
