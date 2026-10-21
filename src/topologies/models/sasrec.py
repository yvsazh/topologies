import torch
from torch import nn


class SASRec(nn.Module):
    architecture_id = "sasrec"
    def __init__(self, num_items, hidden=64, heads=2, layers=2, max_length=100, dropout=0.2):
        super().__init__()
        if num_items < 2 or hidden < 1 or heads < 1 or hidden % heads:
            raise ValueError("num_items and hidden must be valid and hidden must be divisible by heads")
        if layers < 1 or max_length < 1 or not 0 <= dropout < 1:
            raise ValueError("layers, max_length, or dropout is invalid")
        self.num_items = num_items
        self.max_length = max_length
        self.item_embedding = nn.Embedding(num_items, hidden, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, hidden)
        self.input_dropout = nn.Dropout(dropout)
        block = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=hidden * 4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(hidden), enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden)

    def encode(self, item_seq):
        if item_seq.ndim != 2:
            raise ValueError("item_seq must have shape (batch, sequence_length)")
        if item_seq.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("item_seq must contain integer item IDs")
        length = item_seq.shape[1]
        if length < 1 or length > self.max_length:
            raise ValueError("sequence length must be between 1 and max_length")
        if item_seq.numel() and (item_seq.min() < 0 or item_seq.max() >= self.num_items):
            raise ValueError("item_seq contains an out-of-range item ID")
        item_seq = item_seq.long()
        positions = torch.arange(length, device=item_seq.device).unsqueeze(0)
        values = self.input_dropout(self.item_embedding(item_seq) + self.position_embedding(positions))
        future_mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=item_seq.device), diagonal=1)
        padding_mask = item_seq.eq(0)
        all_padding = padding_mask.all(dim=1)
        if all_padding.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padding, 0] = False
        return self.output_norm(self.encoder(values, mask=future_mask, src_key_padding_mask=padding_mask))

    def forward(self, item_seq, positive_items=None, negative_items=None):
        if item_seq.ndim != 2:
            raise ValueError("item_seq must have shape (batch, sequence_length)")
        valid = item_seq.ne(0)
        if not valid.any(dim=1).all():
            raise ValueError("each history must contain at least one non-padding item")
        encoded = self.encode(item_seq)
        position_ids = torch.arange(item_seq.shape[1], device=item_seq.device).expand_as(item_seq)
        last_positions = position_ids.masked_fill(~valid, -1).amax(dim=1)
        hidden = encoded[torch.arange(item_seq.shape[0], device=item_seq.device), last_positions]
        if positive_items is not None or negative_items is not None:
            if positive_items is None or negative_items is None:
                raise ValueError("positive_items and negative_items must be provided together")
            positive_logits = (hidden * self.item_embedding(positive_items)).sum(dim=-1)
            negative_logits = (hidden * self.item_embedding(negative_items)).sum(dim=-1)
            return positive_logits, negative_logits
        return hidden @ self.item_embedding.weight.T
