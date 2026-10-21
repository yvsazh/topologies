import torch
from torch import nn
from torch.nn import functional as F


class DilatedCausalResidualBlock(nn.Module):
    def __init__(self, hidden_size, dilation, kernel_size, dropout):
        super().__init__()
        self.dilation = dilation
        self.padding = dilation * (kernel_size - 1)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size, dilation=dilation)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        update = self.norm1(values).relu().transpose(1, 2)
        update = self.conv1(F.pad(update, (self.padding, 0)))
        update = self.norm2(update.transpose(1, 2)).relu().transpose(1, 2)
        update = self.conv2(F.pad(update, (self.padding, 0))).transpose(1, 2)
        return values + self.dropout(update)


class NextItNet(nn.Module):
    architecture_id = "nextitnet"
    def __init__(self, num_items, hidden_size=64, num_blocks=4, kernel_size=3, dropout=0.0, padding_idx=None, max_len=None):
        super().__init__()
        if num_items <= 0 or hidden_size <= 0 or num_blocks <= 0 or kernel_size <= 0:
            raise ValueError("num_items, hidden_size, num_blocks, and kernel_size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if padding_idx is not None and not 0 <= padding_idx < num_items:
            raise ValueError("padding_idx must be a valid item index")
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.max_len = max_len
        self.item_embedding = nn.Embedding(num_items, hidden_size, padding_idx=padding_idx)
        self.blocks = nn.ModuleList(
            DilatedCausalResidualBlock(hidden_size, 2 ** (index % 4), kernel_size, dropout)
            for index in range(num_blocks)
        )
        self.output_projection = nn.Linear(hidden_size, num_items)

    def forward(self, item_ids, last_only=False):
        if item_ids.ndim != 2 or item_ids.shape[1] == 0:
            raise ValueError("item_ids must have shape (batch, nonempty_time)")
        if item_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("item_ids must contain integer item indices")
        if self.max_len is not None and item_ids.shape[1] > self.max_len:
            raise ValueError("sequence exceeds max_len")
        item_ids = item_ids.long()
        if torch.any(item_ids < 0) or torch.any(item_ids >= self.num_items):
            raise ValueError("item_ids contain an index outside the item vocabulary")
        values = self.item_embedding(item_ids)
        for block in self.blocks:
            values = block(values)
        if last_only:
            if self.padding_idx is None:
                values = values[:, -1]
            else:
                valid = item_ids.ne(self.padding_idx)
                positions = torch.arange(item_ids.shape[1], device=item_ids.device).expand_as(item_ids)
                final_indices = positions.masked_fill(~valid, -1).amax(dim=1)
                if torch.any(final_indices < 0):
                    raise ValueError("each sequence must contain at least one non-padding item")
                values = values[torch.arange(values.shape[0], device=values.device), final_indices]
        return self.output_projection(values)
