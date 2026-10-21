import torch
from torch import nn
from torch.nn import functional as F


class Caser(nn.Module):
    architecture_id = "caser"
    def __init__(self, num_items=32, sequence_length=5, embedding_dim=8, num_horizontal_filters=4, num_vertical_filters=4, dropout=0.0):
        super().__init__()
        values = (num_items, sequence_length, embedding_dim, num_horizontal_filters, num_vertical_filters)
        if any(not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("item count, sequence length, embedding width and filter counts must be positive integers")
        if not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.horizontal = nn.ModuleList(
            nn.Conv2d(1, num_horizontal_filters, kernel_size=(height, embedding_dim))
            for height in range(1, sequence_length + 1)
        )
        self.vertical = nn.Conv2d(1, num_vertical_filters, kernel_size=(sequence_length, 1))
        feature_width = num_horizontal_filters * sequence_length + num_vertical_filters * embedding_dim
        self.dropout = nn.Dropout(float(dropout))
        self.projection = nn.Linear(feature_width, embedding_dim)
        self.item_output = nn.Linear(embedding_dim, num_items, bias=False)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, histories):
        if not isinstance(histories, torch.Tensor):
            histories = torch.as_tensor(histories, dtype=torch.long)
        if histories.ndim != 2 or histories.shape[1] != self.sequence_length:
            raise ValueError("histories must have shape (batch, sequence_length)")
        if histories.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("histories must contain integer item identifiers")
        if histories.numel() and (histories.min() < 0 or histories.max() >= self.num_items):
            raise ValueError("item identifier outside configured range")
        embedded = self.item_embedding(histories).unsqueeze(1)
        horizontal = torch.cat([
            F.relu(convolution(embedded)).amax(dim=(2, 3))
            for convolution in self.horizontal
        ], dim=1)
        vertical = F.relu(self.vertical(embedded)).flatten(start_dim=1)
        context = F.relu(self.projection(self.dropout(torch.cat((horizontal, vertical), dim=1))))
        scores = self.item_output(context)
        scores = scores + self.item_bias.weight.squeeze(-1)
        scores = scores.clone()
        scores[:, 0] = torch.finfo(scores.dtype).min
        return scores
