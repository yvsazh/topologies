import torch
from torch import nn


class BST(nn.Module):
    architecture_id = "bst"

    def __init__(self, num_items=100, max_sequence_length=50, hidden_dim=32, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        if num_items < 2 or max_sequence_length < 1 or hidden_dim < 1:
            raise ValueError("item count, sequence length, and hidden dimension must be positive")
        if num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_layers < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("num_layers must be positive and dropout must be in [0, 1)")
        self.num_items = num_items
        self.max_sequence_length = max_sequence_length
        self.item_embedding = nn.Embedding(num_items, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_sequence_length + 1, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.click_scorer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, behavior_sequence, target_item, padding_mask=None):
        if behavior_sequence.ndim != 2 or target_item.ndim != 1:
            raise ValueError("behavior_sequence and target_item must have shapes (batch, time) and (batch,)")
        batch, length = behavior_sequence.shape
        if batch != target_item.shape[0] or length < 1 or length > self.max_sequence_length:
            raise ValueError("incompatible batch or sequence length")
        if torch.any(behavior_sequence < 0) or torch.any(behavior_sequence >= self.num_items) or torch.any(target_item < 0) or torch.any(target_item >= self.num_items):
            raise ValueError("item identifiers are outside the embedding vocabulary")
        if padding_mask is None:
            padding_mask = behavior_sequence.eq(0)
        if padding_mask.shape != behavior_sequence.shape:
            raise ValueError("padding_mask must match behavior_sequence")
        items = torch.cat((behavior_sequence, target_item.unsqueeze(1)), dim=1)
        positions = torch.arange(length + 1, device=items.device).unsqueeze(0)
        encoded = self.item_embedding(items) + self.position_embedding(positions)
        mask = torch.cat((padding_mask.to(dtype=torch.bool, device=items.device), torch.zeros(batch, 1, dtype=torch.bool, device=items.device)), dim=1)
        states = self.encoder(encoded, src_key_padding_mask=mask)
        return self.click_scorer(states[:, -1]).squeeze(-1)
