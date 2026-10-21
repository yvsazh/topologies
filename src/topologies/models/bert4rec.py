import torch
from torch import nn
from torch.nn import functional as F


class BERT4Rec(nn.Module):
    architecture_id = "bert4rec"

    def __init__(self, num_items, hidden_size=32, num_heads=4, num_layers=2,
                 max_seq_len=50, dropout=0.1):
        super().__init__()
        if num_items < 1 or hidden_size < 1 or num_heads < 1 or hidden_size % num_heads:
            raise ValueError("num_items and hidden dimensions must be valid and hidden_size divisible by num_heads")
        if num_layers < 1 or max_seq_len < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("num_layers, max_seq_len, or dropout is invalid")
        self.num_items = num_items
        self.mask_token_id = num_items + 1
        self.max_seq_len = max_seq_len
        self.item_embedding = nn.Embedding(num_items + 2, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 4,
                                           dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.prediction_transform = nn.Linear(hidden_size, hidden_size)
        self.output_bias = nn.Parameter(torch.zeros(num_items))

    @property
    def item_output_weight(self):
        return self.item_embedding.weight[1:self.num_items + 1]

    def _valid_positions(self, input_ids, attention_mask):
        valid = input_ids.ne(0)
        if attention_mask is not None:
            attention_mask = torch.as_tensor(attention_mask, device=input_ids.device).bool()
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids")
            valid = valid & attention_mask
        if torch.any(~valid.any(dim=1)):
            raise ValueError("each sequence must contain at least one non-padding position")
        return valid

    def forward(self, input_ids, attention_mask=None):
        input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.item_embedding.weight.device)
        if input_ids.ndim != 2 or input_ids.shape[1] < 1 or input_ids.shape[1] > self.max_seq_len:
            raise ValueError("input_ids must have shape (batch, time) with 1 <= time <= max_seq_len")
        if torch.any(input_ids < 0) or torch.any(input_ids > self.mask_token_id):
            raise ValueError("input_ids contain an invalid item or mask token")
        valid_positions = self._valid_positions(input_ids, attention_mask)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.item_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=~valid_positions)
        hidden = F.gelu(self.prediction_transform(self.norm(hidden)))
        return F.linear(hidden, self.item_output_weight, self.output_bias)

    def masked_item_loss(self, input_ids, target_ids, masked_positions, attention_mask=None):
        input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.item_embedding.weight.device)
        logits = self(input_ids, attention_mask=attention_mask)
        valid_positions = self._valid_positions(input_ids, attention_mask)
        positions = torch.as_tensor(masked_positions, device=logits.device).bool()
        targets = torch.as_tensor(target_ids, device=logits.device, dtype=torch.long)
        if positions.shape != logits.shape[:2] or targets.shape != positions.shape:
            raise ValueError("targets and masked_positions must match batch and time dimensions")
        selected = positions & valid_positions
        if not torch.any(selected):
            raise ValueError("at least one masked non-padding position is required")
        selected_targets = targets[selected]
        if torch.any(selected_targets < 1) or torch.any(selected_targets > self.num_items):
            raise ValueError("masked target ids must be in [1, num_items]")
        return F.cross_entropy(logits[selected], selected_targets - 1)
