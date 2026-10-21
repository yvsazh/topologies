import torch
from torch import nn
from torch.nn import functional as F


class Mask2Former(nn.Module):
    architecture_id = "mask2former"
    def __init__(self, in_channels=4, num_queries=2, num_classes=2, hidden_dim=8):
        super().__init__()
        if min(in_channels, num_queries, num_classes, hidden_dim) <= 0:
            raise ValueError("channel counts, query count, class count, and hidden width must be positive")
        self.num_queries = num_queries
        self.input_projection = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, hidden_dim) * hidden_dim**-0.5)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.value_projection = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.mask_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask=None):
        features = self.input_projection(x)
        batch, channels, height, width = features.shape
        pixels = features.flatten(2).transpose(1, 2)
        queries = self.query_embeddings.unsqueeze(0).expand(batch, -1, -1)
        mask_embeddings = self.mask_head(queries)
        mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embeddings, features) / channels**0.5
        if attention_mask is None:
            binary_mask = mask_logits.sigmoid() >= 0.5
        else:
            binary_mask = torch.as_tensor(attention_mask, device=x.device, dtype=torch.bool)
            if binary_mask.shape != (batch, self.num_queries, height, width):
                raise ValueError("attention_mask must have shape (batch, queries, height, width)")
        keys = self.key_projection(features).flatten(2).transpose(1, 2)
        values = self.value_projection(features).flatten(2).transpose(1, 2)
        query = self.query_projection(queries)
        scores = torch.matmul(query, keys.transpose(1, 2)) / channels**0.5
        allowed = binary_mask.flatten(2)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attention = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention, values)
        attended = attended + queries
        classes = self.class_head(attended)
        refined_mask_embeddings = self.mask_head(attended)
        masks = torch.einsum("bqc,bchw->bqhw", refined_mask_embeddings, features) / channels**0.5
        return {"pred_logits": classes, "pred_masks": masks}
