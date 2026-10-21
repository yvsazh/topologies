import torch
from torch import nn


class LayoutLM(nn.Module):
    component_id = "layoutlm"
    def __init__(self, vocab_size=30522, hidden_size=768, max_2d_position_embeddings=1024, max_position_embeddings=None):
        super().__init__()
        if max_position_embeddings is not None:
            max_2d_position_embeddings = max_position_embeddings
        if min(vocab_size, hidden_size, max_2d_position_embeddings) < 1:
            raise ValueError("embedding dimensions must be positive")
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.x_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.y_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.h_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.w_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.max_2d_position_embeddings = max_2d_position_embeddings

    def spatial_embeddings(self, boxes):
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("boxes must have shape (batch, sequence, 4)")
        if boxes.is_floating_point() and not torch.equal(boxes, boxes.round()):
            raise ValueError("box coordinates must be integers")
        boxes = boxes.long()
        x0, y0, x1, y1 = boxes.unbind(dim=-1)
        width = x1 - x0
        height = y1 - y0
        indices = (x0, y0, x1, y1, width, height)
        if any(torch.any(index < 0) or torch.any(index >= self.max_2d_position_embeddings) for index in indices):
            raise ValueError("box coordinates and dimensions must be in the embedding range")
        return (self.x_position_embeddings(x0) + self.y_position_embeddings(y0) + self.x_position_embeddings(x1) + self.y_position_embeddings(y1) + self.w_position_embeddings(width) + self.h_position_embeddings(height))

    def forward(self, input_ids, boxes):
        if input_ids.shape != boxes.shape[:2]:
            raise ValueError("input_ids and boxes must have matching batch and sequence dimensions")
        return self.word_embeddings(input_ids) + self.spatial_embeddings(boxes)
